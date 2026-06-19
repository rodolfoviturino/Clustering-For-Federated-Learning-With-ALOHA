"""Device generation and D2D-SRC clustering helpers.

Clusters are represented as simple Python lists because the clustering phase is
easier to inspect and debug this way:

    [cluster_head, member_1, member_2, ...]

The JAX model simulation later converts this representation to padded numeric
arrays. Keeping the readable representation here is intentional; this module is
where geometry experiments can still be inspected on CPU.
"""

from math import dist

import numpy as np


def _make_rng(seed=None):
    """Return a NumPy RNG while also accepting an already-created Generator."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _coord_from(source, device):
    """Read coordinates from either the generator dictionary or a raw array/list."""
    if isinstance(source, dict):
        value = source[device]
        if isinstance(value, dict):
            return value["x_y_coord"]
        return value
    return source[device]


def _distance_to_head(devices_information_dict, head, device):
    """Distance from a candidate/member device to a cluster head."""
    return dist(
        _coord_from(devices_information_dict, head),
        _coord_from(devices_information_dict, device),
    )


def devices_proximity_checker(
    device_x_coords,
    device_y_coords,
    devices_communication_radius,
):
    """Return True when two devices are within the D2D communication radius."""
    # The bounding-box check is a cheap pre-filter before calculating the exact
    # Euclidean distance. It matters because clustering repeatedly compares
    # many pairs of devices.
    if (
        (device_y_coords[0] - devices_communication_radius)
        <= device_x_coords[0]
        <= (device_y_coords[0] + devices_communication_radius)
        and (device_y_coords[1] - devices_communication_radius)
        <= device_x_coords[1]
        <= (device_y_coords[1] + devices_communication_radius)
    ):
        return dist(device_x_coords, device_y_coords) <= devices_communication_radius
    return False


def devices_generator(number_of_devices, bs_radius, seed=None, uniform_area=False):
    """Generate device metadata around the BS.

    By default this follows the thesis model: radius ~ U(1, R_BS) and
    theta ~ U(0, 2*pi). Set uniform_area=True for an area-uniform disk
    deployment ablation.
    """
    if number_of_devices < 0:
        raise ValueError("number_of_devices must be non-negative")
    if bs_radius < 1:
        raise ValueError("bs_radius must be at least 1 meter")

    rng = _make_rng(seed)

    # The dictionary is convenient for the clustering code because device IDs
    # are used as stable keys. The list is kept for backward compatibility with
    # the original notebook, even though the accelerated simulation now consumes
    # fixed-shape JAX cluster arrays.
    devices_information_dict = {}
    devices_information_list = []

    for device in range(number_of_devices):
        device_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        if uniform_area:
            # Area-uniform sampling needs the square-root transform. It is not
            # the thesis default, but it is useful for deployment sensitivity
            # tests where spatial density should be uniform over the disk.
            distance_from_device_to_BS = float(
                np.sqrt(rng.uniform(1.0, float(bs_radius) ** 2))
            )
        else:
            # Thesis-compatible placement: radius itself is uniformly sampled.
            distance_from_device_to_BS = float(rng.uniform(1.0, float(bs_radius)))

        x_coord = distance_from_device_to_BS * np.cos(device_angle)
        y_coord = distance_from_device_to_BS * np.sin(device_angle)

        # Battery is retained as scenario metadata. The current D2D-SRC
        # clustering heuristic does not use it, but future CH scoring can.
        device_battery = int(rng.integers(1, 101))
        stats_product = distance_from_device_to_BS * device_battery

        devices_information_dict[device] = {
            "device_angle": device_angle,
            "distance_from_device_to_BS": distance_from_device_to_BS,
            "x_y_coord": (float(x_coord), float(y_coord)),
            "device_battery": device_battery,
            "stats_product": float(stats_product),
        }

        devices_information_list.append(
            [
                device,
                device_angle,
                distance_from_device_to_BS,
                device_battery,
            ]
        )

    return devices_information_dict, devices_information_list


def validate_clusters(clusters, coords, r_d2d, cmax, n_devices):
    """Validate D2D-SRC cluster invariants.

    Raises ValueError with a concrete message when an invariant is violated.
    Returns True otherwise.
    """
    if n_devices < 0:
        raise ValueError("n_devices must be non-negative")
    if cmax < 1:
        raise ValueError("cmax must be at least 1")

    # flat_devices is used for uniqueness/all-devices checks after validating
    # each individual cluster's local constraints.
    flat_devices = []
    for cluster_index, cluster in enumerate(clusters):
        if len(cluster) == 0:
            raise ValueError(f"cluster {cluster_index} is empty")
        if len(cluster) > cmax:
            raise ValueError(
                f"cluster {cluster_index} has size {len(cluster)}, above Cmax={cmax}"
            )

        # One-hop D2D-SRC means every member must be directly reachable from
        # the CH. Member-to-member distance is intentionally not required.
        clusterhead = cluster[0]
        for member in cluster[1:]:
            distance_to_ch = dist(
                _coord_from(coords, clusterhead),
                _coord_from(coords, member),
            )
            if distance_to_ch > r_d2d + 1e-9:
                raise ValueError(
                    "cluster "
                    f"{cluster_index} member {member} is {distance_to_ch:.6f} m "
                    f"from CH {clusterhead}, above R_D2D={r_d2d}"
                )
        flat_devices.extend(cluster)

    expected = set(range(n_devices))
    actual = set(flat_devices)
    if len(flat_devices) != len(actual):
        raise ValueError("a device appears in more than one cluster")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"cluster device set mismatch; missing={missing}, extra={extra}")

    return True


def _cluster_devices(clusterhead, members):
    return [clusterhead] + list(members)


def _head_can_cover(head, devices, device_radius, devices_information_dict):
    """Check whether head can serve every device in a proposed merged cluster."""
    for device in devices:
        if device == head:
            continue
        if _distance_to_head(devices_information_dict, head, device) > device_radius + 1e-9:
            return False
    return True


def _head_total_distance(head, devices, devices_information_dict):
    """Tie-breaker score: smaller total member distance gives a tighter cluster."""
    total = 0.0
    for device in devices:
        if device != head:
            total += _distance_to_head(devices_information_dict, head, device)
    return total


def _best_merge(head_a, head_b, dict_of_clusters, device_radius, max_devices_per_cluster, devices_information_dict):
    """Return the safest merge between two clusters, or None if impossible.

    Only the two existing CHs are considered as possible heads. This matches the
    thesis description of CH-to-CH negotiation and avoids silently rotating to a
    member during the merge step.
    """
    devices_a = _cluster_devices(head_a, dict_of_clusters[head_a])
    devices_b = _cluster_devices(head_b, dict_of_clusters[head_b])
    combined_devices = list(dict.fromkeys(devices_a + devices_b))

    # Cmax must be checked before evaluating reachability. The previous code
    # merged first and checked later, which allowed oversized clusters.
    if len(combined_devices) > max_devices_per_cluster:
        return None

    feasible_heads = []
    for candidate_head in (head_a, head_b):
        # Bidirectional merge check: either CH may become the final head, as
        # long as every device in the union is one hop from that CH.
        if _head_can_cover(
            candidate_head,
            combined_devices,
            device_radius,
            devices_information_dict,
        ):
            feasible_heads.append(candidate_head)

    if not feasible_heads:
        return None

    # If both heads can cover the merged cluster, keep the tighter one. This is
    # a deterministic choice that also tends to reduce future D2D link distance.
    chosen_head = min(
        feasible_heads,
        key=lambda head: _head_total_distance(head, combined_devices, devices_information_dict),
    )
    return chosen_head, [device for device in combined_devices if device != chosen_head]


def devices_clusterizer(
    device_radius,
    max_devices_per_cluster,
    min_devices_per_cluster,
    devices_information_dict,
    seed=None,
    shuffle_clusters=True,
    validate=True,
):
    """Cluster devices with the D2D-SRC heuristic.

    The function returns clusters as lists where index 0 is the cluster head.

    The high-level stages mirror the thesis flow:
    1. create initial pairs;
    2. absorb unclustered devices;
    3. reposition CHs in size-2 clusters;
    4. merge compatible clusters;
    5. rebalance small clusters;
    6. convert remaining unclustered devices to singleton CHs.
    """
    if device_radius <= 0:
        raise ValueError("device_radius must be positive")
    if max_devices_per_cluster < 1:
        raise ValueError("max_devices_per_cluster must be at least 1")
    if min_devices_per_cluster < 1:
        raise ValueError("min_devices_per_cluster must be at least 1")
    if min_devices_per_cluster > max_devices_per_cluster:
        raise ValueError("min_devices_per_cluster cannot exceed max_devices_per_cluster")

    rng = _make_rng(seed)

    # Step 1: create initial two-device clusters.
    #
    # The shuffled request order approximates the unsynchronized D2D request
    # process. Without this shuffle, lower device IDs would systematically act
    # first and bias cluster formation.
    temporary_list_of_available_devices = list(devices_information_dict.keys())
    rng.shuffle(temporary_list_of_available_devices)

    # list_of_unclustered_devices stores devices that could not find any
    # available neighbor during step 1. dict_of_clusters maps CH -> members.
    list_of_unclustered_devices = []
    dict_of_clusters = {}

    while temporary_list_of_available_devices:
        reference_device = temporary_list_of_available_devices.pop()
        reference_device_coords = devices_information_dict[reference_device]["x_y_coord"]

        # Neighbor candidates are still unclaimed devices within D2D range. We
        # keep distances so the selected initial pair is local and stable.
        neighbors = []
        for analysed_device in temporary_list_of_available_devices:
            analysed_device_coords = devices_information_dict[analysed_device]["x_y_coord"]
            if devices_proximity_checker(
                device_x_coords=reference_device_coords,
                device_y_coords=analysed_device_coords,
                devices_communication_radius=device_radius,
            ):
                neighbors.append(
                    (
                        analysed_device,
                        dist(reference_device_coords, analysed_device_coords),
                    )
                )

        if neighbors:
            closest_device = min(neighbors, key=lambda item: item[1])[0]
            temporary_list_of_available_devices.remove(closest_device)
            dict_of_clusters[reference_device] = [closest_device]
        else:
            # The device remains eligible for later absorption by an existing
            # CH, so it is not converted to a singleton cluster yet.
            list_of_unclustered_devices.append(reference_device)

    # Step 2: absorb unclustered devices into the nearest available CH.
    for unclustered_device in list(list_of_unclustered_devices):
        candidates = []
        for clusterhead, members in dict_of_clusters.items():
            # A cluster with Cmax devices cannot receive another member.
            if len(members) + 1 >= max_devices_per_cluster:
                continue
            distance_to_ch = _distance_to_head(
                devices_information_dict,
                clusterhead,
                unclustered_device,
            )
            if distance_to_ch <= device_radius + 1e-9:
                candidates.append((clusterhead, distance_to_ch))

        if candidates:
            # Choosing the nearest CH removes dictionary-order bias and usually
            # shortens the D2D member-to-CH link.
            chosen_clusterhead = min(candidates, key=lambda item: item[1])[0]
            dict_of_clusters[chosen_clusterhead].append(unclustered_device)
            list_of_unclustered_devices.remove(unclustered_device)

    # Step 3: reposition CHs in size-2 clusters to absorb one more device.
    for unclustered_device in list(list_of_unclustered_devices):
        candidates = []
        for clusterhead, members in list(dict_of_clusters.items()):
            # This stage is intentionally restricted to size-2 clusters. Trying
            # all possible CH rotations in larger clusters grows the search
            # space quickly and was not part of the thesis algorithm.
            if len(members) + 1 != 2:
                continue
            other_cluster_device = members[0]
            distance_to_other = _distance_to_head(
                devices_information_dict,
                other_cluster_device,
                unclustered_device,
            )
            if distance_to_other <= device_radius + 1e-9:
                candidates.append((clusterhead, other_cluster_device, distance_to_other))

        if candidates:
            # The former member becomes CH because it can reach both the old CH
            # and the unclustered device, creating a valid size-3 cluster.
            old_clusterhead, new_clusterhead, _ = min(candidates, key=lambda item: item[2])
            dict_of_clusters[new_clusterhead] = [old_clusterhead, unclustered_device]
            dict_of_clusters.pop(old_clusterhead)
            list_of_unclustered_devices.remove(unclustered_device)

    # Step 4: merge clusters. Restart after each merge to avoid mutating an
    # iterator and skipping candidates.
    changed = True
    while changed:
        changed = False
        clusterheads = list(dict_of_clusters.keys())

        for index, reference_clusterhead in enumerate(clusterheads):
            if reference_clusterhead not in dict_of_clusters:
                continue

            best_candidate = None
            for analysed_clusterhead in clusterheads[index + 1 :]:
                if analysed_clusterhead not in dict_of_clusters:
                    continue
                # CHs negotiate only when they are themselves within D2D range.
                if _distance_to_head(
                    devices_information_dict,
                    reference_clusterhead,
                    analysed_clusterhead,
                ) > device_radius + 1e-9:
                    continue

                merge = _best_merge(
                    reference_clusterhead,
                    analysed_clusterhead,
                    dict_of_clusters,
                    device_radius,
                    max_devices_per_cluster,
                    devices_information_dict,
                )
                if merge is None:
                    continue

                chosen_head, merged_members = merge
                # Prefer the closest CH pair first, then the tighter resulting
                # cluster. This keeps merge behavior deterministic for a seed.
                pair_distance = _distance_to_head(
                    devices_information_dict,
                    reference_clusterhead,
                    analysed_clusterhead,
                )
                score = (
                    pair_distance,
                    _head_total_distance(
                        chosen_head,
                        [chosen_head] + merged_members,
                        devices_information_dict,
                    ),
                    chosen_head,
                )
                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (score, reference_clusterhead, analysed_clusterhead, chosen_head, merged_members)

            if best_candidate is not None:
                _, head_a, head_b, chosen_head, merged_members = best_candidate
                removed_head = head_b if chosen_head == head_a else head_a
                dict_of_clusters[chosen_head] = merged_members
                dict_of_clusters.pop(removed_head, None)
                # Restart because the cluster dictionary changed; continuing
                # with stale indices could skip possible merges.
                changed = True
                break

    # Step 5: balance small clusters by moving the nearest valid member from
    # larger neighboring clusters. Donor CHs are not moved in this heuristic.
    for clusterhead_that_needs in list(dict_of_clusters.keys()):
        while (
            clusterhead_that_needs in dict_of_clusters
            and len(dict_of_clusters[clusterhead_that_needs]) + 1 < min_devices_per_cluster
            and len(dict_of_clusters[clusterhead_that_needs]) + 1 < max_devices_per_cluster
        ):
            best_transfer = None

            for clusterhead_that_can_give, donor_members in dict_of_clusters.items():
                if clusterhead_that_can_give == clusterhead_that_needs:
                    continue
                # Donor must remain at or above the balancing threshold after
                # giving away one member.
                if len(donor_members) + 1 <= min_devices_per_cluster:
                    continue
                # The 2*R_D2D CH-to-CH pre-filter follows the thesis intuition:
                # if two CHs are farther than this, no member can be within one
                # hop of both cluster neighborhoods.
                if _distance_to_head(
                    devices_information_dict,
                    clusterhead_that_needs,
                    clusterhead_that_can_give,
                ) > (2.0 * device_radius) + 1e-9:
                    continue

                for member in donor_members:
                    # Only members move during balancing. Moving donor CHs
                    # would require validating and possibly rotating the donor
                    # cluster, which is a different algorithmic stage.
                    distance_to_receiver = _distance_to_head(
                        devices_information_dict,
                        clusterhead_that_needs,
                        member,
                    )
                    if distance_to_receiver <= device_radius + 1e-9:
                        candidate = (
                            distance_to_receiver,
                            clusterhead_that_can_give,
                            member,
                        )
                        if best_transfer is None or candidate < best_transfer:
                            best_transfer = candidate

            if best_transfer is None:
                # No valid donor/member can improve this cluster.
                break

            _, donor_clusterhead, transferred_member = best_transfer
            dict_of_clusters[donor_clusterhead].remove(transferred_member)
            dict_of_clusters[clusterhead_that_needs].append(transferred_member)

    # Remaining unclustered devices become singleton CHs. They count as
    # unclustered for the clustering-rate metric but are valid clusters for HFL.
    final_clusters_generated = [[device] for device in list_of_unclustered_devices]
    for clusterhead, members in dict_of_clusters.items():
        final_clusters_generated.append([clusterhead] + list(members))

    if shuffle_clusters:
        rng.shuffle(final_clusters_generated)

    if validate:
        validate_clusters(
            clusters=final_clusters_generated,
            coords=devices_information_dict,
            r_d2d=device_radius,
            cmax=max_devices_per_cluster,
            n_devices=len(devices_information_dict),
        )

    return final_clusters_generated
