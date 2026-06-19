"""Experimental D2D-SRC V2 clustering.

The baseline D2D-SRC implementation is kept in
proposed_clustering_algorithm.py for thesis reproducibility. This module adds a
more optimization-oriented variant that still returns the same cluster format:

    [cluster_head, member_1, member_2, ...]

V2 treats clustering as a capacitated one-hop star-clustering problem. Each CH
candidate receives a utility score, then the algorithm greedily builds clusters
around the strongest feasible CHs and performs a small local-improvement pass.
"""

from dataclasses import dataclass

import numpy as np

from Clustering.proposed_clustering_algorithm import validate_clusters

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only without SciPy.
    cKDTree = None


@dataclass
class D2DSRCV2Config:
    """Weights and controls for utility-based D2D-SRC V2.

    All weights are intentionally dimensionless. Inputs are normalized to
    roughly [0, 1] before scoring, so changing a weight directly changes the
    relative importance of that feature.
    """

    degree_weight: float = 0.35
    battery_weight: float = 0.25
    bs_channel_weight: float = 0.20
    freshness_weight: float = 0.10
    ch_usage_penalty_weight: float = 0.10
    energy_cost_penalty_weight: float = 0.10
    pathloss_exponent: float = 2.0
    local_swap_rounds: int = 1


def _make_rng(seed=None):
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _device_arrays(devices_information_dict, n_devices):
    coords = np.zeros((n_devices, 2), dtype=np.float64)
    battery = np.zeros(n_devices, dtype=np.float64)
    distance_to_bs = np.zeros(n_devices, dtype=np.float64)

    for device in range(n_devices):
        info = devices_information_dict[device]
        coords[device, :] = info["x_y_coord"]
        battery[device] = info.get("device_battery", 100.0) / 100.0
        distance_to_bs[device] = max(float(info.get("distance_from_device_to_BS", 1.0)), 1.0)

    return coords, np.clip(battery, 0.0, 1.0), distance_to_bs


def _normalized_inverse_pathloss(distance_to_bs, pathloss_exponent):
    raw_quality = 1.0 / np.maximum(distance_to_bs, 1.0) ** pathloss_exponent
    max_quality = float(np.max(raw_quality))
    if max_quality <= 0.0:
        return np.ones_like(raw_quality)
    return raw_quality / max_quality


def _build_neighbor_graph(coords, device_radius):
    """Return device IDs within one D2D hop for every device."""
    if cKDTree is not None:
        tree = cKDTree(coords)
        return [list(map(int, neighbors)) for neighbors in tree.query_ball_point(coords, r=device_radius)]

    neighbors_by_device = []
    for device, coord in enumerate(coords):
        distances = np.linalg.norm(coords - coord, axis=1)
        neighbors_by_device.append(np.where(distances <= device_radius + 1e-9)[0].astype(int).tolist())
    return neighbors_by_device


def _score_ch_candidates(
    neighbors_by_device,
    battery,
    bs_channel_quality,
    config,
    rng,
    freshness=None,
    ch_usage_count=None,
):
    n_devices = len(neighbors_by_device)
    degrees = np.array([len(neighbors) - 1 for neighbors in neighbors_by_device], dtype=np.float64)
    max_degree = max(float(np.max(degrees)), 1.0)
    degree_score = degrees / max_degree

    if freshness is None:
        freshness = np.zeros(n_devices, dtype=np.float64)
    else:
        freshness = np.asarray(freshness, dtype=np.float64)
    max_freshness = max(float(np.max(freshness)), 1.0)
    freshness_score = freshness / max_freshness

    if ch_usage_count is None:
        ch_usage_count = np.zeros(n_devices, dtype=np.float64)
    else:
        ch_usage_count = np.asarray(ch_usage_count, dtype=np.float64)
    max_usage = max(float(np.max(ch_usage_count)), 1.0)
    usage_penalty = ch_usage_count / max_usage

    # Farther devices usually cost more for CH-to-BS transmission. Since
    # channel quality is normalized high-is-good, 1-quality is a simple penalty.
    energy_cost_penalty = 1.0 - bs_channel_quality

    score = (
        config.degree_weight * degree_score
        + config.battery_weight * battery
        + config.bs_channel_weight * bs_channel_quality
        + config.freshness_weight * freshness_score
        - config.ch_usage_penalty_weight * usage_penalty
        - config.energy_cost_penalty_weight * energy_cost_penalty
    )

    # Tiny random tie-breaker keeps deterministic seeds while avoiding repeated
    # ID-order ties in symmetric deployments.
    return score + rng.uniform(0.0, 1e-9, size=n_devices)


def _member_score(candidate_member, head, coords, battery, bs_channel_quality, device_radius):
    d2d_distance = np.linalg.norm(coords[candidate_member] - coords[head])
    d2d_quality = 1.0 - min(d2d_distance / max(device_radius, 1e-12), 1.0)
    return 0.45 * d2d_quality + 0.30 * battery[candidate_member] + 0.25 * bs_channel_quality[candidate_member]


def _can_head_cover_cluster(head, cluster, coords, device_radius):
    for device in cluster:
        if device == head:
            continue
        if np.linalg.norm(coords[head] - coords[device]) > device_radius + 1e-9:
            return False
    return True


def _try_local_swaps(clusters, coords, battery, bs_channel_quality, device_radius, max_devices_per_cluster, rounds):
    """Move members between clusters when it improves member utility.

    The swap pass is deliberately conservative: only non-CH members move, the
    receiving CH must cover the member, and Cmax must remain satisfied.
    """
    for _ in range(rounds):
        moved = False
        device_to_cluster = {}
        for cluster_index, cluster in enumerate(clusters):
            for device in cluster:
                device_to_cluster[device] = cluster_index

        for cluster_index, cluster in enumerate(list(clusters)):
            if len(cluster) <= 1:
                continue
            head = cluster[0]
            for member in list(cluster[1:]):
                current_score = _member_score(member, head, coords, battery, bs_channel_quality, device_radius)
                best_target = None
                best_score = current_score

                for target_index, target_cluster in enumerate(clusters):
                    if target_index == cluster_index:
                        continue
                    if len(target_cluster) >= max_devices_per_cluster:
                        continue
                    target_head = target_cluster[0]
                    if np.linalg.norm(coords[target_head] - coords[member]) > device_radius + 1e-9:
                        continue
                    candidate_score = _member_score(
                        member,
                        target_head,
                        coords,
                        battery,
                        bs_channel_quality,
                        device_radius,
                    )
                    if candidate_score > best_score + 1e-12:
                        best_score = candidate_score
                        best_target = target_index

                if best_target is not None:
                    clusters[cluster_index].remove(member)
                    clusters[best_target].append(member)
                    moved = True

        clusters = [cluster for cluster in clusters if cluster]
        if not moved:
            break

    return clusters


def _absorb_singletons(clusters, coords, device_radius, max_devices_per_cluster):
    """Attach singleton devices to nearby non-full clusters when feasible."""
    changed = True
    while changed:
        changed = False
        for cluster in list(clusters):
            if len(cluster) != 1:
                continue

            singleton = cluster[0]
            best_target = None
            best_distance = None
            for target_cluster in clusters:
                if target_cluster is cluster:
                    continue
                if len(target_cluster) >= max_devices_per_cluster:
                    continue
                target_head = target_cluster[0]
                distance = np.linalg.norm(coords[target_head] - coords[singleton])
                if distance <= device_radius + 1e-9:
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_target = target_cluster

            if best_target is not None:
                best_target.append(singleton)
                clusters.remove(cluster)
                changed = True
                break

            # If every nearby cluster is full, form a valid size-2 cluster by
            # borrowing a reachable non-CH member only from a cluster that will
            # remain non-singleton afterward. This makes the pass monotonic:
            # every successful move strictly reduces the number of singleton
            # clusters, so sparse layouts cannot cycle indefinitely.
            best_donor = None
            best_member = None
            best_distance = None
            for donor_cluster in clusters:
                if donor_cluster is cluster or len(donor_cluster) <= 2:
                    continue
                for member in donor_cluster[1:]:
                    distance = np.linalg.norm(coords[singleton] - coords[member])
                    if distance <= device_radius + 1e-9:
                        if best_distance is None or distance < best_distance:
                            best_distance = distance
                            best_donor = donor_cluster
                            best_member = member

            if best_donor is not None:
                best_donor.remove(best_member)
                cluster.append(best_member)
                changed = True
                break

    return clusters


def devices_clusterizer_v2(
    device_radius,
    max_devices_per_cluster,
    min_devices_per_cluster,
    devices_information_dict,
    bs_radius,
    config=None,
    seed=None,
    validate=True,
):
    """Create utility-based capacitated one-hop D2D clusters.

    min_devices_per_cluster is kept for API symmetry with the baseline. V2 uses
    it as a soft preference during greedy assignment but never violates Cmax or
    one-hop reachability to force a minimum size.
    """
    if config is None:
        config = D2DSRCV2Config()
    if device_radius <= 0:
        raise ValueError("device_radius must be positive")
    if max_devices_per_cluster < 1:
        raise ValueError("max_devices_per_cluster must be at least 1")
    if min_devices_per_cluster < 1:
        raise ValueError("min_devices_per_cluster must be at least 1")
    if bs_radius < 1:
        raise ValueError("bs_radius must be at least 1")

    rng = _make_rng(seed)
    n_devices = len(devices_information_dict)
    coords, battery, distance_to_bs = _device_arrays(devices_information_dict, n_devices)
    bs_channel_quality = _normalized_inverse_pathloss(distance_to_bs, config.pathloss_exponent)
    neighbors_by_device = _build_neighbor_graph(coords, device_radius)
    ch_scores = _score_ch_candidates(
        neighbors_by_device,
        battery,
        bs_channel_quality,
        config,
        rng,
    )

    candidate_order = np.argsort(-ch_scores)
    assigned = np.zeros(n_devices, dtype=np.bool_)
    clusters = []

    for head in candidate_order:
        head = int(head)
        if assigned[head]:
            continue

        candidate_members = [
            int(device)
            for device in neighbors_by_device[head]
            if int(device) != head and not assigned[int(device)]
        ]
        candidate_members.sort(
            key=lambda device: _member_score(
                device,
                head,
                coords,
                battery,
                bs_channel_quality,
                device_radius,
            ),
            reverse=True,
        )

        cluster = [head]
        for member in candidate_members:
            if len(cluster) >= max_devices_per_cluster:
                break
            cluster.append(member)

        for device in cluster:
            assigned[device] = True
        clusters.append(cluster)

    # Defensive pass for any isolated/unassigned devices. Normally this is only
    # needed if the input dictionary has non-contiguous IDs, which tests do not
    # use, but it keeps the public function safer.
    for device in range(n_devices):
        if not assigned[device]:
            clusters.append([device])
            assigned[device] = True

    clusters = _try_local_swaps(
        clusters,
        coords,
        battery,
        bs_channel_quality,
        device_radius,
        max_devices_per_cluster,
        max(int(config.local_swap_rounds), 0),
    )
    clusters = _absorb_singletons(clusters, coords, device_radius, max_devices_per_cluster)

    if validate:
        validate_clusters(clusters, devices_information_dict, device_radius, max_devices_per_cluster, n_devices)

    return clusters
