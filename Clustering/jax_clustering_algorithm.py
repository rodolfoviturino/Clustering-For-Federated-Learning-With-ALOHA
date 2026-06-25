"""GPU-oriented JAX helpers for D2D one-hop clustering.

This module is intentionally separate from ``proposed_clustering_algorithm``.
The original implementation is a readable CPU reference for the thesis-style
D2D-SRC heuristic.  The functions here use fixed-shape JAX arrays so the same
experiment can be batched and JIT-compiled on GPU/TPU runtimes.

The default production strategy is ``strategy="dense"`` because it is closest
to the thesis D2D-SRC behavior at ``K=1000``: candidate CHs may absorb any
still-unassigned one-hop neighbor within ``R_D2D``.  ``strategy="grid"`` remains
available for larger sweeps.  Grid cells have side length ``R_D2D / sqrt(2)``,
so any two devices inside the same cell are at most ``R_D2D`` meters apart,
giving a conservative one-hop guarantee without constructing a full all-pairs
graph for very large ``K``.

Important array conventions used throughout this module:

``coords``
    ``float32/float64[K, 2]``. Device x/y coordinates in meters.

``device_ids``
    ``int32[K]``. Stable device identifiers, normally ``0..K-1``.

``cluster_members``
    ``int32[K, Cmax]``. Padded cluster rows. Empty rows contain ``-1``.
    Non-empty rows use column 0 as the cluster head and later columns as
    members.  The number of physically active cluster rows is
    ``number_of_clusters``.

``cluster_sizes``
    ``int32[K]``. Number of valid devices in each row of ``cluster_members``.

The JAX path is not meant to be byte-identical to the CPU D2D-SRC heuristic.
It preserves the model contract that matters for the accelerated experiments:
one-hop CH coverage, maximum cluster size, singleton accounting, and seeded
reproducibility.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

try:  # pragma: no cover - covered by optional-backend tests when JAX exists.
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised on CPU-only envs.
    jax = None
    jnp = None
    _JAX_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial assignment.
    _JAX_IMPORT_ERROR = None


class JaxDeviceBatch(NamedTuple):
    """Fixed-shape device deployment arrays for JAX simulations."""

    device_ids: Any
    coords: Any
    device_angle: Any
    distance_to_bs: Any
    battery: Any
    stats_product: Any
    bs_radius: Any


class JaxClusterResult(NamedTuple):
    """Padded cluster representation produced by the JAX clusterizer."""

    cluster_members: Any
    cluster_sizes: Any
    cluster_heads: Any
    cluster_mask: Any
    number_of_clusters: Any
    clusterized_devices_rate: Any
    singleton_count: Any
    overflow_count: Any
    mode_code: Any
    strategy_code: Any


_MODE_CODES = {
    "no_d2d": 0,
    "geometric": 1,
    "utility": 2,
}

_STRATEGY_CODES = {
    "grid": 1,
    "dense": 2,
}


def _require_jax():
    if jax is None:
        raise ImportError(
            "The JAX GPU backend requires JAX. Install a platform-specific "
            "JAX package, for example `pip install -U \"jax[cuda13]\"` on a "
            "CUDA 13 Colab/runtime, or `pip install jax` for CPU smoke tests."
        ) from _JAX_IMPORT_ERROR


def _key_from_seed(seed):
    """Return a JAX PRNG key from either an integer seed or an existing key."""
    _require_jax()
    if seed is None:
        return jax.random.PRNGKey(0)

    seed_array = jnp.asarray(seed, dtype=jnp.uint32)
    if seed_array.shape == (2,):
        return seed_array
    return jax.random.PRNGKey(seed_array)


def devices_generator_jax(
    number_of_devices: int,
    bs_radius: float,
    seed=None,
    uniform_area: bool = False,
    dtype=None,
) -> JaxDeviceBatch:
    """Generate device deployment arrays directly in JAX.

    Parameters mirror ``devices_generator`` where possible.  The default radius
    model stays thesis-compatible: ``r ~ U(1, R_BS)``.  Set ``uniform_area`` for
    the area-uniform disk ablation.
    """
    _require_jax()
    if number_of_devices < 0:
        raise ValueError("number_of_devices must be non-negative")
    if bs_radius < 1:
        raise ValueError("bs_radius must be at least 1 meter")

    dtype = jnp.float32 if dtype is None else dtype
    key = _key_from_seed(seed)
    angle_key, radius_key, battery_key = jax.random.split(key, 3)

    # Stable device IDs are kept as an explicit array because batched JAX code
    # should not depend on Python dictionaries or implicit row positions.
    device_ids = jnp.arange(number_of_devices, dtype=jnp.int32)

    # Device polar coordinates.  ``angle`` is in radians; ``distance_to_bs`` is
    # in meters and follows the thesis model unless ``uniform_area`` is set.
    device_angle = jax.random.uniform(
        angle_key,
        shape=(number_of_devices,),
        minval=0.0,
        maxval=2.0 * jnp.pi,
        dtype=dtype,
    )
    if uniform_area:
        radius_squared = jax.random.uniform(
            radius_key,
            shape=(number_of_devices,),
            minval=1.0,
            maxval=float(bs_radius) ** 2,
            dtype=dtype,
        )
        distance_to_bs = jnp.sqrt(radius_squared)
    else:
        distance_to_bs = jax.random.uniform(
            radius_key,
            shape=(number_of_devices,),
            minval=1.0,
            maxval=float(bs_radius),
            dtype=dtype,
        )

    # Integer battery percentage, matching the CPU generator's 1..100 range.
    battery = jax.random.randint(
        battery_key,
        shape=(number_of_devices,),
        minval=1,
        maxval=101,
        dtype=jnp.int32,
    )

    x_coord = distance_to_bs * jnp.cos(device_angle)
    y_coord = distance_to_bs * jnp.sin(device_angle)
    coords = jnp.stack((x_coord, y_coord), axis=1)

    return JaxDeviceBatch(
        device_ids=device_ids,
        coords=coords,
        device_angle=device_angle,
        distance_to_bs=distance_to_bs,
        battery=battery,
        stats_product=distance_to_bs * battery.astype(dtype),
        bs_radius=jnp.asarray(bs_radius, dtype=dtype),
    )


def _normalized_inverse_pathloss(distance_to_bs, pathloss_exponent):
    """Return a normalized high-is-good BS channel quality score."""
    raw_quality = 1.0 / jnp.maximum(distance_to_bs, 1.0) ** pathloss_exponent
    return raw_quality / jnp.maximum(jnp.max(raw_quality), 1e-12)


def _rayleigh_outage_success_probability(
    distance_to_bs,
    pathloss_exponent,
    reference_snr,
    snr_threshold,
):
    """Return Rayleigh outage success probability for each BS uplink.

    This score maps the CH-channel term to a physical-layer quantity instead
    of a purely normalized inverse-distance proxy.  For a Rayleigh fading
    channel with average SNR ``gamma_bar``, the probability that instantaneous
    SNR exceeds a threshold is ``exp(-threshold / gamma_bar)``.  The score is
    deterministic here because clustering uses channel statistics, not future
    packet draws.
    """
    dtype = distance_to_bs.dtype
    safe_distance = jnp.maximum(distance_to_bs, jnp.asarray(1.0, dtype=dtype))
    average_snr = jnp.asarray(reference_snr, dtype=dtype) / (
        safe_distance ** jnp.asarray(pathloss_exponent, dtype=dtype)
    )
    return jnp.exp(
        -jnp.asarray(snr_threshold, dtype=dtype)
        / jnp.maximum(average_snr, jnp.asarray(1e-12, dtype=dtype))
    )


def neighbor_counts_tiled_jax(coords, device_radius, tile_size: int = 1024):
    """Count one-hop neighbors using GPU-sized distance tiles.

    ``coords`` is never expanded into a full ``K x K x 2`` tensor.  Instead,
    the function compares all devices against a fixed-size tile of candidates
    and accumulates counts.  This helper is used by utility-based clustering
    and is also useful for profiling GPU memory limits.
    """
    _require_jax()
    if tile_size < 1:
        raise ValueError("tile_size must be positive")

    n_devices = coords.shape[0]
    padded_n = ((n_devices + tile_size - 1) // tile_size) * tile_size
    pad_count = padded_n - n_devices

    # Padding with infinity guarantees padded rows are never within R_D2D of a
    # real device, without needing a separate validity mask in every tile.
    padded_coords = jnp.pad(
        coords,
        pad_width=((0, pad_count), (0, 0)),
        mode="constant",
        constant_values=jnp.inf,
    )
    starts = jnp.arange(0, padded_n, tile_size, dtype=jnp.int32)
    radius_squared = jnp.asarray(device_radius, dtype=coords.dtype) ** 2

    def scan_tile(counts, start):
        # ``dynamic_slice`` requires all start indices to have the same integer
        # dtype.  In x64 mode, a Python literal ``0`` may become int64 while the
        # scanned tile start is int32, so build the second start explicitly.
        zero_start = jnp.asarray(0, dtype=start.dtype)
        tile = jax.lax.dynamic_slice(
            padded_coords,
            (start, zero_start),
            (tile_size, 2),
        )
        delta = coords[:, None, :] - tile[None, :, :]
        within_radius = jnp.sum(delta * delta, axis=-1) <= radius_squared
        counts = counts + jnp.sum(within_radius, axis=1).astype(jnp.int32)
        return counts, None

    counts, _ = jax.lax.scan(
        scan_tile,
        jnp.zeros(n_devices, dtype=jnp.int32),
        starts,
    )
    # Each real device counts itself once; remove that self-neighbor.
    return jnp.maximum(counts - 1, 0)


def _no_d2d_clusters(device_ids, max_devices_per_cluster, strategy: str = "dense"):
    """Return singleton clusters in the same padded format as D2D modes."""
    n_devices = device_ids.shape[0]
    cluster_members = jnp.full(
        (n_devices, max_devices_per_cluster),
        -1,
        dtype=jnp.int32,
    )
    cluster_members = cluster_members.at[:, 0].set(device_ids)
    cluster_sizes = jnp.ones(n_devices, dtype=jnp.int32)
    cluster_mask = jnp.ones(n_devices, dtype=jnp.bool_)
    return JaxClusterResult(
        cluster_members=cluster_members,
        cluster_sizes=cluster_sizes,
        cluster_heads=device_ids,
        cluster_mask=cluster_mask,
        number_of_clusters=jnp.asarray(n_devices, dtype=jnp.int32),
        clusterized_devices_rate=jnp.asarray(0.0, dtype=jnp.float32),
        singleton_count=jnp.asarray(n_devices, dtype=jnp.int32),
        overflow_count=jnp.asarray(0, dtype=jnp.int32),
        mode_code=jnp.asarray(_MODE_CODES["no_d2d"], dtype=jnp.int32),
        strategy_code=jnp.asarray(_STRATEGY_CODES[strategy], dtype=jnp.int32),
    )


def _pack_active_clusters(cluster_members, cluster_sizes):
    """Move non-empty cluster rows to the front of the padded arrays.

    The model simulation indexes cluster rows from ``0`` to
    ``number_of_clusters - 1``.  Repair passes may empty a singleton row after
    moving that device into another cluster, so rows must be compacted before
    returning the final ``JaxClusterResult``.
    """
    n_rows = cluster_sizes.shape[0]
    row_indices = jnp.arange(n_rows, dtype=jnp.int32)
    active = cluster_sizes > 0
    sort_key = jnp.where(active, row_indices, row_indices + n_rows)
    order = jnp.argsort(sort_key, stable=True)

    packed_members = cluster_members[order]
    packed_sizes = cluster_sizes[order]
    number_of_clusters = jnp.sum(active).astype(jnp.int32)
    packed_active = row_indices < number_of_clusters
    packed_members = jnp.where(
        packed_active[:, None],
        packed_members,
        -jnp.ones_like(packed_members),
    )
    packed_sizes = jnp.where(packed_active, packed_sizes, 0)
    return packed_members, packed_sizes, number_of_clusters


def _repair_singleton_join_requests(
    cluster_members,
    cluster_sizes,
    coords,
    device_radius: float,
    max_devices_per_cluster: int,
    repair_passes: int,
):
    """Absorb reachable singletons into nearby CHs with spare capacity.

    This is intentionally a local D2D repair, not a global optimizer.  A
    singleton can move only when it can directly reach an existing CH and that
    CH's cluster has room under ``Cmax``.  The equivalent distributed protocol
    is simple: singleton devices broadcast a join request, nearby CHs with
    spare capacity respond, and the singleton joins the nearest accepting CH.
    """
    cmax = int(max_devices_per_cluster)
    n_rows = cluster_sizes.shape[0]
    row_indices = jnp.arange(n_rows, dtype=jnp.int32)
    radius_squared = jnp.asarray(device_radius, dtype=coords.dtype) ** 2
    passes = jnp.arange(int(repair_passes), dtype=jnp.int32)

    def repair_one_pass(state, _):
        members, sizes = state

        def scan_singleton(inner_state, source_row):
            current_members, current_sizes = inner_state
            singleton_device = current_members[source_row, 0]
            source_is_singleton = current_sizes[source_row] == 1

            safe_heads = jnp.where(current_members[:, 0] >= 0, current_members[:, 0], 0)
            deltas = coords[safe_heads] - coords[jnp.maximum(singleton_device, 0)]
            distance_squared = jnp.sum(deltas * deltas, axis=1)
            eligible_target = (
                source_is_singleton
                & (row_indices != source_row)
                & (current_sizes > 1)
                & (current_sizes < cmax)
                & (distance_squared <= radius_squared)
            )
            best_target = jnp.argmin(jnp.where(eligible_target, distance_squared, jnp.inf))
            has_target = jnp.any(eligible_target)
            safe_target = jnp.where(has_target, best_target, 0)
            insert_position = jnp.where(has_target, current_sizes[safe_target], 0)
            previous_value = current_members[safe_target, insert_position]

            current_members = current_members.at[safe_target, insert_position].set(
                jnp.where(has_target, singleton_device, previous_value)
            )
            current_sizes = current_sizes.at[safe_target].set(
                jnp.where(has_target, current_sizes[safe_target] + 1, current_sizes[safe_target])
            )
            current_members = current_members.at[source_row].set(
                jnp.where(has_target, -jnp.ones(cmax, dtype=jnp.int32), current_members[source_row])
            )
            current_sizes = current_sizes.at[source_row].set(
                jnp.where(has_target, 0, current_sizes[source_row])
            )
            return (current_members, current_sizes), None

        repaired_state, _ = jax.lax.scan(
            scan_singleton,
            (members, sizes),
            row_indices,
        )
        packed_members, packed_sizes, _ = _pack_active_clusters(*repaired_state)
        return (packed_members, packed_sizes), None

    (cluster_members, cluster_sizes), _ = jax.lax.scan(
        repair_one_pass,
        (cluster_members, cluster_sizes),
        passes,
    )
    return cluster_members, cluster_sizes


def _repair_singleton_pair_rotations(
    cluster_members,
    cluster_sizes,
    coords,
    device_radius: float,
    rotation_repair_passes: int,
):
    """Absorb singletons by rotating two-device clusters locally.

    This mirrors a D2D-SRC-style local negotiation: if a singleton cannot reach
    the current CH of a pair but can reach the pair's member, that member can
    become the CH and admit the singleton.  The operation is intentionally
    limited to size-2 clusters because the one-hop validity check is then local
    and simple: the new CH already reaches the old CH, and it must also reach
    the singleton.
    """
    cmax = cluster_members.shape[1]
    if cmax < 3:
        return cluster_members, cluster_sizes

    n_rows = cluster_sizes.shape[0]
    row_indices = jnp.arange(n_rows, dtype=jnp.int32)
    radius_squared = jnp.asarray(device_radius, dtype=coords.dtype) ** 2
    passes = jnp.arange(int(rotation_repair_passes), dtype=jnp.int32)

    def repair_one_pass(state, _):
        members, sizes = state

        def scan_singleton(inner_state, source_row):
            current_members, current_sizes = inner_state
            singleton_device = current_members[source_row, 0]
            source_is_singleton = current_sizes[source_row] == 1

            pair_member = jnp.where(current_members[:, 1] >= 0, current_members[:, 1], 0)
            singleton_coord = coords[jnp.maximum(singleton_device, 0)]
            member_deltas = coords[pair_member] - singleton_coord
            member_distance_squared = jnp.sum(member_deltas * member_deltas, axis=1)
            eligible_target = (
                source_is_singleton
                & (row_indices != source_row)
                & (current_sizes == 2)
                & (member_distance_squared <= radius_squared)
            )
            best_target = jnp.argmin(
                jnp.where(eligible_target, member_distance_squared, jnp.inf)
            )
            has_target = jnp.any(eligible_target)
            safe_target = jnp.where(has_target, best_target, 0)

            old_head = current_members[safe_target, 0]
            new_head = current_members[safe_target, 1]
            rotated_row = jnp.concatenate(
                (
                    jnp.stack((new_head, old_head, singleton_device)).astype(jnp.int32),
                    -jnp.ones(cmax - 3, dtype=jnp.int32),
                )
            )
            current_members = current_members.at[safe_target].set(
                jnp.where(has_target, rotated_row, current_members[safe_target])
            )
            current_sizes = current_sizes.at[safe_target].set(
                jnp.where(has_target, 3, current_sizes[safe_target])
            )
            current_members = current_members.at[source_row].set(
                jnp.where(has_target, -jnp.ones(cmax, dtype=jnp.int32), current_members[source_row])
            )
            current_sizes = current_sizes.at[source_row].set(
                jnp.where(has_target, 0, current_sizes[source_row])
            )
            return (current_members, current_sizes), None

        repaired_state, _ = jax.lax.scan(
            scan_singleton,
            (members, sizes),
            row_indices,
        )
        packed_members, packed_sizes, _ = _pack_active_clusters(*repaired_state)
        return (packed_members, packed_sizes), None

    (cluster_members, cluster_sizes), _ = jax.lax.scan(
        repair_one_pass,
        (cluster_members, cluster_sizes),
        passes,
    )
    return cluster_members, cluster_sizes


def _merge_local_cluster_heads(
    cluster_members,
    cluster_sizes,
    coords,
    device_radius: float,
    merge_passes: int,
):
    """Merge neighboring non-singleton clusters with a local one-hop rule.

    This pass improves cluster quality after the pair-first/repair stages.  It
    is not a centralized graph optimizer: a source cluster may merge into a
    target cluster only when the target CH can directly cover every source
    member and the union still fits inside ``Cmax``.  A real distributed
    interpretation is that nearby CHs exchange a compact cluster summary
    (member IDs/positions or equivalent reachability information), then only
    accept merges that preserve one-hop coverage under the target CH.
    """
    cmax = cluster_members.shape[1]
    if cmax < 2:
        return cluster_members, cluster_sizes

    n_rows = cluster_sizes.shape[0]
    row_indices = jnp.arange(n_rows, dtype=jnp.int32)
    member_positions = jnp.arange(cmax, dtype=jnp.int32)
    radius_squared = jnp.asarray(device_radius, dtype=coords.dtype) ** 2
    passes = jnp.arange(int(merge_passes), dtype=jnp.int32)

    def merge_one_pass(state, _):
        members, sizes = state

        def scan_source(inner_state, source_row):
            current_members, current_sizes = inner_state
            source_size = current_sizes[source_row]
            source_is_mergeable = source_size > 1
            source_members = current_members[source_row]
            safe_source_members = jnp.where(source_members >= 0, source_members, 0)
            source_member_valid = member_positions < source_size

            # Candidate target CHs. Empty rows use device 0 only as a safe
            # gather index; their sizes make them ineligible below.
            target_heads = jnp.where(current_members[:, 0] >= 0, current_members[:, 0], 0)
            source_head = jnp.maximum(current_members[source_row, 0], 0)

            # For every possible target row, test whether its CH can cover all
            # valid members in the source row.  Existing target members are
            # already covered by that target CH by construction.
            deltas_to_source_members = (
                coords[target_heads][:, None, :] - coords[safe_source_members][None, :, :]
            )
            distance_squared_to_source = jnp.sum(
                deltas_to_source_members * deltas_to_source_members,
                axis=-1,
            )
            target_covers_source = jnp.all(
                jnp.where(
                    source_member_valid[None, :],
                    distance_squared_to_source <= radius_squared,
                    True,
                ),
                axis=1,
            )

            combined_sizes = current_sizes + source_size
            eligible_target = (
                source_is_mergeable
                & (row_indices != source_row)
                & (current_sizes > 1)
                & (combined_sizes <= cmax)
                & target_covers_source
            )

            # Among valid local merges, prefer the nearest target CH.  This is
            # deterministic and maps well to a practical CH-to-CH negotiation:
            # the strongest/closest local exchange wins.
            head_deltas = coords[target_heads] - coords[source_head]
            head_distance_squared = jnp.sum(head_deltas * head_deltas, axis=1)
            best_target = jnp.argmin(
                jnp.where(eligible_target, head_distance_squared, jnp.inf)
            )
            has_target = jnp.any(eligible_target)
            safe_target = jnp.where(has_target, best_target, 0)

            target_size = current_sizes[safe_target]
            combined_size = target_size + source_size
            source_insert_indices = jnp.clip(member_positions - target_size, 0, cmax - 1)
            source_values_to_append = source_members[source_insert_indices]
            append_source_member = (
                (member_positions >= target_size) & (member_positions < combined_size)
            )
            merged_target_row = jnp.where(
                append_source_member,
                source_values_to_append,
                current_members[safe_target],
            )

            current_members = current_members.at[safe_target].set(
                jnp.where(has_target, merged_target_row, current_members[safe_target])
            )
            current_sizes = current_sizes.at[safe_target].set(
                jnp.where(has_target, combined_size, current_sizes[safe_target])
            )
            current_members = current_members.at[source_row].set(
                jnp.where(has_target, -jnp.ones(cmax, dtype=jnp.int32), current_members[source_row])
            )
            current_sizes = current_sizes.at[source_row].set(
                jnp.where(has_target, 0, current_sizes[source_row])
            )
            return (current_members, current_sizes), None

        merged_state, _ = jax.lax.scan(
            scan_source,
            (members, sizes),
            row_indices,
        )
        packed_members, packed_sizes, _ = _pack_active_clusters(*merged_state)
        return (packed_members, packed_sizes), None

    (cluster_members, cluster_sizes), _ = jax.lax.scan(
        merge_one_pass,
        (cluster_members, cluster_sizes),
        passes,
    )
    return cluster_members, cluster_sizes


def _split_large_clusters_by_local_reclustering(
    cluster_members,
    cluster_sizes,
    coords,
    device_radius: float,
    split_max_size: int,
):
    """Split large clusters with local one-hop subcluster formation.

    This is a structural ALOHA-compatible ablation: it changes local D2D
    cluster membership before FL rounds start, but it does not reserve slots or
    alter the multichannel ALOHA MAC.  Each large cluster is re-clustered
    internally by scanning its members as candidate local CHs.  A candidate
    absorbs the closest still-unassigned members it can directly cover, up to
    ``split_max_size``.  Members that cannot safely join another subcluster
    become singleton rows, preserving the one-hop invariant.
    """
    cmax = cluster_members.shape[1]
    split_max_size = int(split_max_size)
    if split_max_size <= 0 or split_max_size >= cmax:
        return cluster_members, cluster_sizes

    n_rows = cluster_sizes.shape[0]
    row_indices = jnp.arange(n_rows, dtype=jnp.int32)
    member_positions = jnp.arange(cmax, dtype=jnp.int32)
    radius_squared = jnp.asarray(device_radius, dtype=coords.dtype) ** 2

    def copy_or_skip_row(state, source_row):
        out_members, out_sizes, out_row = state
        source_size = cluster_sizes[source_row]
        has_row = source_size > 0
        safe_out_row = jnp.minimum(out_row, n_rows - 1)
        out_members = out_members.at[safe_out_row].set(
            jnp.where(has_row, cluster_members[source_row], out_members[safe_out_row])
        )
        out_sizes = out_sizes.at[safe_out_row].set(
            jnp.where(has_row, source_size, out_sizes[safe_out_row])
        )
        out_row = out_row + has_row.astype(jnp.int32)
        return out_members, out_sizes, out_row

    def split_row(state, source_row):
        out_members, out_sizes, out_row = state
        source_members = cluster_members[source_row]
        source_size = cluster_sizes[source_row]
        safe_source_members = jnp.where(source_members >= 0, source_members, 0)
        valid_source_position = member_positions < source_size

        def scan_candidate(candidate_state, candidate_position):
            current_members, current_sizes, current_row, assigned = candidate_state
            candidate_unassigned = (
                valid_source_position[candidate_position]
                & (~assigned[candidate_position])
            )
            candidate_device = safe_source_members[candidate_position]
            deltas = coords[safe_source_members] - coords[candidate_device]
            distance_squared = jnp.sum(deltas * deltas, axis=1)
            eligible = (
                valid_source_position
                & (~assigned)
                & (distance_squared <= radius_squared)
            )
            eligible_count = jnp.sum(eligible).astype(jnp.int32)
            selected_count = jnp.minimum(
                eligible_count,
                jnp.asarray(split_max_size, dtype=jnp.int32),
            )
            distance_key = jnp.where(eligible, distance_squared, jnp.inf)
            selected_positions = jnp.argsort(distance_key, stable=True)
            selected_mask = member_positions < selected_count
            selected_devices = source_members[selected_positions]
            split_members = jnp.where(selected_mask, selected_devices, -1)
            selected_position_mask = jnp.zeros(cmax, dtype=jnp.bool_).at[
                selected_positions
            ].set(selected_mask)

            create_row = candidate_unassigned & (selected_count > 0)
            safe_current_row = jnp.minimum(current_row, n_rows - 1)
            current_members = current_members.at[safe_current_row].set(
                jnp.where(
                    create_row,
                    split_members,
                    current_members[safe_current_row],
                )
            )
            current_sizes = current_sizes.at[safe_current_row].set(
                jnp.where(
                    create_row,
                    selected_count,
                    current_sizes[safe_current_row],
                )
            )
            assigned = assigned | (create_row & selected_position_mask)
            current_row = current_row + create_row.astype(jnp.int32)
            return (current_members, current_sizes, current_row, assigned), None

        (out_members, out_sizes, out_row, _), _ = jax.lax.scan(
            scan_candidate,
            (
                out_members,
                out_sizes,
                out_row,
                jnp.zeros(cmax, dtype=jnp.bool_),
            ),
            member_positions,
        )
        return out_members, out_sizes, out_row

    def scan_source(state, source_row):
        source_size = cluster_sizes[source_row]
        should_split = source_size > split_max_size
        next_state = jax.lax.cond(
            should_split,
            lambda current_state: split_row(current_state, source_row),
            lambda current_state: copy_or_skip_row(current_state, source_row),
            state,
        )
        return next_state, None

    empty_members = jnp.full_like(cluster_members, -1)
    empty_sizes = jnp.zeros_like(cluster_sizes)
    initial_state = (
        empty_members,
        empty_sizes,
        jnp.asarray(0, dtype=jnp.int32),
    )
    (split_members, split_sizes, _), _ = jax.lax.scan(
        scan_source,
        initial_state,
        row_indices,
    )
    split_members, split_sizes, _ = _pack_active_clusters(split_members, split_sizes)
    return split_members, split_sizes


def _cluster_head_quality_scores(
    devices: JaxDeviceBatch,
    device_radius,
    pathloss_exponent: float,
    tile_size: int,
    degree_weight: float,
    channel_weight: float,
    battery_weight: float,
    channel_score_mode: str,
    reference_snr: float,
    snr_threshold: float,
):
    """Return a high-is-good score for CH rotation candidates.

    The score is intentionally built from signals that are plausible before an
    FL update is transmitted:

    - D2D degree: local neighbor count inside ``R_D2D``.  A high-degree CH is
      likely to be a stronger local representative and robust to small changes
      in neighborhood membership.
    - BS channel quality: either normalized inverse pathloss or Rayleigh outage
      success probability from device to BS.  The Rayleigh option ties the CH
      channel score to a collision-free decoding metric, while the default
      inverse-pathloss mode preserves earlier experiments.
    - Battery: local battery percentage.  Higher battery makes repeated CH duty
      more realistic.

    These are control-plane or locally measurable quantities.  The function
    does not inspect labels, model error, or future upload outcomes.
    """
    dtype = devices.coords.dtype
    neighbor_counts = neighbor_counts_tiled_jax(
        devices.coords,
        device_radius,
        tile_size=tile_size,
    ).astype(dtype)
    degree_score = neighbor_counts / jnp.maximum(jnp.max(neighbor_counts), 1.0)
    if channel_score_mode == "rayleigh_outage":
        channel_score = _rayleigh_outage_success_probability(
            devices.distance_to_bs,
            pathloss_exponent=pathloss_exponent,
            reference_snr=reference_snr,
            snr_threshold=snr_threshold,
        )
    else:
        channel_score = _normalized_inverse_pathloss(
            devices.distance_to_bs,
            pathloss_exponent,
        )
    battery_score = devices.battery.astype(dtype) / 100.0

    return (
        jnp.asarray(degree_weight, dtype=dtype) * degree_score
        + jnp.asarray(channel_weight, dtype=dtype) * channel_score
        + jnp.asarray(battery_weight, dtype=dtype) * battery_score
    )


def _rotate_cluster_heads_by_quality(
    cluster_members,
    cluster_sizes,
    coords,
    device_radius: float,
    quality_score,
):
    """Move the best valid member to CH position inside each cluster.

    This is a CH-selection refinement, not a cluster-membership optimizer.  The
    member set of every row stays unchanged.  A candidate member may become CH
    only if it can directly cover every valid member in that row, preserving the
    one-hop D2D invariant used by the thesis.  In a real deployment, the same
    operation can be negotiated inside the cluster after local discovery: the
    cluster keeps its members but elects the member with the best D2D/BS/battery
    score among candidates that satisfy one-hop coverage.
    """
    cmax = cluster_members.shape[1]
    dtype = coords.dtype
    member_positions = jnp.arange(cmax, dtype=jnp.int32)
    valid_member = member_positions[None, :] < cluster_sizes[:, None]
    safe_members = jnp.where(cluster_members >= 0, cluster_members, 0)

    candidate_coords = coords[safe_members]
    deltas = candidate_coords[:, :, None, :] - candidate_coords[:, None, :, :]
    distance_squared = jnp.sum(deltas * deltas, axis=-1)
    radius_squared = jnp.asarray(device_radius, dtype=dtype) ** 2

    # valid_cover[row, candidate_pos] is true only when that candidate member
    # reaches every real member in the same row.  Padding columns are ignored.
    valid_cover = jnp.all(
        jnp.where(
            valid_member[:, None, :],
            distance_squared <= radius_squared,
            True,
        ),
        axis=2,
    )
    candidate_valid = valid_member & valid_cover
    candidate_scores = jnp.where(
        candidate_valid,
        quality_score[safe_members],
        -jnp.inf,
    )
    best_position = jnp.argmax(candidate_scores, axis=1).astype(jnp.int32)

    # Move the selected CH to column 0 and shift the previous prefix one slot to
    # the right.  Example: [0, 1, 2, -1] with best_position=2 becomes
    # [2, 0, 1, -1].  This keeps every member exactly once.
    source_positions = jnp.where(
        member_positions[None, :] == 0,
        best_position[:, None],
        jnp.where(
            member_positions[None, :] <= best_position[:, None],
            member_positions[None, :] - 1,
            member_positions[None, :],
        ),
    )
    source_positions = jnp.clip(source_positions, 0, cmax - 1)
    rotated_members = jnp.take_along_axis(cluster_members, source_positions, axis=1)
    return jnp.where((cluster_sizes > 0)[:, None], rotated_members, cluster_members)


def _rank_devices_for_mode(
    devices: JaxDeviceBatch,
    device_radius,
    clustering_mode: str,
    pathloss_exponent: float,
    tile_size: int,
):
    """Return a per-device rank used to order candidate CHs inside a cell."""
    n_devices = devices.device_ids.shape[0]
    if clustering_mode == "geometric":
        # Lowest device ID wins ties, giving a deterministic geometry-only
        # ordering that is easy to audit against the reference implementation.
        return devices.device_ids

    if clustering_mode != "utility":
        raise ValueError("clustering_mode must be 'no_d2d', 'geometric', or 'utility'")

    # Utility mode prefers high local degree, high battery, and good BS channel
    # quality.  The components are normalized so weights are dimensionless.
    neighbor_counts = neighbor_counts_tiled_jax(
        devices.coords,
        device_radius,
        tile_size=tile_size,
    ).astype(devices.coords.dtype)
    max_degree = jnp.maximum(jnp.max(neighbor_counts), 1.0)
    degree_score = neighbor_counts / max_degree
    battery_score = devices.battery.astype(devices.coords.dtype) / 100.0
    channel_score = _normalized_inverse_pathloss(
        devices.distance_to_bs,
        pathloss_exponent,
    )
    utility_score = 0.45 * degree_score + 0.30 * battery_score + 0.25 * channel_score

    # ``argsort(-score)`` gives best device first.  We convert that ordering
    # into an integer rank per device so the grid grouping can sort by
    # ``(cell_key, rank)`` using one composite integer key.
    score_order = jnp.argsort(-utility_score, stable=True)
    return jnp.zeros(n_devices, dtype=jnp.int32).at[score_order].set(
        jnp.arange(n_devices, dtype=jnp.int32)
    )


def _dense_greedy_clusters(
    devices: JaxDeviceBatch,
    device_radius: float,
    max_devices_per_cluster: int,
    min_devices_per_cluster: int,
    clustering_mode: str,
    pathloss_exponent: float,
    tile_size: int,
    repair_passes: int,
    initial_cluster_size: int,
    rotation_repair_passes: int,
    merge_passes: int,
    cluster_head_selection_mode: str,
    cluster_head_degree_weight: float,
    cluster_head_channel_weight: float,
    cluster_head_battery_weight: float,
    cluster_head_channel_score_mode: str,
    cluster_head_reference_snr: float,
    cluster_head_snr_threshold: float,
    cluster_split_mode: str,
    cluster_split_max_size: int,
) -> JaxClusterResult:
    """Build one-hop clusters from the full radius graph.

    This strategy is closer to the original thesis D2D-SRC behavior than the
    grid strategy: a candidate CH may absorb any still-unassigned device within
    ``R_D2D``, not only devices that happen to fall in the same grid cell.  It
    is more expensive than grid clustering, but it is the right choice for
    thesis-scale reproduction at ``K=1000``.
    """
    cmax = int(max_devices_per_cluster)
    initial_cluster_size = max(1, min(int(initial_cluster_size), cmax))
    n_devices = devices.device_ids.shape[0]
    radius_squared = jnp.asarray(device_radius, dtype=devices.coords.dtype) ** 2

    if clustering_mode == "geometric":
        # Dense geometric mode should not use device ID as the primary CH
        # priority.  A high-degree device can cover more one-hop neighbors and
        # is less likely to strand nearby devices as singletons.  This is still
        # geometry-only: the score uses only the R_D2D neighbor graph, with the
        # device ID used only as a deterministic tie-breaker.
        neighbor_counts = neighbor_counts_tiled_jax(
            devices.coords,
            device_radius,
            tile_size=tile_size,
        ).astype(devices.coords.dtype)
        rank_by_device = (
            -neighbor_counts
            + devices.device_ids.astype(devices.coords.dtype) / float(n_devices + 1)
        )
    else:
        rank_by_device = _rank_devices_for_mode(
            devices,
            device_radius=device_radius,
            clustering_mode=clustering_mode,
            pathloss_exponent=pathloss_exponent,
            tile_size=tile_size,
        )
    head_order = jnp.argsort(rank_by_device, stable=True)
    member_positions = jnp.arange(cmax, dtype=jnp.int32)

    def scan_head(state, head):
        assigned, cluster_members, cluster_sizes, row_index = state
        head = head.astype(jnp.int32)
        head_is_available = ~assigned[head]

        deltas = devices.coords - devices.coords[head]
        distance_squared = jnp.sum(deltas * deltas, axis=1)
        candidates = (~assigned) & (distance_squared <= radius_squared)
        candidate_count = jnp.sum(candidates).astype(jnp.int32)
        selected_count = jnp.minimum(candidate_count, initial_cluster_size)

        # Sort by distance so the CH keeps the closest one-hop members.  Stable
        # sorting preserves device-ID order for exact distance ties.
        distance_key = jnp.where(candidates, distance_squared, jnp.inf)
        selected_devices = jnp.argsort(distance_key, stable=True)[:cmax].astype(jnp.int32)
        selected_mask = member_positions < selected_count
        row_members = jnp.where(selected_mask, selected_devices, -1)

        # Keep singleton accounting separate from the CH scan.  The scan only
        # emits real multi-device D2D clusters; remaining devices are converted
        # to singleton rows after all CH candidates have been considered.
        starts_cluster = head_is_available & (selected_count > 1)
        newly_assigned = jnp.zeros(n_devices, dtype=jnp.bool_).at[selected_devices].set(
            starts_cluster & selected_mask
        )
        assigned = assigned | newly_assigned
        cluster_members = cluster_members.at[row_index].set(
            jnp.where(starts_cluster, row_members, cluster_members[row_index])
        )
        cluster_sizes = cluster_sizes.at[row_index].set(
            jnp.where(starts_cluster, selected_count, cluster_sizes[row_index])
        )
        row_index = row_index + starts_cluster.astype(jnp.int32)
        return (assigned, cluster_members, cluster_sizes, row_index), None

    initial_state = (
        jnp.zeros(n_devices, dtype=jnp.bool_),
        jnp.full((n_devices, cmax), -1, dtype=jnp.int32),
        jnp.zeros(n_devices, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    final_state, _ = jax.lax.scan(scan_head, initial_state, head_order)
    assigned, cluster_members, cluster_sizes, number_of_clusters = final_state

    # Devices that never joined a multi-device one-hop cluster are appended as
    # singleton clusters.  This keeps the padded representation complete while
    # avoiding premature singleton assignment during the greedy pass.
    unassigned_count = jnp.sum(~assigned).astype(jnp.int32)
    unassigned_sort_key = jnp.where(
        ~assigned,
        devices.device_ids,
        devices.device_ids + n_devices,
    )
    unassigned_order = jnp.argsort(unassigned_sort_key, stable=True).astype(jnp.int32)
    row_numbers = jnp.arange(n_devices, dtype=jnp.int32)
    singleton_offsets = row_numbers - number_of_clusters
    singleton_rows = (singleton_offsets >= 0) & (singleton_offsets < unassigned_count)
    singleton_devices = unassigned_order[
        jnp.clip(singleton_offsets, 0, n_devices - 1)
    ]
    cluster_members = cluster_members.at[:, 0].set(
        jnp.where(singleton_rows, singleton_devices, cluster_members[:, 0])
    )
    cluster_sizes = jnp.where(singleton_rows, 1, cluster_sizes)
    cluster_members, cluster_sizes, number_of_clusters = _pack_active_clusters(
        cluster_members,
        cluster_sizes,
    )

    if repair_passes > 0:
        cluster_members, cluster_sizes = _repair_singleton_join_requests(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=devices.coords,
            device_radius=device_radius,
            max_devices_per_cluster=max_devices_per_cluster,
            repair_passes=repair_passes,
        )
        cluster_members, cluster_sizes, number_of_clusters = _pack_active_clusters(
            cluster_members,
            cluster_sizes,
        )

    if rotation_repair_passes > 0:
        cluster_members, cluster_sizes = _repair_singleton_pair_rotations(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=devices.coords,
            device_radius=device_radius,
            rotation_repair_passes=rotation_repair_passes,
        )
        cluster_members, cluster_sizes, number_of_clusters = _pack_active_clusters(
            cluster_members,
            cluster_sizes,
        )

        if repair_passes > 0:
            cluster_members, cluster_sizes = _repair_singleton_join_requests(
                cluster_members=cluster_members,
                cluster_sizes=cluster_sizes,
                coords=devices.coords,
                device_radius=device_radius,
                max_devices_per_cluster=max_devices_per_cluster,
                repair_passes=repair_passes,
            )
            cluster_members, cluster_sizes, number_of_clusters = _pack_active_clusters(
                cluster_members,
                cluster_sizes,
            )

    if merge_passes > 0:
        cluster_members, cluster_sizes = _merge_local_cluster_heads(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=devices.coords,
            device_radius=device_radius,
            merge_passes=merge_passes,
        )
        cluster_members, cluster_sizes, number_of_clusters = _pack_active_clusters(
            cluster_members,
            cluster_sizes,
        )

    if cluster_split_mode == "max_size":
        cluster_members, cluster_sizes = _split_large_clusters_by_local_reclustering(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=devices.coords,
            device_radius=device_radius,
            split_max_size=cluster_split_max_size,
        )
        cluster_members, cluster_sizes, number_of_clusters = _pack_active_clusters(
            cluster_members,
            cluster_sizes,
        )

    if cluster_head_selection_mode == "quality":
        quality_score = _cluster_head_quality_scores(
            devices=devices,
            device_radius=device_radius,
            pathloss_exponent=pathloss_exponent,
            tile_size=tile_size,
            degree_weight=cluster_head_degree_weight,
            channel_weight=cluster_head_channel_weight,
            battery_weight=cluster_head_battery_weight,
            channel_score_mode=cluster_head_channel_score_mode,
            reference_snr=cluster_head_reference_snr,
            snr_threshold=cluster_head_snr_threshold,
        )
        cluster_members = _rotate_cluster_heads_by_quality(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=devices.coords,
            device_radius=device_radius,
            quality_score=quality_score,
        )

    cluster_mask = cluster_sizes > 0
    cluster_heads = jnp.where(cluster_mask, cluster_members[:, 0], -1)
    singleton_count = jnp.sum((cluster_sizes == 1) & cluster_mask).astype(jnp.int32)
    clusterized_devices_rate = (
        1.0 - singleton_count.astype(devices.coords.dtype) / max(float(n_devices), 1.0)
    ) * 100.0

    return JaxClusterResult(
        cluster_members=cluster_members,
        cluster_sizes=cluster_sizes,
        cluster_heads=cluster_heads,
        cluster_mask=cluster_mask,
        number_of_clusters=number_of_clusters,
        clusterized_devices_rate=clusterized_devices_rate,
        singleton_count=singleton_count,
        overflow_count=jnp.asarray(0, dtype=jnp.int32),
        mode_code=jnp.asarray(_MODE_CODES[clustering_mode], dtype=jnp.int32),
        strategy_code=jnp.asarray(_STRATEGY_CODES["dense"], dtype=jnp.int32),
    )


def clusterizer_jax(
    devices: JaxDeviceBatch,
    device_radius: float,
    max_devices_per_cluster: int,
    min_devices_per_cluster: int = 1,
    clustering_mode: str = "geometric",
    strategy: str = "dense",
    pathloss_exponent: float = 2.0,
    tile_size: int = 1024,
    repair_passes: int = 1,
    initial_cluster_size: int = 2,
    rotation_repair_passes: int = 1,
    merge_passes: int = 1,
    cluster_head_selection_mode: str = "first",
    cluster_head_degree_weight: float = 0.40,
    cluster_head_channel_weight: float = 0.40,
    cluster_head_battery_weight: float = 0.20,
    cluster_head_channel_score_mode: str = "inverse_pathloss",
    cluster_head_reference_snr: float = 100000.0,
    cluster_head_snr_threshold: float = 1.0,
    cluster_split_mode: str = "none",
    cluster_split_max_size: int = 0,
) -> JaxClusterResult:
    """Build padded one-hop clusters with GPU-friendly fixed-shape arrays.

    ``strategy="dense"`` compares candidate CHs against every unassigned
    device and is closest to the thesis D2D-SRC behavior.  ``strategy="grid"``
    is more conservative and much cheaper for very large sweeps.
    """
    _require_jax()
    if device_radius <= 0:
        raise ValueError("device_radius must be positive")
    if max_devices_per_cluster < 1:
        raise ValueError("max_devices_per_cluster must be at least 1")
    if min_devices_per_cluster < 1:
        raise ValueError("min_devices_per_cluster must be at least 1")
    if min_devices_per_cluster > max_devices_per_cluster:
        raise ValueError("min_devices_per_cluster cannot exceed max_devices_per_cluster")
    if repair_passes < 0:
        raise ValueError("repair_passes must be non-negative")
    if initial_cluster_size < 1:
        raise ValueError("initial_cluster_size must be at least 1")
    if rotation_repair_passes < 0:
        raise ValueError("rotation_repair_passes must be non-negative")
    if merge_passes < 0:
        raise ValueError("merge_passes must be non-negative")
    if cluster_head_selection_mode not in {"first", "quality"}:
        raise ValueError("cluster_head_selection_mode must be 'first' or 'quality'")
    if cluster_split_mode not in {"none", "max_size"}:
        raise ValueError("cluster_split_mode must be 'none' or 'max_size'")
    if cluster_split_mode == "max_size":
        if cluster_split_max_size < 1:
            raise ValueError(
                "cluster_split_max_size must be positive when cluster_split_mode=max_size"
            )
        if cluster_split_max_size > max_devices_per_cluster:
            raise ValueError("cluster_split_max_size cannot exceed max_devices_per_cluster")
    if cluster_head_degree_weight < 0.0:
        raise ValueError("cluster_head_degree_weight must be non-negative")
    if cluster_head_channel_weight < 0.0:
        raise ValueError("cluster_head_channel_weight must be non-negative")
    if cluster_head_battery_weight < 0.0:
        raise ValueError("cluster_head_battery_weight must be non-negative")
    if cluster_head_channel_score_mode not in {"inverse_pathloss", "rayleigh_outage"}:
        raise ValueError(
            "cluster_head_channel_score_mode must be 'inverse_pathloss' or "
            "'rayleigh_outage'"
        )
    if cluster_head_reference_snr <= 0.0:
        raise ValueError("cluster_head_reference_snr must be positive")
    if cluster_head_snr_threshold < 0.0:
        raise ValueError("cluster_head_snr_threshold must be non-negative")
    if (
        cluster_head_selection_mode == "quality"
        and cluster_head_degree_weight
        + cluster_head_channel_weight
        + cluster_head_battery_weight
        <= 0.0
    ):
        raise ValueError(
            "quality CH selection requires at least one positive CH score weight"
        )
    if strategy not in _STRATEGY_CODES:
        raise ValueError("strategy must be 'dense' or 'grid'")

    cmax = int(max_devices_per_cluster)
    n_devices = devices.device_ids.shape[0]
    if clustering_mode == "no_d2d":
        return _no_d2d_clusters(devices.device_ids, cmax, strategy=strategy)

    if strategy == "dense":
        return _dense_greedy_clusters(
            devices=devices,
            device_radius=device_radius,
            max_devices_per_cluster=max_devices_per_cluster,
            min_devices_per_cluster=min_devices_per_cluster,
            clustering_mode=clustering_mode,
            pathloss_exponent=pathloss_exponent,
            tile_size=tile_size,
            repair_passes=repair_passes,
            initial_cluster_size=initial_cluster_size,
            rotation_repair_passes=rotation_repair_passes,
            merge_passes=merge_passes,
            cluster_head_selection_mode=cluster_head_selection_mode,
            cluster_head_degree_weight=cluster_head_degree_weight,
            cluster_head_channel_weight=cluster_head_channel_weight,
            cluster_head_battery_weight=cluster_head_battery_weight,
            cluster_head_channel_score_mode=cluster_head_channel_score_mode,
            cluster_head_reference_snr=cluster_head_reference_snr,
            cluster_head_snr_threshold=cluster_head_snr_threshold,
            cluster_split_mode=cluster_split_mode,
            cluster_split_max_size=cluster_split_max_size,
        )

    # Grid cell side in meters.  Any two points inside the same cell are within
    # the diagonal length, which equals R_D2D.
    cell_side = jnp.asarray(device_radius, dtype=devices.coords.dtype) / jnp.sqrt(
        jnp.asarray(2.0, dtype=devices.coords.dtype)
    )
    shifted_coords = devices.coords + devices.bs_radius
    cell_xy = jnp.floor(shifted_coords / cell_side).astype(jnp.int32)
    cell_xy = jnp.maximum(cell_xy, 0)

    # Flatten 2-D cell coordinates into a sortable key.  The grid width is
    # derived from the deployment radius and cell side, with a small safety
    # margin for boundary points.
    grid_width = (
        jnp.ceil((2.0 * devices.bs_radius + cell_side) / cell_side).astype(jnp.int32)
        + 2
    )
    cell_key = cell_xy[:, 0] + cell_xy[:, 1] * grid_width

    rank_by_device = _rank_devices_for_mode(
        devices,
        device_radius=device_radius,
        clustering_mode=clustering_mode,
        pathloss_exponent=pathloss_exponent,
        tile_size=tile_size,
    )
    sort_key = cell_key * (n_devices + 1) + rank_by_device
    order = jnp.argsort(sort_key, stable=True)

    sorted_devices = devices.device_ids[order]
    sorted_cell_key = cell_key[order]

    def assign_cell_position(carry, current_cell_key):
        previous_cell_key, cell_index, position_in_cell = carry
        same_cell = current_cell_key == previous_cell_key
        starts_new_cell = (previous_cell_key < 0) | (~same_cell)
        next_cell_index = jnp.where(starts_new_cell, cell_index + 1, cell_index)
        next_position = jnp.where(starts_new_cell, 0, position_in_cell + 1)
        return (
            current_cell_key,
            next_cell_index,
            next_position,
        ), (
            next_cell_index,
            next_position,
        )

    # First identify contiguous sorted cells and each device's offset inside
    # its cell.  Cell rows are logical groups before Cmax/Cmin chunking.
    _, (cell_indices, positions_in_cell) = jax.lax.scan(
        assign_cell_position,
        (
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        ),
        sorted_cell_key,
    )

    cell_sizes = jnp.zeros(n_devices, dtype=jnp.int32).at[cell_indices].max(
        positions_in_cell + 1
    )
    cell_has_devices = cell_sizes > 0

    # For a dense cell, split into balanced chunks instead of greedily filling
    # Cmax and leaving a tiny tail.  Example: 12 devices with Cmax=10 becomes
    # 6+6, not 10+2.  If a whole cell is below Cmin and has more than one
    # device, the conservative fallback is singleton rows.
    split_cell_to_singletons = (
        (cell_sizes > 1) & (cell_sizes < int(min_devices_per_cluster))
    )
    chunks_for_dense_cell = (cell_sizes + cmax - 1) // cmax
    chunks_per_cell = jnp.where(
        cell_has_devices,
        jnp.where(split_cell_to_singletons, cell_sizes, chunks_for_dense_cell),
        0,
    )
    cell_row_offsets = jnp.cumsum(chunks_per_cell) - chunks_per_cell

    sizes_for_sorted_devices = cell_sizes[cell_indices]
    chunks_for_sorted_devices = chunks_per_cell[cell_indices]
    singleton_fallback_for_sorted_devices = split_cell_to_singletons[cell_indices]
    balanced_chunk_size = jnp.where(
        singleton_fallback_for_sorted_devices,
        1,
        (sizes_for_sorted_devices + chunks_for_sorted_devices - 1)
        // chunks_for_sorted_devices,
    )

    chunk_in_cell = positions_in_cell // balanced_chunk_size
    final_member_positions = positions_in_cell - chunk_in_cell * balanced_chunk_size
    final_row_indices = cell_row_offsets[cell_indices] + chunk_in_cell

    cluster_members = jnp.full((n_devices, cmax), -1, dtype=jnp.int32)
    cluster_members = cluster_members.at[final_row_indices, final_member_positions].set(
        sorted_devices
    )
    cluster_sizes = jnp.zeros(n_devices, dtype=jnp.int32).at[final_row_indices].max(
        final_member_positions + 1
    )
    if cluster_split_mode == "max_size":
        cluster_members, cluster_sizes = _split_large_clusters_by_local_reclustering(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=devices.coords,
            device_radius=device_radius,
            split_max_size=cluster_split_max_size,
        )
    if cluster_head_selection_mode == "quality":
        quality_score = _cluster_head_quality_scores(
            devices=devices,
            device_radius=device_radius,
            pathloss_exponent=pathloss_exponent,
            tile_size=tile_size,
            degree_weight=cluster_head_degree_weight,
            channel_weight=cluster_head_channel_weight,
            battery_weight=cluster_head_battery_weight,
            channel_score_mode=cluster_head_channel_score_mode,
            reference_snr=cluster_head_reference_snr,
            snr_threshold=cluster_head_snr_threshold,
        )
        cluster_members = _rotate_cluster_heads_by_quality(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=devices.coords,
            device_radius=device_radius,
            quality_score=quality_score,
        )
    cluster_mask = cluster_sizes > 0
    cluster_heads = jnp.where(cluster_mask, cluster_members[:, 0], -1)

    number_of_clusters = jnp.sum(cluster_mask).astype(jnp.int32)
    singleton_count = jnp.sum((cluster_sizes == 1) & cluster_mask).astype(jnp.int32)
    clusterized_devices_rate = (
        1.0 - singleton_count.astype(devices.coords.dtype) / max(float(n_devices), 1.0)
    ) * 100.0

    return JaxClusterResult(
        cluster_members=cluster_members,
        cluster_sizes=cluster_sizes,
        cluster_heads=cluster_heads,
        cluster_mask=cluster_mask,
        number_of_clusters=number_of_clusters,
        clusterized_devices_rate=clusterized_devices_rate,
        singleton_count=singleton_count,
        # Grid chunking creates more rows instead of overflowing Cmax.
        overflow_count=jnp.asarray(0, dtype=jnp.int32),
        mode_code=jnp.asarray(_MODE_CODES[clustering_mode], dtype=jnp.int32),
        strategy_code=jnp.asarray(_STRATEGY_CODES[strategy], dtype=jnp.int32),
    )


def jax_clusters_to_python(clusters: JaxClusterResult):
    """Convert padded JAX clusters to the legacy list-of-lists format."""
    members = np.asarray(clusters.cluster_members)
    sizes = np.asarray(clusters.cluster_sizes)
    output = []
    for row, size in zip(members, sizes):
        if int(size) > 0:
            output.append([int(device) for device in row[: int(size)]])
    return output


def validate_jax_cluster_result(
    clusters: JaxClusterResult,
    coords,
    device_radius: float,
    max_devices_per_cluster: int,
    n_devices: int | None = None,
):
    """Host-side validator for tests and CPU/JAX comparison reports."""
    members = np.asarray(clusters.cluster_members)
    sizes = np.asarray(clusters.cluster_sizes)
    coords = np.asarray(coords)
    if n_devices is None:
        n_devices = coords.shape[0]

    seen = []
    for row_index, size in enumerate(sizes):
        size = int(size)
        if size == 0:
            continue
        if size > max_devices_per_cluster:
            raise ValueError(f"cluster {row_index} exceeds Cmax={max_devices_per_cluster}")

        row = members[row_index, :size].astype(int)
        head = row[0]
        for member in row[1:]:
            distance = float(np.linalg.norm(coords[head] - coords[member]))
            if distance > device_radius + 1e-5:
                raise ValueError(
                    f"cluster {row_index} member {member} is {distance:.6f} m "
                    f"from CH {head}, above R_D2D={device_radius}"
                )
        seen.extend(row.tolist())

    expected = set(range(n_devices))
    actual = set(seen)
    if len(seen) != len(actual):
        raise ValueError("a device appears in more than one JAX cluster")
    if actual != expected:
        raise ValueError(
            "JAX cluster device set mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return True
