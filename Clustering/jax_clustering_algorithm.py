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
) -> JaxClusterResult:
    """Build one-hop clusters from the full radius graph.

    This strategy is closer to the original thesis D2D-SRC behavior than the
    grid strategy: a candidate CH may absorb any still-unassigned device within
    ``R_D2D``, not only devices that happen to fall in the same grid cell.  It
    is more expensive than grid clustering, but it is the right choice for
    thesis-scale reproduction at ``K=1000``.
    """
    cmax = int(max_devices_per_cluster)
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
        selected_count = jnp.minimum(candidate_count, cmax)

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
    number_of_clusters = number_of_clusters + unassigned_count

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
