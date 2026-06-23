"""JAX implementation of the HFL/ALOHA simulation model.

This module is the accelerated model path.  It is written around JAX arrays,
``jax.lax.scan`` for the FL time loop, and fixed-shape padded clusters so the
same code can run on CPU, GPU, or TPU.

The simulation follows the same thesis-level structure as the original model:

* synthetic linear regression data are generated for ``K`` mobile devices;
* each device computes a local gradient/update for the current model state;
* polling, fixed ALOHA, and optimized ALOHA decide which updates reach the BS;
* D2D variants aggregate active cluster members at the CH before CH-to-BS ALOHA;
* the thesis-style SGD step applies ``w <- w - u1 * gradient`` by default;
* ``normalize_by_k=True`` enables the more conservative ``gradient / K`` variant.
* optional optimized-access floors can keep optimized ALOHA from underusing
  channels after D2D aggregation drives update norms very small;
* optimized D2D can use load-controlled utility policies based on aggregate
  norm, active aggregate size, and freshness.
* the enhanced max-weight D2D policy uses a dual threshold that can be
  broadcast by the BS, while each CH computes its own local utility score.
* the hybrid utility D2D policy keeps smooth load allocation, but discounts
  CH aggregates that are directionally redundant with recent successful uploads.
* the adaptive-diversity D2D policy keeps the same distributed load controller,
  but shifts from early high-utility aggregates to late diversity/freshness.
* the AoI-aware utility D2D policy keeps the same load controller, but gives an
  extra bounded priority bonus to clusters whose successful-upload age is in
  the stale tail of the current D2D cluster population.
* the AoI-floor utility D2D policy is a more conservative freshness variant:
  it first computes the base utility access probability and only raises stale
  clusters to a bounded minimum probability when they would otherwise be almost
  ignored.
* optional CH-to-BS link realism can make a collision-free CH upload succeed
  with probability derived from the elected CH's BS channel quality and battery.
* optional device-to-BS link realism can apply the same physical decoding model
  to the non-D2D polling/fixed/optimized curves, making comparisons against
  D2D physically symmetric when requested.
* optional dynamic energy drain tracks a separate battery vector for each of
  the six curves; direct devices, D2D members, and CHs pay configurable costs
  for attempted transmissions, and battery-aware channel models see the updated
  energy in the next iteration.
* optional energy-aware D2D CH rotation can re-elect each cluster's CH during
  the FL trajectory using BS channel quality, current battery, and a stability
  term while preserving one-hop coverage and fixed cluster membership.
* ``dtype=jnp.float64`` is recommended when reproducing very small thesis error
  norms; ``float32`` is faster but floors optimized curves near single-precision
  machine accuracy.

The implementation is trace-oriented: it runs once to ``max_iterations_t`` and
returns metrics after every iteration or after selected checkpoints.  This is
the right shape for fast sweeps because it avoids rerunning the first 199
iterations separately for every ``t`` value in ``1..200``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from Clustering.jax_clustering_algorithm import JaxClusterResult

try:  # pragma: no cover - covered when JAX is installed.
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised on this local env.
    jax = None
    jnp = None
    _JAX_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial assignment.
    _JAX_IMPORT_ERROR = None


class JaxTraceResult(NamedTuple):
    """Metric arrays produced by ``error_calculator_trace_jax``.

    Arrays are checkpoint-indexed.  The six scenario columns follow the fixed
    order used by the CSV writer: polling, fixed ALOHA, optimized ALOHA,
    polling+D2D, fixed+D2D, optimized+D2D.
    """

    clusterized_devices_rate: Any
    error_norms: Any
    successful_uploads: Any
    successful_clusterhead_uploads: Any
    mean_battery: Any
    mean_clusterhead_battery: Any
    mean_energy_used: Any
    energy_efficiency: Any
    mean_clusterhead_energy_used: Any
    mean_aoi: Any
    peak_aoi: Any
    p75_aoi: Any
    p90_aoi: Any
    p95_aoi: Any
    stale_fraction_50: Any
    stale_fraction_75: Any
    stale_fraction_100: Any
    checkpoints: Any


D2D_ENERGY_EFFICIENCY_PROFILES = {
    "performance": (0.85, 0.10, 0.05),
    "balanced": (0.65, 0.25, 0.10),
    "eco": (0.45, 0.45, 0.10),
}


def _require_jax():
    if jax is None:
        raise ImportError(
            "The simulation model now requires JAX. Install `jax` for CPU "
            "smoke tests or a CUDA-enabled extra such as `jax[cuda13]` for "
            "Colab/GPU execution."
        ) from _JAX_IMPORT_ERROR


def _key_from_seed(seed):
    _require_jax()
    if seed is None:
        return jax.random.PRNGKey(0)
    seed_array = jnp.asarray(seed, dtype=jnp.uint32)
    if seed_array.shape == (2,):
        return seed_array
    return jax.random.PRNGKey(seed_array)


def prepare_clusters_for_jax(clusters, max_cluster_size=None) -> JaxClusterResult:
    """Convert legacy list clusters to padded JAX cluster arrays.

    This helper keeps notebooks/scripts ergonomic while ensuring the model
    kernel receives fixed-shape arrays.  The returned rows use the same contract
    as ``clusterizer_jax``: column 0 is the CH, unused cells are ``-1``.
    """
    _require_jax()
    normalized_clusters = [list(map(int, cluster)) for cluster in clusters]
    if not normalized_clusters:
        raise ValueError("clusters must contain at least one cluster")

    if max_cluster_size is None:
        max_cluster_size = max(len(cluster) for cluster in normalized_clusters)
    if max_cluster_size < 1:
        raise ValueError("max_cluster_size must be at least 1")

    number_of_clusters = len(normalized_clusters)
    cluster_members_np = np.full(
        (number_of_clusters, max_cluster_size),
        -1,
        dtype=np.int32,
    )
    cluster_sizes_np = np.zeros(number_of_clusters, dtype=np.int32)

    for cluster_index, cluster in enumerate(normalized_clusters):
        if not cluster:
            raise ValueError(f"cluster {cluster_index} is empty")
        if len(cluster) > max_cluster_size:
            raise ValueError(
                f"cluster {cluster_index} has size {len(cluster)}, "
                f"above max_cluster_size={max_cluster_size}"
            )
        cluster_sizes_np[cluster_index] = len(cluster)
        cluster_members_np[cluster_index, : len(cluster)] = cluster

    cluster_members = jnp.asarray(cluster_members_np, dtype=jnp.int32)
    cluster_sizes = jnp.asarray(cluster_sizes_np, dtype=jnp.int32)
    cluster_mask = cluster_sizes > 0
    cluster_heads = jnp.where(cluster_mask, cluster_members[:, 0], -1)
    singleton_count = jnp.sum((cluster_sizes == 1) & cluster_mask).astype(jnp.int32)
    n_devices = max(max(cluster) for cluster in normalized_clusters) + 1
    clusterized_devices_rate = (
        1.0 - singleton_count.astype(jnp.float32) / max(float(n_devices), 1.0)
    ) * 100.0

    return JaxClusterResult(
        cluster_members=cluster_members,
        cluster_sizes=cluster_sizes,
        cluster_heads=cluster_heads,
        cluster_mask=cluster_mask,
        number_of_clusters=jnp.asarray(number_of_clusters, dtype=jnp.int32),
        clusterized_devices_rate=clusterized_devices_rate,
        singleton_count=singleton_count,
        overflow_count=jnp.asarray(0, dtype=jnp.int32),
        mode_code=jnp.asarray(-1, dtype=jnp.int32),
        strategy_code=jnp.asarray(-1, dtype=jnp.int32),
    )


def prepare_device_metrics_for_jax(
    devices_information_dict,
    n_devices,
    bs_radius,
    pathloss_exponent=2.0,
):
    """Prepare battery/energy and BS channel arrays for JAX experiments."""
    _require_jax()
    if n_devices < 1:
        raise ValueError("n_devices must be positive")
    if bs_radius < 1:
        raise ValueError("bs_radius must be at least 1")

    initial_energy = np.ones(n_devices, dtype=np.float32)
    distance_to_bs = np.ones(n_devices, dtype=np.float32)
    for device in range(n_devices):
        info = devices_information_dict[device]
        initial_energy[device] = float(info.get("device_battery", 100.0)) / 100.0
        distance_to_bs[device] = max(
            float(info.get("distance_from_device_to_BS", bs_radius)),
            1.0,
        )

    raw_quality = 1.0 / np.maximum(distance_to_bs, 1.0) ** float(pathloss_exponent)
    max_quality = float(np.max(raw_quality))
    if max_quality > 0.0:
        bs_channel_quality = raw_quality / max_quality
    else:
        bs_channel_quality = np.ones(n_devices, dtype=np.float32)

    return {
        "initial_energy": jnp.asarray(np.clip(initial_energy, 0.0, 1.0)),
        "bs_channel_quality": jnp.asarray(np.clip(bs_channel_quality, 0.0, 1.0)),
        "distance_to_bs": jnp.asarray(distance_to_bs),
    }


def _safe_access_probability(norm_value, psi):
    """Optimized ALOHA access probability used by model 3."""
    eps = jnp.asarray(1e-12, dtype=norm_value.dtype)
    euler_number = jnp.asarray(jnp.e, dtype=norm_value.dtype)
    raw_probability = euler_number * jnp.log(jnp.maximum(norm_value, eps)) - psi
    return jnp.where(norm_value <= eps, 0.0, jnp.clip(raw_probability, 0.0, 1.0))


def _apply_access_floor(probability, floor_fraction, fixed_access_probability, pcomp):
    """Keep optimized ALOHA from starving the channel late in training.

    The thesis optimized controller prioritizes large update norms, but with
    strong D2D aggregation those norms can shrink quickly and the dual variable
    can react too slowly.  A floor equal to a fraction of the fixed-ALOHA access
    probability preserves the distributed nature of the method: the BS can
    broadcast the scalar baseline, while each device/CH still decides locally
    whether its norm deserves more access than the baseline.
    """
    fixed_floor = jnp.minimum(fixed_access_probability, pcomp) * floor_fraction
    return jnp.minimum(jnp.maximum(probability, fixed_floor), pcomp)


def _normalized_device_battery(device_battery, number_of_devices, dtype):
    """Return device battery as normalized energy in ``[0, 1]``.

    Device generation stores battery as integer percentages in ``[1, 100]``.
    Some tests and legacy callers pass already-normalized energy values.  This
    helper accepts both conventions and gives the dynamic energy model a single
    scale to work with.
    """
    if device_battery is None:
        return jnp.ones((number_of_devices,), dtype=dtype)

    normalized = jnp.asarray(device_battery, dtype=dtype)
    battery_max = jnp.max(jnp.where(normalized > 0.0, normalized, 0.0))
    normalized = jnp.where(battery_max > 1.0, normalized / 100.0, normalized)
    return jnp.clip(normalized, 0.0, 1.0).astype(dtype)


def _d2d_energy_efficiency_profile_weights(level, dtype=None):
    """Return channel/battery/stability weights for dynamic CH rotation.

    The profiles are intentionally small and named instead of exposing three
    more free parameters.  That keeps the experiments easier to defend: each
    profile is a clear deployment posture rather than an overfit coefficient
    vector.
    """
    if level not in D2D_ENERGY_EFFICIENCY_PROFILES:
        raise ValueError(
            "d2d_energy_efficiency_level must be 'performance', 'balanced', or 'eco'"
        )

    weights = D2D_ENERGY_EFFICIENCY_PROFILES[level]
    if dtype is None or jnp is None:
        return weights
    return tuple(jnp.asarray(weight, dtype=dtype) for weight in weights)


def _normalized_bs_channel_quality(
    number_of_devices,
    dtype,
    device_distance_to_bs=None,
    pathloss_exponent=2.0,
):
    """Return normalized inverse-pathloss BS channel quality for each device."""
    if device_distance_to_bs is None:
        device_distance_to_bs = jnp.ones((number_of_devices,), dtype=dtype)
    else:
        device_distance_to_bs = jnp.asarray(device_distance_to_bs, dtype=dtype)

    one = jnp.asarray(1.0, dtype=dtype)
    eps = jnp.asarray(1e-12, dtype=dtype)
    pathloss_exponent = jnp.asarray(pathloss_exponent, dtype=dtype)
    raw_channel = one / jnp.maximum(device_distance_to_bs, one) ** pathloss_exponent
    return raw_channel / jnp.maximum(jnp.max(raw_channel), eps)


def _rayleigh_outage_success_probability(
    number_of_devices,
    dtype,
    device_distance_to_bs=None,
    pathloss_exponent=2.0,
    reference_snr=100000.0,
    snr_threshold=1.0,
):
    """Return collision-free packet success probability under Rayleigh outage.

    The model uses the standard outage form for an exponentially distributed
    channel power gain: a packet succeeds when instantaneous SNR is above a
    decoding threshold.  ``reference_snr`` is the average SNR at one meter, and
    distance/pathloss reduce that average SNR before evaluating
    ``P[SNR >= threshold] = exp(-threshold / avg_snr)``.
    """
    if device_distance_to_bs is None:
        device_distance_to_bs = jnp.ones((number_of_devices,), dtype=dtype)
    else:
        device_distance_to_bs = jnp.asarray(device_distance_to_bs, dtype=dtype)

    one = jnp.asarray(1.0, dtype=dtype)
    eps = jnp.asarray(1e-12, dtype=dtype)
    pathloss_exponent = jnp.asarray(pathloss_exponent, dtype=dtype)
    reference_snr = jnp.asarray(reference_snr, dtype=dtype)
    snr_threshold = jnp.asarray(snr_threshold, dtype=dtype)
    average_snr = reference_snr / (
        jnp.maximum(device_distance_to_bs, one) ** pathloss_exponent
    )
    probability = jnp.exp(-snr_threshold / jnp.maximum(average_snr, eps))
    return jnp.clip(probability, 0.0, 1.0).astype(dtype)


def _first_order_bs_tx_energy(
    distance_to_bs,
    packet_size,
    electronics_cost,
    amplifier_cost,
    pathloss_exponent,
):
    """Return normalized first-order radio energy for a BS uplink packet."""
    distance_to_bs = jnp.asarray(distance_to_bs)
    dtype = distance_to_bs.dtype
    packet_size = jnp.asarray(packet_size, dtype=dtype)
    electronics_cost = jnp.asarray(electronics_cost, dtype=dtype)
    amplifier_cost = jnp.asarray(amplifier_cost, dtype=dtype)
    pathloss_exponent = jnp.asarray(pathloss_exponent, dtype=dtype)
    return packet_size * electronics_cost + packet_size * amplifier_cost * (
        jnp.maximum(distance_to_bs, 1.0) ** pathloss_exponent
    )


def _first_order_d2d_tx_energy(
    distance_to_ch,
    packet_size,
    electronics_cost,
    amplifier_cost,
    pathloss_exponent,
):
    """Return normalized first-order radio energy for a D2D member packet."""
    distance_to_ch = jnp.asarray(distance_to_ch)
    dtype = distance_to_ch.dtype
    packet_size = jnp.asarray(packet_size, dtype=dtype)
    electronics_cost = jnp.asarray(electronics_cost, dtype=dtype)
    amplifier_cost = jnp.asarray(amplifier_cost, dtype=dtype)
    pathloss_exponent = jnp.asarray(pathloss_exponent, dtype=dtype)
    return packet_size * electronics_cost + packet_size * amplifier_cost * (
        jnp.maximum(distance_to_ch, 1.0) ** pathloss_exponent
    )


def _device_bs_success_probability(
    number_of_devices,
    dtype,
    success_mode,
    min_success_probability,
    pathloss_exponent,
    battery_exponent,
    reference_snr=100000.0,
    snr_threshold=1.0,
    device_distance_to_bs=None,
    device_battery=None,
):
    """Return per-device direct device-to-BS decoding probabilities.

    The baseline thesis-compatible simulator treats a CH upload as successful
    whenever the device/CH is allowed to transmit and does not collide on the
    selected ALOHA channel.  In that mode the physical uplink is idealized.

    ``success_mode="channel_quality"`` keeps the legacy inverse-pathloss
    probability model.  ``success_mode="rayleigh_outage"`` maps distance to an
    average SNR and evaluates the Rayleigh outage success probability.  In both
    enhanced modes, attempted transmissions still contend and collide exactly as
    before; the physical draw happens only after a collision-free ALOHA attempt.

    Arrays:

    - ``device_distance_to_bs``: float[K], meters from each device to the BS.
    - ``device_battery``: float/int[K], battery percentage or normalized energy.
    """
    if success_mode == "none":
        return jnp.ones((number_of_devices,), dtype=dtype)

    if device_distance_to_bs is None:
        device_distance_to_bs = jnp.ones((number_of_devices,), dtype=dtype)
    else:
        device_distance_to_bs = jnp.asarray(device_distance_to_bs, dtype=dtype)

    device_battery = _normalized_device_battery(
        device_battery,
        number_of_devices,
        dtype,
    )

    one = jnp.asarray(1.0, dtype=dtype)
    min_success = jnp.asarray(min_success_probability, dtype=dtype)
    battery_exponent = jnp.asarray(battery_exponent, dtype=dtype)

    if success_mode == "rayleigh_outage":
        return _rayleigh_outage_success_probability(
            number_of_devices,
            dtype,
            device_distance_to_bs=device_distance_to_bs,
            pathloss_exponent=pathloss_exponent,
            reference_snr=reference_snr,
            snr_threshold=snr_threshold,
        )

    channel_quality = _normalized_bs_channel_quality(
        number_of_devices,
        dtype,
        device_distance_to_bs=device_distance_to_bs,
        pathloss_exponent=pathloss_exponent,
    )
    battery_quality = jnp.clip(device_battery, 0.0, 1.0)
    raw_success = channel_quality * battery_quality**battery_exponent
    success_probability = min_success + (one - min_success) * jnp.clip(
        raw_success,
        0.0,
        1.0,
    )
    return jnp.clip(success_probability, 0.0, 1.0).astype(dtype)


def _d2d_ch_bs_success_probability(
    cluster_heads,
    cluster_mask,
    number_of_devices,
    dtype,
    success_mode,
    min_success_probability,
    pathloss_exponent,
    battery_exponent,
    reference_snr=100000.0,
    snr_threshold=1.0,
    device_distance_to_bs=None,
    device_battery=None,
):
    """Return per-cluster CH-to-BS decoding probabilities.

    The D2D version uses the same physical link model as direct device-to-BS
    uploads, but indexes the probability by the elected CH in each cluster.
    This keeps the physical model symmetric while preserving the D2D semantics:
    member aggregation still happens locally, then only the CH contends on the
    BS uplink.

    Arrays:

    - ``cluster_heads``: int[max_clusters], elected CH device id per row.
    - ``cluster_mask``: bool[max_clusters], active cluster rows.
    - ``device_distance_to_bs``: float[K], meters from each device to the BS.
    - ``device_battery``: float/int[K], battery percentage or normalized energy.
    """
    safe_heads = jnp.where(cluster_mask, cluster_heads, 0)
    per_device_probability = _device_bs_success_probability(
        number_of_devices=number_of_devices,
        dtype=dtype,
        success_mode=success_mode,
        min_success_probability=min_success_probability,
        pathloss_exponent=pathloss_exponent,
        battery_exponent=battery_exponent,
        reference_snr=reference_snr,
        snr_threshold=snr_threshold,
        device_distance_to_bs=device_distance_to_bs,
        device_battery=device_battery,
    )
    success_probability = per_device_probability[safe_heads]
    return jnp.where(cluster_mask, jnp.clip(success_probability, 0.0, 1.0), 0.0)


def _utility_load_controlled_access_probability(
    aggregate_norms,
    cluster_sizes,
    freshness,
    cluster_mask,
    n_channels,
    pcomp,
    fixed_access_probability,
    floor_fraction,
    norm_exponent,
    cluster_size_exponent,
    freshness_exponent,
    load_target_factor,
    load_allocation_mode,
    redistribution_fraction,
    redistribution_trigger_ratio,
    density_trigger_threshold,
    dense_trigger_ratio,
    clusterized_devices_fraction,
    optimized_success_ewma,
    fixed_success_target,
):
    """Allocate optimized D2D access using utility while preserving ALOHA load.

    Fixed D2D ALOHA is strong because it keeps the expected number of
    contenders near the number of channels.  This utility mode keeps that load
    target, but redistributes access probability toward CHs that carry larger
    aggregate updates, represent larger clusters, or have waited longer since a
    successful optimized-D2D transmission.
    """
    utility = _cluster_utility_scores(
        aggregate_norms=aggregate_norms,
        cluster_sizes=cluster_sizes,
        freshness=freshness,
        cluster_mask=cluster_mask,
        norm_exponent=norm_exponent,
        cluster_size_exponent=cluster_size_exponent,
        freshness_exponent=freshness_exponent,
    )
    return _load_controlled_access_from_utility(
        utility=utility,
        cluster_mask=cluster_mask,
        n_channels=n_channels,
        pcomp=pcomp,
        fixed_access_probability=fixed_access_probability,
        floor_fraction=floor_fraction,
        load_target_factor=load_target_factor,
        load_allocation_mode=load_allocation_mode,
        redistribution_fraction=redistribution_fraction,
        redistribution_trigger_ratio=redistribution_trigger_ratio,
        density_trigger_threshold=density_trigger_threshold,
        dense_trigger_ratio=dense_trigger_ratio,
        clusterized_devices_fraction=clusterized_devices_fraction,
        optimized_success_ewma=optimized_success_ewma,
        fixed_success_target=fixed_success_target,
    )


def _aoi_aware_utility_access_probability(
    aggregate_norms,
    cluster_sizes,
    freshness,
    cluster_aoi,
    cluster_mask,
    n_channels,
    pcomp,
    fixed_access_probability,
    floor_fraction,
    norm_exponent,
    cluster_size_exponent,
    freshness_exponent,
    aoi_weight,
    aoi_exponent,
    aoi_threshold_fraction,
    load_target_factor,
    load_allocation_mode,
    redistribution_fraction,
    redistribution_trigger_ratio,
    density_trigger_threshold,
    dense_trigger_ratio,
    clusterized_devices_fraction,
    optimized_success_ewma,
    fixed_success_target,
):
    """Allocate optimized D2D access with an explicit AoI-tail pressure term.

    ``utility`` already has a smooth freshness factor.  In the physical-energy
    experiments, however, optimized D2D reduced error and energy while keeping
    mean AoI higher than fixed D2D.  This policy tests a targeted correction:
    keep the same norm/size/freshness utility, then multiply it by a bounded
    bonus only for CHs whose cluster AoI is in the stale tail.

    Deployment interpretation:
    - ``cluster_aoi`` is the age since the BS last ACKed this cluster's CH
      aggregate.  The CH can maintain it locally from ACK/no-ACK feedback, and
      the BS can broadcast the scalar normalizers used by the score.
    - No centralized CH scheduling is introduced.  The score still becomes a
      local ALOHA access probability through the same load controller used by
      ``utility``, ``hybrid``, and ``adaptive_diversity``.

    Array contracts:
    - aggregate_norms: float[max_clusters], one aggregate norm per CH.
    - cluster_sizes: int[max_clusters], active members represented by the CH.
    - freshness: float[max_clusters], legacy age since optimized-D2D success.
    - cluster_aoi: float[max_clusters], explicit optimized-D2D cluster AoI.
    - cluster_mask: bool[max_clusters], true for real padded cluster rows.
    """
    dtype = aggregate_norms.dtype
    eps = jnp.asarray(1e-12, dtype=dtype)
    aoi_weight = jnp.asarray(aoi_weight, dtype=dtype)
    aoi_exponent = jnp.asarray(aoi_exponent, dtype=dtype)
    aoi_threshold_fraction = jnp.asarray(aoi_threshold_fraction, dtype=dtype)

    base_utility = _cluster_utility_scores(
        aggregate_norms=aggregate_norms,
        cluster_sizes=cluster_sizes,
        freshness=freshness,
        cluster_mask=cluster_mask,
        norm_exponent=norm_exponent,
        cluster_size_exponent=cluster_size_exponent,
        freshness_exponent=freshness_exponent,
    )

    # cluster_aoi: float[max_clusters], rounds since each cluster's aggregate
    # last reached the BS in optimized D2D.  The score is normalized by the
    # current maximum active-cluster AoI so the parameter remains dimensionless
    # across horizons such as t=200 or longer sweeps.
    active_aoi = jnp.where(cluster_mask, cluster_aoi, 0.0)
    normalized_aoi = (active_aoi + eps) / (jnp.max(active_aoi) + eps)

    # Only the stale tail receives the extra bonus.  With the default threshold
    # 0.75, clusters below 75% of the current maximum AoI keep the base utility.
    # This is deliberately less aggressive than replacing the objective with
    # AoI: the goal is to improve freshness without discarding the error/energy
    # gains from norm and cluster-size prioritization.
    remaining_tail_width = jnp.maximum(1.0 - aoi_threshold_fraction, eps)
    tail_pressure = jnp.clip(
        (normalized_aoi - aoi_threshold_fraction) / remaining_tail_width,
        0.0,
        1.0,
    )
    aoi_bonus = 1.0 + aoi_weight * tail_pressure**aoi_exponent
    aoi_aware_utility = base_utility * aoi_bonus

    return _load_controlled_access_from_utility(
        utility=aoi_aware_utility,
        cluster_mask=cluster_mask,
        n_channels=n_channels,
        pcomp=pcomp,
        fixed_access_probability=fixed_access_probability,
        floor_fraction=floor_fraction,
        load_target_factor=load_target_factor,
        load_allocation_mode=load_allocation_mode,
        redistribution_fraction=redistribution_fraction,
        redistribution_trigger_ratio=redistribution_trigger_ratio,
        density_trigger_threshold=density_trigger_threshold,
        dense_trigger_ratio=dense_trigger_ratio,
        clusterized_devices_fraction=clusterized_devices_fraction,
        optimized_success_ewma=optimized_success_ewma,
        fixed_success_target=fixed_success_target,
    )


def _aoi_floor_utility_access_probability(
    aggregate_norms,
    cluster_sizes,
    freshness,
    cluster_aoi,
    cluster_mask,
    n_channels,
    pcomp,
    fixed_access_probability,
    floor_fraction,
    norm_exponent,
    cluster_size_exponent,
    freshness_exponent,
    aoi_weight,
    aoi_exponent,
    aoi_threshold_fraction,
    load_target_factor,
    load_allocation_mode,
    redistribution_fraction,
    redistribution_trigger_ratio,
    density_trigger_threshold,
    dense_trigger_ratio,
    clusterized_devices_fraction,
    optimized_success_ewma,
    fixed_success_target,
):
    """Apply a conservative AoI floor on top of base utility access.

    ``aoi_aware_utility`` multiplies utility by an AoI bonus, which can move
    probability mass away from high-norm/high-channel-value clusters.  This
    alternative keeps the base utility allocator as the primary decision and
    only raises very stale clusters to a minimum access probability.

    Deployment interpretation:
    - CHs still make local ALOHA decisions from a probability broadcast/derived
      from scalar normalizers; no centralized CH scheduling is introduced.
    - ``aoi_weight`` is interpreted as a fraction of the fixed-D2D access
      probability.  For example, ``0.25`` means the oldest stale clusters get
      at least ``0.25 * fixed_access_probability`` unless ``pcomp`` is smaller.
    - This can slightly increase total offered load when many clusters are
      stale, so it should be evaluated as an AoI/fairness tradeoff rather than
      as a guaranteed error-norm improvement.

    Array contracts:
    - aggregate_norms: float[max_clusters], one aggregate norm per CH.
    - cluster_sizes: int[max_clusters], active members represented by the CH.
    - freshness: float[max_clusters], legacy age since optimized-D2D success.
    - cluster_aoi: float[max_clusters], explicit optimized-D2D cluster AoI.
    - cluster_mask: bool[max_clusters], true for real padded cluster rows.
    """
    dtype = aggregate_norms.dtype
    eps = jnp.asarray(1e-12, dtype=dtype)
    aoi_weight = jnp.asarray(aoi_weight, dtype=dtype)
    aoi_exponent = jnp.asarray(aoi_exponent, dtype=dtype)
    aoi_threshold_fraction = jnp.asarray(aoi_threshold_fraction, dtype=dtype)

    base_probability = _utility_load_controlled_access_probability(
        aggregate_norms=aggregate_norms,
        cluster_sizes=cluster_sizes,
        freshness=freshness,
        cluster_mask=cluster_mask,
        n_channels=n_channels,
        pcomp=pcomp,
        fixed_access_probability=fixed_access_probability,
        floor_fraction=floor_fraction,
        norm_exponent=norm_exponent,
        cluster_size_exponent=cluster_size_exponent,
        freshness_exponent=freshness_exponent,
        load_target_factor=load_target_factor,
        load_allocation_mode=load_allocation_mode,
        redistribution_fraction=redistribution_fraction,
        redistribution_trigger_ratio=redistribution_trigger_ratio,
        density_trigger_threshold=density_trigger_threshold,
        dense_trigger_ratio=dense_trigger_ratio,
        clusterized_devices_fraction=clusterized_devices_fraction,
        optimized_success_ewma=optimized_success_ewma,
        fixed_success_target=fixed_success_target,
    )

    # cluster_aoi: float[max_clusters], ACK age for optimized-D2D clusters.
    # Normalizing by the current maximum active-cluster age keeps the stale-tail
    # test independent of whether the sweep runs to t=200 or a longer horizon.
    active_aoi = jnp.where(cluster_mask, cluster_aoi, 0.0)
    normalized_aoi = (active_aoi + eps) / (jnp.max(active_aoi) + eps)

    # The tail pressure is zero below the configured stale threshold and one
    # for the currently oldest cluster(s).  The exponent shapes whether the
    # floor is spread across the whole tail or concentrated on the oldest rows.
    remaining_tail_width = jnp.maximum(1.0 - aoi_threshold_fraction, eps)
    tail_pressure = jnp.clip(
        (normalized_aoi - aoi_threshold_fraction) / remaining_tail_width,
        0.0,
        1.0,
    )
    stale_floor_probability = (
        fixed_access_probability * aoi_weight * tail_pressure**aoi_exponent
    )
    probability = jnp.maximum(base_probability, stale_floor_probability)
    return jnp.where(cluster_mask, jnp.clip(probability, 0.0, pcomp), 0.0)


def _load_controlled_access_from_utility(
    utility,
    cluster_mask,
    n_channels,
    pcomp,
    fixed_access_probability,
    floor_fraction,
    load_target_factor,
    load_allocation_mode,
    redistribution_fraction,
    redistribution_trigger_ratio,
    optimized_success_ewma,
    fixed_success_target,
    density_trigger_threshold=None,
    dense_trigger_ratio=None,
    clusterized_devices_fraction=None,
):
    """Convert CH utilities into smooth load-controlled ALOHA probabilities.

    ``proportional_clip`` is the original enhanced-mode allocator: distribute
    remaining probability mass in proportion to CH utility and clip each CH at
    ``pcomp``.  The clipping is local and simple, but any clipped excess mass is
    lost.

    ``water_filling`` keeps the same distributed ALOHA interpretation while
    using more of the expected CH contention budget.  The BS can broadcast the
    resulting scalar water level or its equivalent normalizer; every CH still
    computes its own access probability from local utility and shared scalars.
    No CH is centrally scheduled or forced to transmit.

    ``selective_water_filling`` is the intermediate ablation.  It starts from
    the selective ``proportional_clip`` probabilities, measures how much target
    load was lost to clipping, and redistributes only a configured fraction of
    that lost load to CHs that still have spare access capacity.

    ``conditional_selective_water_filling`` is the deployable combined policy.
    It uses the same partial redistribution, but only when the observed EWMA of
    successful optimized-D2D CH uploads falls below a configured fraction of the
    expected fixed-D2D CH throughput.  The effective trigger can become more
    conservative when the deployment is already densely clusterized: dense
    networks have many CHs with one-hop coverage, so extra redistributed load
    can create collisions without adding much new information.  The BS can know
    or estimate this density after cluster formation and broadcast only the
    scalar effective trigger; CHs still run local ALOHA trials.
    """
    dtype = utility.dtype
    eps = jnp.asarray(1e-12, dtype=dtype)
    if density_trigger_threshold is None:
        density_trigger_threshold = jnp.asarray(1.0, dtype=dtype)
    if dense_trigger_ratio is None:
        dense_trigger_ratio = redistribution_trigger_ratio
    if clusterized_devices_fraction is None:
        clusterized_devices_fraction = jnp.asarray(0.0, dtype=dtype)

    density_trigger_threshold = jnp.asarray(density_trigger_threshold, dtype=dtype)
    dense_trigger_ratio = jnp.asarray(dense_trigger_ratio, dtype=dtype)
    clusterized_devices_fraction = jnp.asarray(
        clusterized_devices_fraction,
        dtype=dtype,
    )
    active_count = jnp.sum(cluster_mask).astype(dtype)
    target_contenders = jnp.minimum(
        jnp.asarray(n_channels, dtype=dtype) * load_target_factor,
        pcomp * active_count,
    )

    floor_probability = jnp.minimum(fixed_access_probability, pcomp) * floor_fraction
    floor_by_cluster = jnp.where(cluster_mask, floor_probability, 0.0)
    floor_load = jnp.sum(floor_by_cluster)
    remaining_load = jnp.maximum(target_contenders - floor_load, 0.0)

    utility = jnp.where(cluster_mask, utility, 0.0)
    utility_sum = jnp.sum(utility)

    proportional_probability = jnp.where(
        utility_sum > eps,
        remaining_load * utility / utility_sum,
        0.0,
    )
    proportional_clip_probability = jnp.minimum(
        floor_by_cluster + proportional_probability,
        pcomp,
    )
    if load_allocation_mode == "proportional_clip":
        return proportional_clip_probability

    proportional_clip_load = jnp.sum(proportional_clip_probability)
    target_gap = jnp.maximum(target_contenders - proportional_clip_load, 0.0)
    redistribution_fraction = jnp.clip(redistribution_fraction, 0.0, 1.0)

    if load_allocation_mode == "selective_water_filling":
        redistribution_scale = redistribution_fraction
    elif load_allocation_mode == "conditional_selective_water_filling":
        throughput_ratio = optimized_success_ewma / jnp.maximum(fixed_success_target, eps)
        dense_cluster_regime = clusterized_devices_fraction >= jnp.clip(
            density_trigger_threshold,
            0.0,
            1.0,
        )
        effective_trigger_ratio = jnp.where(
            dense_cluster_regime,
            dense_trigger_ratio,
            redistribution_trigger_ratio,
        )
        # The trigger uses observed optimized-D2D CH throughput against the
        # expected fixed-D2D throughput, not attempted contenders and not model
        # error.  A dense clusterization regime uses a lower threshold by
        # default because the previous K=3000 runs showed that extra load is
        # often collision-dominated there.  The BS can estimate both inputs
        # from cluster formation plus ACKs, so the rule remains deployable.
        should_redistribute = throughput_ratio < jnp.clip(
            effective_trigger_ratio,
            0.0,
            1.0,
        )
        redistribution_scale = jnp.where(
            should_redistribute,
            redistribution_fraction,
            0.0,
        )
    else:
        redistribution_scale = jnp.asarray(1.0, dtype=dtype)

    selective_target_load = proportional_clip_load + redistribution_scale * target_gap
    selective_remaining_load = jnp.maximum(selective_target_load - floor_load, 0.0)

    capacity = jnp.where(
        cluster_mask,
        jnp.maximum(pcomp - floor_by_cluster, 0.0),
        0.0,
    )
    capped_remaining_load = jnp.minimum(selective_remaining_load, jnp.sum(capacity))
    max_water_level = jnp.max(capacity / jnp.maximum(utility, eps)) + 1.0

    def binary_search_step(bounds, _):
        low, high = bounds
        midpoint = (low + high) / 2.0
        allocated = jnp.sum(jnp.minimum(capacity, midpoint * utility))
        low = jnp.where(allocated < capped_remaining_load, midpoint, low)
        high = jnp.where(allocated < capped_remaining_load, high, midpoint)
        return (low, high), None

    # Fixed iteration count keeps the operation JIT/static and works on CPU/GPU.
    (low, high), _ = jax.lax.scan(
        binary_search_step,
        (jnp.asarray(0.0, dtype=dtype), max_water_level),
        None,
        length=32,
    )
    water_level = (low + high) / 2.0
    water_filling_probability = floor_by_cluster + jnp.minimum(
        capacity,
        water_level * utility,
    )
    bounded_water_filling_probability = jnp.minimum(water_filling_probability, pcomp)
    if load_allocation_mode == "conditional_selective_water_filling":
        # When the throughput trigger is off, this mode must be exactly the
        # legacy proportional clip allocator.  Returning water-filling with an
        # equivalent total load would still reshuffle CH probabilities and can
        # regress dense cases such as K=3000.
        return jnp.where(
            should_redistribute,
            bounded_water_filling_probability,
            proportional_clip_probability,
        )
    return bounded_water_filling_probability


def _cluster_utility_scores(
    aggregate_norms,
    cluster_sizes,
    freshness,
    cluster_mask,
    norm_exponent,
    cluster_size_exponent,
    freshness_exponent,
):
    """Return a dimensionless local priority score for each CH.

    Every term is normalized to ``[0, 1]`` across the active CH rows for this
    simulated round.  In a deployed system these normalizers can be broadcast
    by the BS from the previous control interval or estimated by gossip/summary
    messages; the CH still needs only its own aggregate update, active member
    count, and age since its last successful upload to compute its score.
    """
    dtype = aggregate_norms.dtype
    eps = jnp.asarray(1e-12, dtype=dtype)

    # aggregate_norms: float[max_clusters], Euclidean norm of the CH aggregate.
    # cluster_sizes: int[max_clusters], number of active members in the aggregate.
    # freshness: float[max_clusters], FL rounds since the last successful CH upload.
    active_norms = jnp.where(cluster_mask, aggregate_norms, 0.0)
    active_sizes = jnp.where(cluster_mask, cluster_sizes.astype(dtype), 0.0)
    active_freshness = jnp.where(cluster_mask, freshness, 0.0)

    norm_score = (active_norms + eps) / (jnp.max(active_norms) + eps)
    size_score = (active_sizes + eps) / (jnp.max(active_sizes) + eps)
    freshness_score = (active_freshness + 1.0) / (jnp.max(active_freshness) + 1.0)

    utility = (
        norm_score**norm_exponent
        * size_score**cluster_size_exponent
        * freshness_score**freshness_exponent
    )
    return jnp.where(cluster_mask, utility, 0.0)


def _cluster_novelty_scores(
    aggregate_updates,
    reference_direction,
    cluster_mask,
    novelty_floor,
):
    """Return how directionally new each CH aggregate is.

    The BS can maintain ``reference_direction`` from previously successful
    optimized-D2D uploads and broadcast it with the FL model.  A CH then
    compares its own aggregate update with that reference.  Aggregates aligned
    with the recent direction still keep ``novelty_floor`` credit, while
    orthogonal aggregates receive full novelty credit.
    """
    dtype = aggregate_updates.dtype
    eps = jnp.asarray(1e-12, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)

    # reference_direction: float[L], recent successful optimized-D2D direction.
    # aggregate_updates: float[max_clusters, L], one aggregate update per CH.
    reference_norm = jnp.linalg.norm(reference_direction)
    reference_unit = reference_direction / jnp.maximum(reference_norm, eps)
    aggregate_norms = jnp.linalg.norm(aggregate_updates, axis=1)
    cosine = jnp.sum(aggregate_updates * reference_unit[None, :], axis=1) / jnp.maximum(
        aggregate_norms,
        eps,
    )
    aligned_fraction = jnp.minimum(jnp.abs(cosine), one)
    orthogonal_fraction = jnp.sqrt(jnp.maximum(one - aligned_fraction**2, 0.0))

    novelty = novelty_floor + (one - novelty_floor) * orthogonal_fraction
    novelty = jnp.where(reference_norm > eps, novelty, one)
    return jnp.where(cluster_mask, novelty, 0.0)


def _hybrid_utility_access_probability(
    aggregate_updates,
    cluster_sizes,
    freshness,
    reference_direction,
    cluster_mask,
    n_channels,
    pcomp,
    fixed_access_probability,
    floor_fraction,
    norm_exponent,
    cluster_size_exponent,
    freshness_exponent,
    novelty_exponent,
    novelty_floor,
    load_target_factor,
    load_allocation_mode,
    redistribution_fraction,
    redistribution_trigger_ratio,
    density_trigger_threshold,
    dense_trigger_ratio,
    clusterized_devices_fraction,
    optimized_success_ewma,
    fixed_success_target,
):
    """Smooth optimized-D2D access with utility and marginal-direction novelty."""
    aggregate_norms = jnp.linalg.norm(aggregate_updates, axis=1)
    base_utility = _cluster_utility_scores(
        aggregate_norms=aggregate_norms,
        cluster_sizes=cluster_sizes,
        freshness=freshness,
        cluster_mask=cluster_mask,
        norm_exponent=norm_exponent,
        cluster_size_exponent=cluster_size_exponent,
        freshness_exponent=freshness_exponent,
    )
    novelty = _cluster_novelty_scores(
        aggregate_updates=aggregate_updates,
        reference_direction=reference_direction,
        cluster_mask=cluster_mask,
        novelty_floor=novelty_floor,
    )
    hybrid_utility = base_utility * novelty**novelty_exponent
    return _load_controlled_access_from_utility(
        utility=hybrid_utility,
        cluster_mask=cluster_mask,
        n_channels=n_channels,
        pcomp=pcomp,
        fixed_access_probability=fixed_access_probability,
        floor_fraction=floor_fraction,
        load_target_factor=load_target_factor,
        load_allocation_mode=load_allocation_mode,
        redistribution_fraction=redistribution_fraction,
        redistribution_trigger_ratio=redistribution_trigger_ratio,
        optimized_success_ewma=optimized_success_ewma,
        fixed_success_target=fixed_success_target,
        density_trigger_threshold=density_trigger_threshold,
        dense_trigger_ratio=dense_trigger_ratio,
        clusterized_devices_fraction=clusterized_devices_fraction,
    )


def _adaptive_diversity_access_probability(
    aggregate_updates,
    cluster_sizes,
    freshness,
    reference_direction,
    cluster_mask,
    n_channels,
    pcomp,
    fixed_access_probability,
    floor_fraction,
    early_norm_exponent,
    late_norm_exponent,
    cluster_size_exponent,
    early_freshness_exponent,
    late_freshness_exponent,
    novelty_exponent,
    novelty_floor,
    load_target_factor,
    load_allocation_mode,
    redistribution_fraction,
    redistribution_trigger_ratio,
    density_trigger_threshold,
    dense_trigger_ratio,
    clusterized_devices_fraction,
    optimized_success_ewma,
    fixed_success_target,
    switch_fraction,
    switch_gain,
    iteration_index,
    max_iterations,
):
    """Two-phase optimized-D2D access with utility first and diversity later.

    This policy is intentionally still deployable as a distributed CH decision:
    each CH needs its local aggregate update, active D2D member count, and
    freshness age.  The BS can broadcast the normalizers, the recent reference
    direction, and the current phase scalar with the global model.

    Array contracts:
    - aggregate_updates: float[max_clusters, L], one active aggregate per CH.
    - cluster_sizes: int[max_clusters], active member count in each aggregate.
    - freshness: float[max_clusters], rounds since the CH last uploaded.
    - reference_direction: float[L], recent successful optimized-D2D direction.
    - cluster_mask: bool[max_clusters], true for real padded cluster rows.
    """
    dtype = aggregate_updates.dtype
    aggregate_norms = jnp.linalg.norm(aggregate_updates, axis=1)

    # The early phase is aggressive: prioritize high-norm, large aggregates so
    # the global model moves quickly while errors are still large.  This mirrors
    # utility-guided participant selection without requiring the BS to choose a
    # deterministic client set.
    early_utility = _cluster_utility_scores(
        aggregate_norms=aggregate_norms,
        cluster_sizes=cluster_sizes,
        freshness=freshness,
        cluster_mask=cluster_mask,
        norm_exponent=early_norm_exponent,
        cluster_size_exponent=cluster_size_exponent,
        freshness_exponent=early_freshness_exponent,
    )

    # The late phase is more conservative about raw norm and gives more room to
    # freshness and directional novelty.  This tests whether avoiding redundant
    # CH aggregate directions helps once the largest updates have already driven
    # most of the error down.
    late_base_utility = _cluster_utility_scores(
        aggregate_norms=aggregate_norms,
        cluster_sizes=cluster_sizes,
        freshness=freshness,
        cluster_mask=cluster_mask,
        norm_exponent=late_norm_exponent,
        cluster_size_exponent=cluster_size_exponent,
        freshness_exponent=late_freshness_exponent,
    )
    novelty = _cluster_novelty_scores(
        aggregate_updates=aggregate_updates,
        reference_direction=reference_direction,
        cluster_mask=cluster_mask,
        novelty_floor=novelty_floor,
    )
    late_utility = late_base_utility * novelty**novelty_exponent

    # phase_progress uses t / max_t instead of measured error.  The BS knows the
    # planned horizon and can broadcast this scalar, whereas true optimization
    # error is unavailable in real deployments.
    phase_progress = (
        (iteration_index.astype(dtype) + 1.0)
        / jnp.asarray(max_iterations, dtype=dtype)
    )
    phase = jax.nn.sigmoid(switch_gain * (phase_progress - switch_fraction))
    adaptive_utility = (1.0 - phase) * early_utility + phase * late_utility

    return _load_controlled_access_from_utility(
        utility=adaptive_utility,
        cluster_mask=cluster_mask,
        n_channels=n_channels,
        pcomp=pcomp,
        fixed_access_probability=fixed_access_probability,
        floor_fraction=floor_fraction,
        load_target_factor=load_target_factor,
        load_allocation_mode=load_allocation_mode,
        redistribution_fraction=redistribution_fraction,
        redistribution_trigger_ratio=redistribution_trigger_ratio,
        optimized_success_ewma=optimized_success_ewma,
        fixed_success_target=fixed_success_target,
        density_trigger_threshold=density_trigger_threshold,
        dense_trigger_ratio=dense_trigger_ratio,
        clusterized_devices_fraction=clusterized_devices_fraction,
    )


def _max_weight_threshold_access_probability(
    aggregate_norms,
    cluster_sizes,
    freshness,
    cluster_mask,
    threshold,
    pcomp,
    fixed_access_probability,
    floor_fraction,
    norm_exponent,
    cluster_size_exponent,
    freshness_exponent,
    threshold_gain,
):
    """Distributed max-weight-style optimized D2D access probability.

    The proportional utility policy spreads a fixed load budget across all
    active CHs.  This mode is more selective: a CH maps its local utility score
    through a sigmoid gate centered on a global threshold.  A successful system
    needs only a scalar threshold broadcast by the BS.  If too many CHs contend,
    the threshold rises; if too few contend, it falls.
    """
    dtype = aggregate_norms.dtype
    floor_probability = jnp.minimum(fixed_access_probability, pcomp) * floor_fraction
    remaining_probability = jnp.maximum(pcomp - floor_probability, 0.0)
    utility = _cluster_utility_scores(
        aggregate_norms=aggregate_norms,
        cluster_sizes=cluster_sizes,
        freshness=freshness,
        cluster_mask=cluster_mask,
        norm_exponent=norm_exponent,
        cluster_size_exponent=cluster_size_exponent,
        freshness_exponent=freshness_exponent,
    )

    # threshold: scalar dual state. Larger values make the CH gate stricter.
    # threshold_gain: sigmoid slope. Larger values concentrate access on the
    # highest-utility CHs; smaller values behave closer to smooth utility mode.
    gate = jax.nn.sigmoid(threshold_gain * (utility - threshold))
    probability = floor_probability + remaining_probability * gate
    return jnp.where(cluster_mask, jnp.minimum(probability, pcomp), 0.0).astype(dtype)


def _successful_from_draws(
    channel_key,
    random_draws,
    probability,
    eligibility,
    n_channels,
    link_key=None,
    link_success_probability=None,
):
    """Resolve multichannel ALOHA contenders for one model.

    ``random_draws`` supplies the compute/access draw.  ``probability`` may be
    scalar or vector.  Each candidate picks one channel; a candidate succeeds
    only if no other candidate picked the same channel.  When
    ``link_success_probability`` is supplied, a collision-free candidate still
    needs a successful physical-layer decoding draw.  Attempts that fail this
    link draw still counted as contenders, which is important because weak
    transmitters can consume channel opportunities even when the BS cannot
    decode them.
    """
    probability = jnp.broadcast_to(probability, random_draws.shape)
    candidates = (random_draws < probability) & eligibility
    selected_channels = jax.random.randint(
        channel_key,
        shape=random_draws.shape,
        minval=0,
        maxval=n_channels,
        dtype=jnp.int32,
    )
    channel_counts = jnp.bincount(
        jnp.where(candidates, selected_channels, 0),
        weights=candidates.astype(jnp.int32),
        length=n_channels,
    )
    collision_free = candidates & (channel_counts[selected_channels] == 1)
    if link_success_probability is None:
        successful = collision_free
    else:
        if link_key is None:
            link_key = channel_key
        link_success_probability = jnp.broadcast_to(
            link_success_probability,
            random_draws.shape,
        )
        link_draws = jax.random.uniform(
            link_key,
            random_draws.shape,
            dtype=random_draws.dtype,
        )
        successful = collision_free & (link_draws < link_success_probability)
    return successful, jnp.sum(candidates).astype(jnp.int32)


def error_calculator_trace_jax(
    number_of_mobile_devices__k: int,
    data_dimension__L: int,
    number_of_parallel_channels__M: int,
    probability_that_user_can_compute_its_local_update__pcomp: float,
    max_iterations_t: int,
    learning_rate__u1: float,
    step_size__u: float,
    clusters: JaxClusterResult,
    seed=None,
    normalize_by_k: bool = False,
    d2d_member_compute_probability: float = 1.0,
    d2d_member_link_success_probability: float = 1.0,
    d2d_ch_bs_success_mode: str = "none",
    d2d_ch_bs_min_success_probability: float = 0.20,
    d2d_ch_bs_pathloss_exponent: float = 2.0,
    d2d_ch_bs_battery_exponent: float = 0.0,
    d2d_ch_bs_reference_snr: float = 100000.0,
    d2d_ch_bs_snr_threshold: float = 1.0,
    device_bs_success_mode: str = "none",
    device_bs_min_success_probability: float = 0.20,
    device_bs_pathloss_exponent: float = 2.0,
    device_bs_battery_exponent: float = 0.0,
    device_bs_reference_snr: float = 100000.0,
    device_bs_snr_threshold: float = 1.0,
    energy_drain_mode: str = "none",
    energy_model: str = "constant",
    battery_feasibility_mode: str = "off",
    energy_direct_bs_cost: float = 0.0,
    energy_d2d_member_cost: float = 0.0,
    energy_ch_bs_cost: float = 0.0,
    energy_electronics_cost: float = 0.0002,
    energy_bs_amplifier_cost: float = 2e-8,
    energy_d2d_amplifier_cost: float = 1e-6,
    energy_bs_pathloss_exponent: float = 2.0,
    energy_d2d_pathloss_exponent: float = 2.0,
    energy_aggregation_cost: float = 0.00002,
    energy_update_size: float = 1.0,
    energy_aggregate_size: float = 1.0,
    energy_rotation_control_cost: float = 0.0,
    d2d_ch_rotation_mode: str = "static",
    d2d_ch_rotation_interval: int = 10,
    d2d_energy_efficiency_level: str = "balanced",
    device_coords=None,
    device_radius=None,
    device_distance_to_bs=None,
    device_battery=None,
    optimized_access_floor_fraction: float = 0.0,
    optimized_d2d_access_floor_fraction: float = 0.0,
    optimized_d2d_access_mode: str = "norm",
    optimized_d2d_norm_exponent: float = 1.0,
    optimized_d2d_cluster_size_exponent: float = 1.0,
    optimized_d2d_freshness_exponent: float = 0.5,
    optimized_d2d_threshold_gain: float = 8.0,
    optimized_d2d_novelty_exponent: float = 1.0,
    optimized_d2d_novelty_floor: float = 0.25,
    optimized_d2d_reference_decay: float = 0.90,
    optimized_d2d_load_target_factor: float = 1.0,
    optimized_d2d_load_allocation_mode: str = "conditional_selective_water_filling",
    optimized_d2d_redistribution_fraction: float = 0.5,
    optimized_d2d_redistribution_trigger_ratio: float = 0.95,
    optimized_d2d_density_trigger_threshold: float = 0.95,
    optimized_d2d_dense_trigger_ratio: float = 0.90,
    optimized_d2d_throughput_ewma_decay: float = 0.90,
    optimized_d2d_late_norm_exponent: float = 1.25,
    optimized_d2d_late_freshness_exponent: float = 1.0,
    optimized_d2d_adaptive_switch_fraction: float = 0.30,
    optimized_d2d_adaptive_switch_gain: float = 12.0,
    optimized_d2d_aoi_weight: float = 0.5,
    optimized_d2d_aoi_exponent: float = 1.0,
    optimized_d2d_aoi_threshold_fraction: float = 0.75,
    checkpoints=None,
    dtype=None,
) -> JaxTraceResult:
    """Run all six HFL/ALOHA scenarios and return metrics over time.

    The six columns of ``error_norms`` and ``successful_uploads`` are:

    0. polling without D2D
    1. fixed ALOHA without D2D
    2. optimized ALOHA without D2D
    3. polling with D2D
    4. fixed ALOHA with D2D
    5. optimized ALOHA with D2D
    """
    _require_jax()
    if number_of_mobile_devices__k < 1:
        raise ValueError("number_of_mobile_devices__k must be positive")
    if data_dimension__L < 1:
        raise ValueError("data_dimension__L must be positive")
    if number_of_parallel_channels__M < 1:
        raise ValueError("number_of_parallel_channels__M must be positive")
    if max_iterations_t < 1:
        raise ValueError("max_iterations_t must be positive")
    if not 0.0 <= probability_that_user_can_compute_its_local_update__pcomp <= 1.0:
        raise ValueError("pcomp must be in [0, 1]")
    if not 0.0 <= d2d_member_compute_probability <= 1.0:
        raise ValueError("d2d_member_compute_probability must be in [0, 1]")
    if not 0.0 <= d2d_member_link_success_probability <= 1.0:
        raise ValueError("d2d_member_link_success_probability must be in [0, 1]")
    if d2d_ch_bs_success_mode not in {"none", "channel_quality", "rayleigh_outage"}:
        raise ValueError(
            "d2d_ch_bs_success_mode must be 'none', 'channel_quality', or "
            "'rayleigh_outage'"
        )
    if not 0.0 <= d2d_ch_bs_min_success_probability <= 1.0:
        raise ValueError("d2d_ch_bs_min_success_probability must be in [0, 1]")
    if d2d_ch_bs_pathloss_exponent < 0.0:
        raise ValueError("d2d_ch_bs_pathloss_exponent must be non-negative")
    if d2d_ch_bs_battery_exponent < 0.0:
        raise ValueError("d2d_ch_bs_battery_exponent must be non-negative")
    if d2d_ch_bs_reference_snr <= 0.0:
        raise ValueError("d2d_ch_bs_reference_snr must be positive")
    if d2d_ch_bs_snr_threshold < 0.0:
        raise ValueError("d2d_ch_bs_snr_threshold must be non-negative")
    if device_bs_success_mode not in {"none", "channel_quality", "rayleigh_outage"}:
        raise ValueError(
            "device_bs_success_mode must be 'none', 'channel_quality', or "
            "'rayleigh_outage'"
        )
    if not 0.0 <= device_bs_min_success_probability <= 1.0:
        raise ValueError("device_bs_min_success_probability must be in [0, 1]")
    if device_bs_pathloss_exponent < 0.0:
        raise ValueError("device_bs_pathloss_exponent must be non-negative")
    if device_bs_battery_exponent < 0.0:
        raise ValueError("device_bs_battery_exponent must be non-negative")
    if device_bs_reference_snr <= 0.0:
        raise ValueError("device_bs_reference_snr must be positive")
    if device_bs_snr_threshold < 0.0:
        raise ValueError("device_bs_snr_threshold must be non-negative")
    if energy_drain_mode not in {"none", "dynamic"}:
        raise ValueError("energy_drain_mode must be 'none' or 'dynamic'")
    if energy_model not in {"constant", "first_order_radio"}:
        raise ValueError("energy_model must be 'constant' or 'first_order_radio'")
    if battery_feasibility_mode not in {"off", "required_energy"}:
        raise ValueError(
            "battery_feasibility_mode must be 'off' or 'required_energy'"
        )
    if energy_direct_bs_cost < 0.0:
        raise ValueError("energy_direct_bs_cost must be non-negative")
    if energy_d2d_member_cost < 0.0:
        raise ValueError("energy_d2d_member_cost must be non-negative")
    if energy_ch_bs_cost < 0.0:
        raise ValueError("energy_ch_bs_cost must be non-negative")
    if energy_electronics_cost < 0.0:
        raise ValueError("energy_electronics_cost must be non-negative")
    if energy_bs_amplifier_cost < 0.0:
        raise ValueError("energy_bs_amplifier_cost must be non-negative")
    if energy_d2d_amplifier_cost < 0.0:
        raise ValueError("energy_d2d_amplifier_cost must be non-negative")
    if energy_bs_pathloss_exponent < 0.0:
        raise ValueError("energy_bs_pathloss_exponent must be non-negative")
    if energy_d2d_pathloss_exponent < 0.0:
        raise ValueError("energy_d2d_pathloss_exponent must be non-negative")
    if energy_aggregation_cost < 0.0:
        raise ValueError("energy_aggregation_cost must be non-negative")
    if energy_update_size <= 0.0:
        raise ValueError("energy_update_size must be positive")
    if energy_aggregate_size <= 0.0:
        raise ValueError("energy_aggregate_size must be positive")
    if energy_rotation_control_cost < 0.0:
        raise ValueError("energy_rotation_control_cost must be non-negative")
    if d2d_ch_rotation_mode not in {"static", "energy_aware"}:
        raise ValueError("d2d_ch_rotation_mode must be 'static' or 'energy_aware'")
    if d2d_ch_rotation_interval < 1:
        raise ValueError("d2d_ch_rotation_interval must be at least 1")
    _d2d_energy_efficiency_profile_weights(d2d_energy_efficiency_level)
    if d2d_ch_rotation_mode == "energy_aware":
        if energy_drain_mode != "dynamic":
            raise ValueError(
                "energy-aware D2D CH rotation requires energy_drain_mode='dynamic'"
            )
        if device_coords is None or device_radius is None:
            raise ValueError(
                "energy-aware D2D CH rotation requires device_coords and device_radius"
            )
    if not 0.0 <= optimized_access_floor_fraction <= 1.0:
        raise ValueError("optimized_access_floor_fraction must be in [0, 1]")
    if not 0.0 <= optimized_d2d_access_floor_fraction <= 1.0:
        raise ValueError("optimized_d2d_access_floor_fraction must be in [0, 1]")
    if optimized_d2d_access_mode not in {
        "norm",
        "utility",
        "max_weight",
        "hybrid",
        "adaptive_diversity",
        "aoi_aware_utility",
        "aoi_floor_utility",
    }:
        raise ValueError(
            "optimized_d2d_access_mode must be 'norm', 'utility', "
            "'max_weight', 'hybrid', 'adaptive_diversity', or "
            "'aoi_aware_utility'/'aoi_floor_utility'"
        )
    if optimized_d2d_norm_exponent < 0.0:
        raise ValueError("optimized_d2d_norm_exponent must be non-negative")
    if optimized_d2d_cluster_size_exponent < 0.0:
        raise ValueError("optimized_d2d_cluster_size_exponent must be non-negative")
    if optimized_d2d_freshness_exponent < 0.0:
        raise ValueError("optimized_d2d_freshness_exponent must be non-negative")
    if optimized_d2d_threshold_gain <= 0.0:
        raise ValueError("optimized_d2d_threshold_gain must be positive")
    if optimized_d2d_novelty_exponent < 0.0:
        raise ValueError("optimized_d2d_novelty_exponent must be non-negative")
    if not 0.0 <= optimized_d2d_novelty_floor <= 1.0:
        raise ValueError("optimized_d2d_novelty_floor must be in [0, 1]")
    if not 0.0 <= optimized_d2d_reference_decay <= 1.0:
        raise ValueError("optimized_d2d_reference_decay must be in [0, 1]")
    if optimized_d2d_load_target_factor <= 0.0:
        raise ValueError("optimized_d2d_load_target_factor must be positive")
    if optimized_d2d_load_allocation_mode not in {
        "water_filling",
        "selective_water_filling",
        "conditional_selective_water_filling",
        "proportional_clip",
    }:
        raise ValueError(
            "optimized_d2d_load_allocation_mode must be "
            "'water_filling', 'selective_water_filling', "
            "'conditional_selective_water_filling', or 'proportional_clip'"
        )
    if not 0.0 <= optimized_d2d_redistribution_fraction <= 1.0:
        raise ValueError("optimized_d2d_redistribution_fraction must be in [0, 1]")
    if not 0.0 <= optimized_d2d_redistribution_trigger_ratio <= 1.0:
        raise ValueError(
            "optimized_d2d_redistribution_trigger_ratio must be in [0, 1]"
        )
    if not 0.0 <= optimized_d2d_density_trigger_threshold <= 1.0:
        raise ValueError("optimized_d2d_density_trigger_threshold must be in [0, 1]")
    if not 0.0 <= optimized_d2d_dense_trigger_ratio <= 1.0:
        raise ValueError("optimized_d2d_dense_trigger_ratio must be in [0, 1]")
    if not 0.0 <= optimized_d2d_throughput_ewma_decay <= 1.0:
        raise ValueError("optimized_d2d_throughput_ewma_decay must be in [0, 1]")
    if optimized_d2d_late_norm_exponent < 0.0:
        raise ValueError("optimized_d2d_late_norm_exponent must be non-negative")
    if optimized_d2d_late_freshness_exponent < 0.0:
        raise ValueError(
            "optimized_d2d_late_freshness_exponent must be non-negative"
        )
    if not 0.0 <= optimized_d2d_adaptive_switch_fraction <= 1.0:
        raise ValueError("optimized_d2d_adaptive_switch_fraction must be in [0, 1]")
    if optimized_d2d_adaptive_switch_gain <= 0.0:
        raise ValueError("optimized_d2d_adaptive_switch_gain must be positive")
    if optimized_d2d_aoi_weight < 0.0:
        raise ValueError("optimized_d2d_aoi_weight must be non-negative")
    if optimized_d2d_aoi_exponent < 0.0:
        raise ValueError("optimized_d2d_aoi_exponent must be non-negative")
    if not 0.0 <= optimized_d2d_aoi_threshold_fraction < 1.0:
        raise ValueError("optimized_d2d_aoi_threshold_fraction must be in [0, 1)")

    dtype = jnp.float32 if dtype is None else dtype
    pcomp = jnp.asarray(
        probability_that_user_can_compute_its_local_update__pcomp,
        dtype=dtype,
    )
    learning_rate = jnp.asarray(learning_rate__u1, dtype=dtype)
    step_size = jnp.asarray(step_size__u, dtype=dtype)
    d2d_compute_probability = jnp.asarray(d2d_member_compute_probability, dtype=dtype)
    d2d_link_probability = jnp.asarray(d2d_member_link_success_probability, dtype=dtype)
    d2d_ch_bs_min_success_probability = jnp.asarray(
        d2d_ch_bs_min_success_probability,
        dtype=dtype,
    )
    d2d_ch_bs_pathloss_exponent = jnp.asarray(
        d2d_ch_bs_pathloss_exponent,
        dtype=dtype,
    )
    d2d_ch_bs_battery_exponent = jnp.asarray(
        d2d_ch_bs_battery_exponent,
        dtype=dtype,
    )
    d2d_ch_bs_reference_snr = jnp.asarray(d2d_ch_bs_reference_snr, dtype=dtype)
    d2d_ch_bs_snr_threshold = jnp.asarray(d2d_ch_bs_snr_threshold, dtype=dtype)
    device_bs_min_success_probability = jnp.asarray(
        device_bs_min_success_probability,
        dtype=dtype,
    )
    device_bs_pathloss_exponent = jnp.asarray(
        device_bs_pathloss_exponent,
        dtype=dtype,
    )
    device_bs_battery_exponent = jnp.asarray(
        device_bs_battery_exponent,
        dtype=dtype,
    )
    device_bs_reference_snr = jnp.asarray(device_bs_reference_snr, dtype=dtype)
    device_bs_snr_threshold = jnp.asarray(device_bs_snr_threshold, dtype=dtype)
    energy_enabled = jnp.asarray(
        1.0 if energy_drain_mode == "dynamic" else 0.0,
        dtype=dtype,
    )
    energy_direct_bs_cost = jnp.asarray(energy_direct_bs_cost, dtype=dtype)
    energy_d2d_member_cost = jnp.asarray(energy_d2d_member_cost, dtype=dtype)
    energy_ch_bs_cost = jnp.asarray(energy_ch_bs_cost, dtype=dtype)
    energy_electronics_cost = jnp.asarray(energy_electronics_cost, dtype=dtype)
    energy_bs_amplifier_cost = jnp.asarray(energy_bs_amplifier_cost, dtype=dtype)
    energy_d2d_amplifier_cost = jnp.asarray(energy_d2d_amplifier_cost, dtype=dtype)
    energy_bs_pathloss_exponent = jnp.asarray(
        energy_bs_pathloss_exponent,
        dtype=dtype,
    )
    energy_d2d_pathloss_exponent = jnp.asarray(
        energy_d2d_pathloss_exponent,
        dtype=dtype,
    )
    energy_aggregation_cost = jnp.asarray(energy_aggregation_cost, dtype=dtype)
    energy_update_size = jnp.asarray(energy_update_size, dtype=dtype)
    energy_aggregate_size = jnp.asarray(energy_aggregate_size, dtype=dtype)
    energy_rotation_control_cost = jnp.asarray(
        energy_rotation_control_cost,
        dtype=dtype,
    )
    battery_feasibility_enabled = battery_feasibility_mode == "required_energy"
    d2d_ch_rotation_enabled = d2d_ch_rotation_mode == "energy_aware"
    d2d_rotation_channel_weight, d2d_rotation_battery_weight, d2d_rotation_stability_weight = (
        _d2d_energy_efficiency_profile_weights(d2d_energy_efficiency_level, dtype)
    )
    access_floor_fraction = jnp.asarray(optimized_access_floor_fraction, dtype=dtype)
    d2d_access_floor_fraction = jnp.asarray(
        optimized_d2d_access_floor_fraction,
        dtype=dtype,
    )
    d2d_norm_exponent = jnp.asarray(optimized_d2d_norm_exponent, dtype=dtype)
    d2d_cluster_size_exponent = jnp.asarray(
        optimized_d2d_cluster_size_exponent,
        dtype=dtype,
    )
    d2d_freshness_exponent = jnp.asarray(optimized_d2d_freshness_exponent, dtype=dtype)
    d2d_threshold_gain = jnp.asarray(optimized_d2d_threshold_gain, dtype=dtype)
    d2d_novelty_exponent = jnp.asarray(optimized_d2d_novelty_exponent, dtype=dtype)
    d2d_novelty_floor = jnp.asarray(optimized_d2d_novelty_floor, dtype=dtype)
    d2d_reference_decay = jnp.asarray(optimized_d2d_reference_decay, dtype=dtype)
    d2d_load_target_factor = jnp.asarray(optimized_d2d_load_target_factor, dtype=dtype)
    d2d_redistribution_fraction = jnp.asarray(
        optimized_d2d_redistribution_fraction,
        dtype=dtype,
    )
    d2d_redistribution_trigger_ratio = jnp.asarray(
        optimized_d2d_redistribution_trigger_ratio,
        dtype=dtype,
    )
    d2d_density_trigger_threshold = jnp.asarray(
        optimized_d2d_density_trigger_threshold,
        dtype=dtype,
    )
    d2d_dense_trigger_ratio = jnp.asarray(
        optimized_d2d_dense_trigger_ratio,
        dtype=dtype,
    )
    d2d_throughput_ewma_decay = jnp.asarray(
        optimized_d2d_throughput_ewma_decay,
        dtype=dtype,
    )
    d2d_late_norm_exponent = jnp.asarray(
        optimized_d2d_late_norm_exponent,
        dtype=dtype,
    )
    d2d_late_freshness_exponent = jnp.asarray(
        optimized_d2d_late_freshness_exponent,
        dtype=dtype,
    )
    d2d_adaptive_switch_fraction = jnp.asarray(
        optimized_d2d_adaptive_switch_fraction,
        dtype=dtype,
    )
    d2d_adaptive_switch_gain = jnp.asarray(
        optimized_d2d_adaptive_switch_gain,
        dtype=dtype,
    )
    d2d_aoi_weight = jnp.asarray(optimized_d2d_aoi_weight, dtype=dtype)
    d2d_aoi_exponent = jnp.asarray(optimized_d2d_aoi_exponent, dtype=dtype)
    d2d_aoi_threshold_fraction = jnp.asarray(
        optimized_d2d_aoi_threshold_fraction,
        dtype=dtype,
    )

    key = _key_from_seed(seed)
    data_key, true_weight_key, init_weight_key, scan_key = jax.random.split(key, 4)

    k_devices = int(number_of_mobile_devices__k)
    data_dimension = int(data_dimension__L)
    n_channels = int(number_of_parallel_channels__M)
    cluster_members = clusters.cluster_members.astype(jnp.int32)
    cluster_sizes = clusters.cluster_sizes.astype(jnp.int32)
    cluster_mask = clusters.cluster_mask
    # Fraction in [0, 1].  The clustering result stores this public metric as a
    # percentage for CSV/figures, while the controller needs a compact scalar to
    # decide whether the current deployment is dense enough to use the more
    # conservative trigger.
    clusterized_devices_fraction = (
        jnp.asarray(clusters.clusterized_devices_rate, dtype=dtype) / 100.0
    )
    max_cluster_size = cluster_members.shape[1]
    safe_members = jnp.where(cluster_members >= 0, cluster_members, 0)

    # Synthetic linear-regression task from the thesis.
    users_input__x = jax.random.normal(
        data_key,
        (k_devices, data_dimension),
        dtype=dtype,
    )
    weights_vector__w = jax.random.normal(
        true_weight_key,
        (data_dimension,),
        dtype=dtype,
    )
    users_output__y = users_input__x @ weights_vector__w

    initial_weight = jax.random.normal(init_weight_key, (data_dimension,), dtype=dtype)
    weights = jnp.stack([initial_weight] * 6, axis=0)

    normalization_factor = (
        jnp.asarray(k_devices, dtype=users_input__x.dtype)
        if normalize_by_k
        else jnp.asarray(1.0, dtype=users_input__x.dtype)
    )

    number_of_clusterheads = jnp.maximum(clusters.number_of_clusters, 1)
    access_probability = jnp.minimum(
        jnp.asarray(n_channels / k_devices, dtype=users_input__x.dtype),
        1.0,
    )
    access_probability_d2d = jnp.minimum(
        jnp.asarray(n_channels, dtype=users_input__x.dtype)
        / number_of_clusterheads.astype(users_input__x.dtype),
        1.0,
    )
    fixed_d2d_attempt_probability = jnp.minimum(access_probability_d2d, pcomp)
    active_clusterhead_count = jnp.sum(cluster_mask).astype(users_input__x.dtype)
    same_channel_escape_probability = jnp.maximum(
        1.0
        - fixed_d2d_attempt_probability
        / jnp.asarray(n_channels, dtype=users_input__x.dtype),
        0.0,
    )
    expected_fixed_d2d_ch_successes = (
        active_clusterhead_count
        * fixed_d2d_attempt_probability
        * same_channel_escape_probability
        ** jnp.maximum(active_clusterhead_count - 1.0, 0.0)
    )

    member_positions = jnp.arange(max_cluster_size, dtype=jnp.int32)[None, :]
    member_mask = member_positions < cluster_sizes[:, None]
    cluster_heads = jnp.where(cluster_mask, cluster_members[:, 0], 0)
    if device_coords is None or device_radius is None:
        candidate_can_cover_members = member_mask
    else:
        coords = jnp.asarray(device_coords, dtype=dtype)
        candidate_coords = coords[safe_members]
        coverage_deltas = (
            candidate_coords[:, :, None, :] - candidate_coords[:, None, :, :]
        )
        distance_squared = jnp.sum(coverage_deltas * coverage_deltas, axis=-1)
        radius_squared = jnp.asarray(device_radius, dtype=dtype) ** 2
        candidate_can_cover_members = jnp.all(
            jnp.where(
                member_mask[:, None, :],
                distance_squared <= radius_squared,
                True,
            ),
            axis=2,
        )
    valid_ch_candidate = member_mask & cluster_mask[:, None] & candidate_can_cover_members
    bs_channel_quality = _normalized_bs_channel_quality(
        k_devices,
        dtype,
        device_distance_to_bs=device_distance_to_bs,
        pathloss_exponent=d2d_ch_bs_pathloss_exponent,
    )
    initial_device_battery = _normalized_device_battery(
        device_battery,
        k_devices,
        dtype,
    )
    if device_distance_to_bs is None:
        device_distance_to_bs_array = jnp.ones((k_devices,), dtype=dtype)
    else:
        device_distance_to_bs_array = jnp.asarray(device_distance_to_bs, dtype=dtype)

    if energy_model == "first_order_radio":
        direct_bs_energy_by_device = _first_order_bs_tx_energy(
            device_distance_to_bs_array,
            energy_update_size,
            energy_electronics_cost,
            energy_bs_amplifier_cost,
            energy_bs_pathloss_exponent,
        ).astype(dtype)
        ch_bs_energy_by_device = _first_order_bs_tx_energy(
            device_distance_to_bs_array,
            energy_aggregate_size,
            energy_electronics_cost,
            energy_bs_amplifier_cost,
            energy_bs_pathloss_exponent,
        ).astype(dtype)
        member_rx_energy = energy_update_size * energy_electronics_cost
        aggregate_update_energy = energy_update_size * energy_aggregation_cost
        coords_for_energy = (
            None if device_coords is None else jnp.asarray(device_coords, dtype=dtype)
        )
        d2d_member_energy_by_position = None
    else:
        coords_for_energy = None
        direct_bs_energy_by_device = jnp.full(
            (k_devices,),
            energy_direct_bs_cost,
            dtype=dtype,
        )
        ch_bs_energy_by_device = jnp.full(
            (k_devices,),
            energy_ch_bs_cost,
            dtype=dtype,
        )
        member_rx_energy = jnp.asarray(0.0, dtype=dtype)
        aggregate_update_energy = jnp.asarray(0.0, dtype=dtype)
        d2d_member_energy_by_position = jnp.full(
            cluster_members.shape,
            energy_d2d_member_cost,
            dtype=dtype,
        )
    # Battery state is tracked independently for every curve.  This avoids a
    # modeling artifact where, for example, direct optimized ALOHA could drain
    # the batteries used by optimized D2D in the same Monte Carlo trajectory.
    # Shape: float[6, K], normalized energy in [0, 1].
    initial_ch_bs_success_probability = _d2d_ch_bs_success_probability(
        cluster_heads=cluster_heads,
        cluster_mask=cluster_mask,
        number_of_devices=k_devices,
        dtype=dtype,
        success_mode=d2d_ch_bs_success_mode,
        min_success_probability=d2d_ch_bs_min_success_probability,
        pathloss_exponent=d2d_ch_bs_pathloss_exponent,
        battery_exponent=d2d_ch_bs_battery_exponent,
        reference_snr=d2d_ch_bs_reference_snr,
        snr_threshold=d2d_ch_bs_snr_threshold,
        device_distance_to_bs=device_distance_to_bs,
        device_battery=initial_device_battery,
    )
    mean_initial_ch_bs_success_probability = (
        jnp.sum(initial_ch_bs_success_probability)
        / jnp.maximum(
            active_clusterhead_count,
            jnp.asarray(1.0, dtype=users_input__x.dtype),
        )
    )
    initial_expected_fixed_d2d_ch_successes = (
        expected_fixed_d2d_ch_successes * mean_initial_ch_bs_success_probability
    )

    def local_updates_for_weights(current_weights):
        # current_weights: float[6, L]
        # predictions: float[K, 6], one column per scenario.
        predictions = users_input__x @ current_weights.T
        errors = predictions - users_output__y[:, None]
        return errors[:, :, None] * users_input__x[:, None, :]

    def aggregate_cluster_updates(device_updates, active_member_mask):
        # device_updates: float[K, L]
        # gathered: float[max_clusters, Cmax, L]
        gathered = device_updates[safe_members]
        masked = jnp.where(active_member_mask[:, :, None], gathered, 0.0)
        return jnp.sum(masked, axis=1)

    def apply_gradient(current_weight, gradient):
        return current_weight - learning_rate * gradient / normalization_factor

    def select_energy_aware_cluster_heads(current_heads, scenario_battery):
        """Select one valid CH per cluster from current battery/channel state.

        current_heads: int[max_clusters], the CHs used by this D2D scenario in
        the previous iteration.
        scenario_battery: float[K], normalized battery for this D2D scenario.

        The member set is fixed.  A candidate is eligible only if it can cover
        every valid member in the cluster, so re-election cannot violate the
        one-hop D2D invariant.
        """
        stability_score = (
            safe_members == current_heads[:, None]
        ).astype(users_input__x.dtype)
        candidate_scores = (
            d2d_rotation_channel_weight * bs_channel_quality[safe_members]
            + d2d_rotation_battery_weight * scenario_battery[safe_members]
            + d2d_rotation_stability_weight * stability_score
        )
        candidate_scores = jnp.where(
            valid_ch_candidate,
            candidate_scores,
            -jnp.inf,
        )
        best_position = jnp.argmax(candidate_scores, axis=1).astype(jnp.int32)
        proposed_heads = jnp.take_along_axis(
            cluster_members,
            best_position[:, None],
            axis=1,
        )[:, 0]
        has_candidate = jnp.any(valid_ch_candidate, axis=1)
        return jnp.where(cluster_mask & has_candidate, proposed_heads, current_heads)

    def direct_device_has_required_energy(scenario_battery):
        if not battery_feasibility_enabled:
            return jnp.ones((k_devices,), dtype=jnp.bool_)
        return scenario_battery >= direct_bs_energy_by_device

    def member_energy_for_heads(heads):
        if energy_model != "first_order_radio":
            return d2d_member_energy_by_position
        if coords_for_energy is None:
            distances = jnp.ones(cluster_members.shape, dtype=dtype)
        else:
            member_coords = coords_for_energy[safe_members]
            head_coords = coords_for_energy[heads][:, None, :]
            deltas = member_coords - head_coords
            distances = jnp.sqrt(jnp.sum(deltas * deltas, axis=-1))
        return _first_order_d2d_tx_energy(
            distances,
            energy_update_size,
            energy_electronics_cost,
            energy_d2d_amplifier_cost,
            energy_d2d_pathloss_exponent,
        ).astype(dtype)

    def ch_required_energy_for_heads(heads, active_member_mask):
        if energy_model != "first_order_radio":
            return jnp.where(
                cluster_mask,
                ch_bs_energy_by_device[heads],
                0.0,
            )
        is_scenario_ch = safe_members == heads[:, None]
        active_count = jnp.sum(active_member_mask, axis=1).astype(dtype)
        active_non_ch_count = jnp.sum(
            active_member_mask & (~is_scenario_ch),
            axis=1,
        ).astype(dtype)
        required = (
            ch_bs_energy_by_device[heads]
            + active_non_ch_count * member_rx_energy
            + active_count * aggregate_update_energy
        )
        return jnp.where(cluster_mask, required, 0.0)

    def ch_has_required_energy(heads, scenario_battery, active_member_mask):
        if not battery_feasibility_enabled:
            return cluster_mask
        required = ch_required_energy_for_heads(heads, active_member_mask)
        return cluster_mask & (scenario_battery[heads] >= required)

    def scenario_aoi_summary(non_d2d_aoi, d2d_aoi, current_iteration):
        d2d_denominator = jnp.maximum(
            active_clusterhead_count,
            jnp.asarray(1.0, dtype=users_input__x.dtype),
        )
        d2d_masked = jnp.where(cluster_mask[None, :], d2d_aoi, 0.0)
        mean_non_d2d = jnp.mean(non_d2d_aoi, axis=1)
        mean_d2d = jnp.sum(d2d_masked, axis=1) / d2d_denominator
        peak_non_d2d = jnp.max(non_d2d_aoi, axis=1)
        peak_d2d = jnp.max(d2d_masked, axis=1)
        non_d2d_mask = jnp.ones(non_d2d_aoi.shape, dtype=jnp.bool_)
        d2d_mask = jnp.broadcast_to(cluster_mask[None, :], d2d_aoi.shape)

        def masked_percentile(values, mask, percentile):
            # values: float[scenario_count, item_count], AoI samples.
            # mask: bool[scenario_count, item_count], true for real samples.
            # AoI is integer-valued and bounded by max_iterations_t + 1.  A
            # histogram avoids sorting thousands of devices/clusters inside
            # every scan step, which matters for large-K GPU sweeps.
            histogram_length = max_iterations_t + 2

            def one_row_percentile(row_values, row_mask):
                ages = jnp.clip(
                    row_values.astype(jnp.int32),
                    0,
                    histogram_length - 1,
                )
                weights = row_mask.astype(jnp.int32)
                histogram = jnp.bincount(
                    ages,
                    weights=weights,
                    length=histogram_length,
                )
                sample_count = jnp.sum(weights)
                target_count = jnp.ceil(
                    percentile * sample_count.astype(values.dtype)
                ).astype(jnp.int32)
                cumulative = jnp.cumsum(histogram)
                value = jnp.argmax(cumulative >= target_count).astype(values.dtype)
                return jnp.where(sample_count > 0, value, 0.0)

            return jax.vmap(one_row_percentile)(values, mask)

        def masked_fraction_above(values, mask, threshold):
            # values: float[scenario_count, item_count], AoI samples.
            # threshold: scalar AoI age.  The output is a fraction in [0, 1].
            sample_count = jnp.sum(mask.astype(values.dtype), axis=1)
            stale_count = jnp.sum(
                (values > threshold).astype(values.dtype) * mask.astype(values.dtype),
                axis=1,
            )
            return jnp.where(sample_count > 0.0, stale_count / sample_count, 0.0)

        p75_non_d2d = masked_percentile(non_d2d_aoi, non_d2d_mask, 0.75)
        p75_d2d = masked_percentile(d2d_aoi, d2d_mask, 0.75)
        p90_non_d2d = masked_percentile(non_d2d_aoi, non_d2d_mask, 0.90)
        p90_d2d = masked_percentile(d2d_aoi, d2d_mask, 0.90)
        p95_non_d2d = masked_percentile(non_d2d_aoi, non_d2d_mask, 0.95)
        p95_d2d = masked_percentile(d2d_aoi, d2d_mask, 0.95)

        # Stale-tail fractions are normalized by elapsed time, not final max_t,
        # so the curves are meaningful at early checkpoints as well as t=200.
        current_t = current_iteration.astype(users_input__x.dtype)
        stale_50_threshold = 0.50 * current_t
        stale_75_threshold = 0.75 * current_t
        stale_100_threshold = jnp.asarray(100.0, dtype=users_input__x.dtype)
        stale_50_non_d2d = masked_fraction_above(
            non_d2d_aoi,
            non_d2d_mask,
            stale_50_threshold,
        )
        stale_50_d2d = masked_fraction_above(
            d2d_aoi,
            d2d_mask,
            stale_50_threshold,
        )
        stale_75_non_d2d = masked_fraction_above(
            non_d2d_aoi,
            non_d2d_mask,
            stale_75_threshold,
        )
        stale_75_d2d = masked_fraction_above(
            d2d_aoi,
            d2d_mask,
            stale_75_threshold,
        )
        stale_100_non_d2d = masked_fraction_above(
            non_d2d_aoi,
            non_d2d_mask,
            stale_100_threshold,
        )
        stale_100_d2d = masked_fraction_above(
            d2d_aoi,
            d2d_mask,
            stale_100_threshold,
        )
        return (
            jnp.concatenate([mean_non_d2d, mean_d2d], axis=0),
            jnp.concatenate([peak_non_d2d, peak_d2d], axis=0),
            jnp.concatenate([p75_non_d2d, p75_d2d], axis=0),
            jnp.concatenate([p90_non_d2d, p90_d2d], axis=0),
            jnp.concatenate([p95_non_d2d, p95_d2d], axis=0),
            jnp.concatenate([stale_50_non_d2d, stale_50_d2d], axis=0),
            jnp.concatenate([stale_75_non_d2d, stale_75_d2d], axis=0),
            jnp.concatenate([stale_100_non_d2d, stale_100_d2d], axis=0),
        )

    def scan_iteration(state, iteration_index):
        (
            key,
            current_weights,
            # current_batteries: float[6, K], normalized battery remaining for
            # each scenario.  It is static when energy_drain_mode="none" and
            # evolves when energy_drain_mode="dynamic".
            current_batteries,
            current_d2d_heads,
            clusterhead_energy_totals,
            psi,
            psi_d2d,
            upload_totals,
            clusterhead_upload_totals,
            optimized_d2d_freshness,
            optimized_d2d_reference_direction,
            optimized_d2d_success_ewma,
            non_d2d_aoi,
            d2d_aoi,
        ) = state
        (
            key,
            draw_key,
            polling_key,
            active_compute_key,
            active_link_key,
            channel_key_2,
            channel_key_2_d2d,
            channel_key_3,
            channel_key_3_d2d,
            polling_d2d_link_key,
            fixed_d2d_link_key,
            optimized_d2d_link_key,
        ) = jax.random.split(key, 12)

        device_draws = jax.random.uniform(draw_key, (k_devices,), dtype=dtype)
        channel_draws = jax.random.uniform(polling_key, (n_channels,), dtype=dtype)
        direct_polling_link_key = jax.random.fold_in(polling_key, 101)
        direct_fixed_link_key = jax.random.fold_in(channel_key_2, 101)
        direct_optimized_link_key = jax.random.fold_in(channel_key_3, 101)
        should_rotate_d2d_heads = (
            d2d_ch_rotation_enabled
            & ((iteration_index % d2d_ch_rotation_interval) == 0)
        )
        proposed_d2d_heads = jnp.stack(
            [
                select_energy_aware_cluster_heads(
                    current_d2d_heads[0],
                    current_batteries[3],
                ),
                select_energy_aware_cluster_heads(
                    current_d2d_heads[1],
                    current_batteries[4],
                ),
                select_energy_aware_cluster_heads(
                    current_d2d_heads[2],
                    current_batteries[5],
                ),
            ],
            axis=0,
        )
        d2d_heads = jnp.where(
            should_rotate_d2d_heads,
            proposed_d2d_heads,
            current_d2d_heads,
        )
        polling_d2d_heads = d2d_heads[0]
        fixed_d2d_heads = d2d_heads[1]
        optimized_d2d_heads = d2d_heads[2]
        device_bs_success_probability_1 = _device_bs_success_probability(
            number_of_devices=k_devices,
            dtype=dtype,
            success_mode=device_bs_success_mode,
            min_success_probability=device_bs_min_success_probability,
            pathloss_exponent=device_bs_pathloss_exponent,
            battery_exponent=device_bs_battery_exponent,
            reference_snr=device_bs_reference_snr,
            snr_threshold=device_bs_snr_threshold,
            device_distance_to_bs=device_distance_to_bs,
            device_battery=current_batteries[0],
        )
        device_bs_success_probability_2 = _device_bs_success_probability(
            number_of_devices=k_devices,
            dtype=dtype,
            success_mode=device_bs_success_mode,
            min_success_probability=device_bs_min_success_probability,
            pathloss_exponent=device_bs_pathloss_exponent,
            battery_exponent=device_bs_battery_exponent,
            reference_snr=device_bs_reference_snr,
            snr_threshold=device_bs_snr_threshold,
            device_distance_to_bs=device_distance_to_bs,
            device_battery=current_batteries[1],
        )
        device_bs_success_probability_3 = _device_bs_success_probability(
            number_of_devices=k_devices,
            dtype=dtype,
            success_mode=device_bs_success_mode,
            min_success_probability=device_bs_min_success_probability,
            pathloss_exponent=device_bs_pathloss_exponent,
            battery_exponent=device_bs_battery_exponent,
            reference_snr=device_bs_reference_snr,
            snr_threshold=device_bs_snr_threshold,
            device_distance_to_bs=device_distance_to_bs,
            device_battery=current_batteries[2],
        )
        ch_bs_success_probability_1 = _d2d_ch_bs_success_probability(
            cluster_heads=polling_d2d_heads,
            cluster_mask=cluster_mask,
            number_of_devices=k_devices,
            dtype=dtype,
            success_mode=d2d_ch_bs_success_mode,
            min_success_probability=d2d_ch_bs_min_success_probability,
            pathloss_exponent=d2d_ch_bs_pathloss_exponent,
            battery_exponent=d2d_ch_bs_battery_exponent,
            reference_snr=d2d_ch_bs_reference_snr,
            snr_threshold=d2d_ch_bs_snr_threshold,
            device_distance_to_bs=device_distance_to_bs,
            device_battery=current_batteries[3],
        )
        ch_bs_success_probability_2 = _d2d_ch_bs_success_probability(
            cluster_heads=fixed_d2d_heads,
            cluster_mask=cluster_mask,
            number_of_devices=k_devices,
            dtype=dtype,
            success_mode=d2d_ch_bs_success_mode,
            min_success_probability=d2d_ch_bs_min_success_probability,
            pathloss_exponent=d2d_ch_bs_pathloss_exponent,
            battery_exponent=d2d_ch_bs_battery_exponent,
            reference_snr=d2d_ch_bs_reference_snr,
            snr_threshold=d2d_ch_bs_snr_threshold,
            device_distance_to_bs=device_distance_to_bs,
            device_battery=current_batteries[4],
        )
        ch_bs_success_probability_3 = _d2d_ch_bs_success_probability(
            cluster_heads=optimized_d2d_heads,
            cluster_mask=cluster_mask,
            number_of_devices=k_devices,
            dtype=dtype,
            success_mode=d2d_ch_bs_success_mode,
            min_success_probability=d2d_ch_bs_min_success_probability,
            pathloss_exponent=d2d_ch_bs_pathloss_exponent,
            battery_exponent=d2d_ch_bs_battery_exponent,
            reference_snr=d2d_ch_bs_reference_snr,
            snr_threshold=d2d_ch_bs_snr_threshold,
            device_distance_to_bs=device_distance_to_bs,
            device_battery=current_batteries[5],
        )
        direct_link_success_probability_2 = (
            device_bs_success_probability_2
            if device_bs_success_mode != "none"
            else None
        )
        direct_link_success_probability_3 = (
            device_bs_success_probability_3
            if device_bs_success_mode != "none"
            else None
        )
        mean_fixed_d2d_ch_success_probability = (
            jnp.sum(ch_bs_success_probability_2)
            / jnp.maximum(
                active_clusterhead_count,
                jnp.asarray(1.0, dtype=users_input__x.dtype),
            )
        )
        current_expected_fixed_d2d_ch_successes = (
            expected_fixed_d2d_ch_successes * mean_fixed_d2d_ch_success_probability
        )

        # Optional member-to-CH realism.  With static CHs, the CH is in column
        # 0.  With energy-aware rotation, the CH can move to another member
        # position independently for each D2D scenario.  The elected CH is
        # always locally active; non-CH members still draw compute/link success.
        active_compute_draws = jax.random.uniform(
            active_compute_key,
            cluster_members.shape,
            dtype=dtype,
        )
        active_link_draws = jax.random.uniform(
            active_link_key,
            cluster_members.shape,
            dtype=dtype,
        )
        member_compute_link_success = (
            (active_compute_draws < d2d_compute_probability)
            & (active_link_draws < d2d_link_probability)
        )

        direct_can_attempt_1 = direct_device_has_required_energy(current_batteries[0])
        direct_can_attempt_2 = direct_device_has_required_energy(current_batteries[1])
        direct_can_attempt_3 = direct_device_has_required_energy(current_batteries[2])

        member_energy_1 = member_energy_for_heads(polling_d2d_heads)
        member_energy_2 = member_energy_for_heads(fixed_d2d_heads)
        member_energy_3 = member_energy_for_heads(optimized_d2d_heads)

        def active_member_mask_for_heads(heads, scenario_battery, member_energy):
            is_scenario_ch = safe_members == heads[:, None]
            if battery_feasibility_enabled:
                member_has_energy = scenario_battery[safe_members] >= member_energy
            else:
                member_has_energy = jnp.ones(cluster_members.shape, dtype=jnp.bool_)
            return (
                member_mask
                & cluster_mask[:, None]
                & (
                    is_scenario_ch
                    | (member_compute_link_success & member_has_energy)
                )
            )

        active_member_mask_1 = active_member_mask_for_heads(
            polling_d2d_heads,
            current_batteries[3],
            member_energy_1,
        )
        active_member_mask_2 = active_member_mask_for_heads(
            fixed_d2d_heads,
            current_batteries[4],
            member_energy_2,
        )
        active_member_mask_3 = active_member_mask_for_heads(
            optimized_d2d_heads,
            current_batteries[5],
            member_energy_3,
        )
        active_member_counts_1 = jnp.sum(active_member_mask_1, axis=1).astype(jnp.int32)
        active_member_counts_2 = jnp.sum(active_member_mask_2, axis=1).astype(jnp.int32)
        active_member_counts_3 = jnp.sum(active_member_mask_3, axis=1).astype(jnp.int32)
        ch_can_attempt_1 = ch_has_required_energy(
            polling_d2d_heads,
            current_batteries[3],
            active_member_mask_1,
        )
        ch_can_attempt_2 = ch_has_required_energy(
            fixed_d2d_heads,
            current_batteries[4],
            active_member_mask_2,
        )
        ch_can_attempt_3 = ch_has_required_energy(
            optimized_d2d_heads,
            current_batteries[5],
            active_member_mask_3,
        )

        local_updates = local_updates_for_weights(current_weights)
        aggregate_updates_model_1 = aggregate_cluster_updates(
            local_updates[:, 3, :],
            active_member_mask_1,
        )
        aggregate_updates_model_2 = aggregate_cluster_updates(
            local_updates[:, 4, :],
            active_member_mask_2,
        )
        aggregate_updates_model_3 = aggregate_cluster_updates(
            local_updates[:, 5, :],
            active_member_mask_3,
        )
        aggregate_norms_model_3 = jnp.linalg.norm(aggregate_updates_model_3, axis=1)

        # Model 1: polling without D2D.
        scheduled_users = (
            iteration_index * n_channels + jnp.arange(n_channels, dtype=jnp.int32)
        ) % k_devices
        polling_compute_success = (
            (channel_draws < pcomp) & direct_can_attempt_1[scheduled_users]
        )
        if device_bs_success_mode != "none":
            direct_polling_link_draws = jax.random.uniform(
                direct_polling_link_key,
                (n_channels,),
                dtype=dtype,
            )
            polling_success = polling_compute_success & (
                direct_polling_link_draws
                < device_bs_success_probability_1[scheduled_users]
            )
        else:
            polling_success = polling_compute_success
        gradient_1 = jnp.sum(
            jnp.where(polling_success[:, None], local_updates[scheduled_users, 0, :], 0.0),
            axis=0,
        )
        upload_1 = jnp.sum(polling_success).astype(jnp.int32)

        # Model 1 with D2D: polling schedules CH rows instead of device IDs.
        scheduled_clusters = (
            iteration_index * n_channels + jnp.arange(n_channels, dtype=jnp.int32)
        ) % number_of_clusterheads
        polling_d2d_link_draws = jax.random.uniform(
            polling_d2d_link_key,
            (n_channels,),
            dtype=dtype,
        )
        polling_d2d_attempt = polling_compute_success & ch_can_attempt_1[
            scheduled_clusters
        ]
        polling_success_d2d = polling_d2d_attempt & (
            polling_d2d_link_draws < ch_bs_success_probability_1[scheduled_clusters]
        )
        gradient_1_d2d = jnp.sum(
            jnp.where(
                polling_success_d2d[:, None],
                aggregate_updates_model_1[scheduled_clusters],
                0.0,
            ),
            axis=0,
        )
        upload_1_d2d = jnp.sum(
            jnp.where(
                polling_success_d2d,
                active_member_counts_1[scheduled_clusters],
                0,
            )
        ).astype(jnp.int32)
        ch_upload_1_d2d = jnp.sum(polling_success_d2d).astype(jnp.int32)

        # Model 2: fixed ALOHA without and with D2D.
        threshold = jnp.minimum(
            access_probability,
            pcomp,
        )
        candidates_2_mask = (device_draws < threshold) & direct_can_attempt_2
        success_2, _ = _successful_from_draws(
            channel_key_2,
            device_draws,
            threshold,
            direct_can_attempt_2,
            n_channels,
            link_key=direct_fixed_link_key,
            link_success_probability=direct_link_success_probability_2,
        )
        gradient_2 = jnp.sum(jnp.where(success_2[:, None], local_updates[:, 1, :], 0.0), axis=0)
        upload_2 = jnp.sum(success_2).astype(jnp.int32)

        fixed_cluster_draws = device_draws[fixed_d2d_heads]
        threshold_d2d = jnp.minimum(
            access_probability_d2d,
            pcomp,
        )
        candidates_2_d2d_mask = (fixed_cluster_draws < threshold_d2d) & cluster_mask
        candidates_2_d2d_mask = candidates_2_d2d_mask & ch_can_attempt_2
        success_2_d2d, _ = _successful_from_draws(
            channel_key_2_d2d,
            fixed_cluster_draws,
            threshold_d2d,
            cluster_mask & ch_can_attempt_2,
            n_channels,
            link_key=fixed_d2d_link_key,
            link_success_probability=ch_bs_success_probability_2,
        )
        gradient_2_d2d = jnp.sum(
            jnp.where(success_2_d2d[:, None], aggregate_updates_model_2, 0.0),
            axis=0,
        )
        upload_2_d2d = jnp.sum(
            jnp.where(success_2_d2d, active_member_counts_2, 0)
        ).astype(jnp.int32)
        ch_upload_2_d2d = jnp.sum(success_2_d2d).astype(jnp.int32)

        # Model 3: optimized ALOHA probabilities from update norms.
        local_norms_model_3 = jnp.linalg.norm(local_updates[:, 2, :], axis=1)
        optimized_probability = jnp.where(
            iteration_index == 0,
            access_probability,
            _safe_access_probability(local_norms_model_3, psi),
        )
        optimized_probability = jnp.minimum(
            optimized_probability,
            pcomp,
        )
        optimized_probability = _apply_access_floor(
            probability=optimized_probability,
            floor_fraction=access_floor_fraction,
            fixed_access_probability=access_probability,
            pcomp=pcomp,
        )
        candidates_3_mask = (device_draws < optimized_probability) & direct_can_attempt_3
        success_3, candidates_3 = _successful_from_draws(
            channel_key_3,
            device_draws,
            optimized_probability,
            direct_can_attempt_3,
            n_channels,
            link_key=direct_optimized_link_key,
            link_success_probability=direct_link_success_probability_3,
        )
        gradient_3 = jnp.sum(jnp.where(success_3[:, None], local_updates[:, 2, :], 0.0), axis=0)
        upload_3 = jnp.sum(success_3).astype(jnp.int32)
        next_psi = psi + step_size * (
            candidates_3.astype(users_input__x.dtype) - n_channels
        )

        if optimized_d2d_access_mode == "utility":
            optimized_probability_d2d = _utility_load_controlled_access_probability(
                aggregate_norms=aggregate_norms_model_3,
                cluster_sizes=active_member_counts_3,
                freshness=optimized_d2d_freshness,
                cluster_mask=cluster_mask,
                n_channels=n_channels,
                pcomp=pcomp,
                fixed_access_probability=access_probability_d2d,
                floor_fraction=d2d_access_floor_fraction,
                norm_exponent=d2d_norm_exponent,
                cluster_size_exponent=d2d_cluster_size_exponent,
                freshness_exponent=d2d_freshness_exponent,
                load_target_factor=d2d_load_target_factor,
                load_allocation_mode=optimized_d2d_load_allocation_mode,
                redistribution_fraction=d2d_redistribution_fraction,
                redistribution_trigger_ratio=d2d_redistribution_trigger_ratio,
                density_trigger_threshold=d2d_density_trigger_threshold,
                dense_trigger_ratio=d2d_dense_trigger_ratio,
                clusterized_devices_fraction=clusterized_devices_fraction,
                optimized_success_ewma=optimized_d2d_success_ewma,
                fixed_success_target=current_expected_fixed_d2d_ch_successes,
            )
        elif optimized_d2d_access_mode == "aoi_aware_utility":
            optimized_probability_d2d = _aoi_aware_utility_access_probability(
                aggregate_norms=aggregate_norms_model_3,
                cluster_sizes=active_member_counts_3,
                freshness=optimized_d2d_freshness,
                cluster_aoi=d2d_aoi[2],
                cluster_mask=cluster_mask,
                n_channels=n_channels,
                pcomp=pcomp,
                fixed_access_probability=access_probability_d2d,
                floor_fraction=d2d_access_floor_fraction,
                norm_exponent=d2d_norm_exponent,
                cluster_size_exponent=d2d_cluster_size_exponent,
                freshness_exponent=d2d_freshness_exponent,
                aoi_weight=d2d_aoi_weight,
                aoi_exponent=d2d_aoi_exponent,
                aoi_threshold_fraction=d2d_aoi_threshold_fraction,
                load_target_factor=d2d_load_target_factor,
                load_allocation_mode=optimized_d2d_load_allocation_mode,
                redistribution_fraction=d2d_redistribution_fraction,
                redistribution_trigger_ratio=d2d_redistribution_trigger_ratio,
                density_trigger_threshold=d2d_density_trigger_threshold,
                dense_trigger_ratio=d2d_dense_trigger_ratio,
                clusterized_devices_fraction=clusterized_devices_fraction,
                optimized_success_ewma=optimized_d2d_success_ewma,
                fixed_success_target=current_expected_fixed_d2d_ch_successes,
            )
        elif optimized_d2d_access_mode == "aoi_floor_utility":
            optimized_probability_d2d = _aoi_floor_utility_access_probability(
                aggregate_norms=aggregate_norms_model_3,
                cluster_sizes=active_member_counts_3,
                freshness=optimized_d2d_freshness,
                cluster_aoi=d2d_aoi[2],
                cluster_mask=cluster_mask,
                n_channels=n_channels,
                pcomp=pcomp,
                fixed_access_probability=access_probability_d2d,
                floor_fraction=d2d_access_floor_fraction,
                norm_exponent=d2d_norm_exponent,
                cluster_size_exponent=d2d_cluster_size_exponent,
                freshness_exponent=d2d_freshness_exponent,
                aoi_weight=d2d_aoi_weight,
                aoi_exponent=d2d_aoi_exponent,
                aoi_threshold_fraction=d2d_aoi_threshold_fraction,
                load_target_factor=d2d_load_target_factor,
                load_allocation_mode=optimized_d2d_load_allocation_mode,
                redistribution_fraction=d2d_redistribution_fraction,
                redistribution_trigger_ratio=d2d_redistribution_trigger_ratio,
                density_trigger_threshold=d2d_density_trigger_threshold,
                dense_trigger_ratio=d2d_dense_trigger_ratio,
                clusterized_devices_fraction=clusterized_devices_fraction,
                optimized_success_ewma=optimized_d2d_success_ewma,
                fixed_success_target=current_expected_fixed_d2d_ch_successes,
            )
        elif optimized_d2d_access_mode == "max_weight":
            optimized_probability_d2d = _max_weight_threshold_access_probability(
                aggregate_norms=aggregate_norms_model_3,
                cluster_sizes=active_member_counts_3,
                freshness=optimized_d2d_freshness,
                cluster_mask=cluster_mask,
                threshold=psi_d2d,
                pcomp=pcomp,
                fixed_access_probability=access_probability_d2d,
                floor_fraction=d2d_access_floor_fraction,
                norm_exponent=d2d_norm_exponent,
                cluster_size_exponent=d2d_cluster_size_exponent,
                freshness_exponent=d2d_freshness_exponent,
                threshold_gain=d2d_threshold_gain,
            )
        elif optimized_d2d_access_mode == "hybrid":
            optimized_probability_d2d = _hybrid_utility_access_probability(
                aggregate_updates=aggregate_updates_model_3,
                cluster_sizes=active_member_counts_3,
                freshness=optimized_d2d_freshness,
                reference_direction=optimized_d2d_reference_direction,
                cluster_mask=cluster_mask,
                n_channels=n_channels,
                pcomp=pcomp,
                fixed_access_probability=access_probability_d2d,
                floor_fraction=d2d_access_floor_fraction,
                norm_exponent=d2d_norm_exponent,
                cluster_size_exponent=d2d_cluster_size_exponent,
                freshness_exponent=d2d_freshness_exponent,
                novelty_exponent=d2d_novelty_exponent,
                novelty_floor=d2d_novelty_floor,
                load_target_factor=d2d_load_target_factor,
                load_allocation_mode=optimized_d2d_load_allocation_mode,
                redistribution_fraction=d2d_redistribution_fraction,
                redistribution_trigger_ratio=d2d_redistribution_trigger_ratio,
                density_trigger_threshold=d2d_density_trigger_threshold,
                dense_trigger_ratio=d2d_dense_trigger_ratio,
                clusterized_devices_fraction=clusterized_devices_fraction,
                optimized_success_ewma=optimized_d2d_success_ewma,
                fixed_success_target=current_expected_fixed_d2d_ch_successes,
            )
        elif optimized_d2d_access_mode == "adaptive_diversity":
            optimized_probability_d2d = _adaptive_diversity_access_probability(
                aggregate_updates=aggregate_updates_model_3,
                cluster_sizes=active_member_counts_3,
                freshness=optimized_d2d_freshness,
                reference_direction=optimized_d2d_reference_direction,
                cluster_mask=cluster_mask,
                n_channels=n_channels,
                pcomp=pcomp,
                fixed_access_probability=access_probability_d2d,
                floor_fraction=d2d_access_floor_fraction,
                early_norm_exponent=d2d_norm_exponent,
                late_norm_exponent=d2d_late_norm_exponent,
                cluster_size_exponent=d2d_cluster_size_exponent,
                early_freshness_exponent=d2d_freshness_exponent,
                late_freshness_exponent=d2d_late_freshness_exponent,
                novelty_exponent=d2d_novelty_exponent,
                novelty_floor=d2d_novelty_floor,
                load_target_factor=d2d_load_target_factor,
                load_allocation_mode=optimized_d2d_load_allocation_mode,
                redistribution_fraction=d2d_redistribution_fraction,
                redistribution_trigger_ratio=d2d_redistribution_trigger_ratio,
                density_trigger_threshold=d2d_density_trigger_threshold,
                dense_trigger_ratio=d2d_dense_trigger_ratio,
                clusterized_devices_fraction=clusterized_devices_fraction,
                optimized_success_ewma=optimized_d2d_success_ewma,
                fixed_success_target=current_expected_fixed_d2d_ch_successes,
                switch_fraction=d2d_adaptive_switch_fraction,
                switch_gain=d2d_adaptive_switch_gain,
                iteration_index=iteration_index,
                max_iterations=max_iterations_t,
            )
        else:
            optimized_probability_d2d = jnp.where(
                iteration_index == 0,
                access_probability_d2d,
                _safe_access_probability(aggregate_norms_model_3, psi_d2d),
            )
            optimized_probability_d2d = jnp.minimum(
                optimized_probability_d2d,
                pcomp,
            )
            optimized_probability_d2d = _apply_access_floor(
                probability=optimized_probability_d2d,
                floor_fraction=d2d_access_floor_fraction,
                fixed_access_probability=access_probability_d2d,
                pcomp=pcomp,
            )
        optimized_cluster_draws = device_draws[optimized_d2d_heads]
        candidates_3_d2d_mask = (
            optimized_cluster_draws < optimized_probability_d2d
        ) & cluster_mask
        candidates_3_d2d_mask = candidates_3_d2d_mask & ch_can_attempt_3
        success_3_d2d, candidates_3_d2d = _successful_from_draws(
            channel_key_3_d2d,
            optimized_cluster_draws,
            optimized_probability_d2d,
            cluster_mask & ch_can_attempt_3,
            n_channels,
            link_key=optimized_d2d_link_key,
            link_success_probability=ch_bs_success_probability_3,
        )
        gradient_3_d2d = jnp.sum(
            jnp.where(success_3_d2d[:, None], aggregate_updates_model_3, 0.0),
            axis=0,
        )
        upload_3_d2d = jnp.sum(
            jnp.where(success_3_d2d, active_member_counts_3, 0)
        ).astype(jnp.int32)
        ch_upload_3_d2d = jnp.sum(success_3_d2d).astype(jnp.int32)
        d2d_load_error = candidates_3_d2d.astype(users_input__x.dtype) - n_channels
        threshold_load_error = d2d_load_error / jnp.asarray(
            max(n_channels, 1),
            dtype=users_input__x.dtype,
        )
        if optimized_d2d_access_mode == "max_weight":
            next_psi_d2d = psi_d2d + step_size * threshold_load_error
        else:
            next_psi_d2d = psi_d2d + step_size * d2d_load_error
        next_optimized_d2d_freshness = jnp.where(
            cluster_mask,
            optimized_d2d_freshness + 1.0,
            0.0,
        )
        next_optimized_d2d_freshness = jnp.where(
            success_3_d2d,
            1.0,
            next_optimized_d2d_freshness,
        )
        successful_reference_update = jnp.sum(
            jnp.where(success_3_d2d[:, None], aggregate_updates_model_3, 0.0),
            axis=0,
        )
        has_optimized_d2d_success = jnp.any(success_3_d2d)
        blended_reference_direction = (
            d2d_reference_decay * optimized_d2d_reference_direction
            + (1.0 - d2d_reference_decay) * successful_reference_update
        )
        if optimized_d2d_access_mode in {"hybrid", "adaptive_diversity"}:
            next_optimized_d2d_reference_direction = jnp.where(
                has_optimized_d2d_success,
                blended_reference_direction,
                optimized_d2d_reference_direction,
            )
        else:
            next_optimized_d2d_reference_direction = optimized_d2d_reference_direction
        next_optimized_d2d_success_ewma = (
            d2d_throughput_ewma_decay * optimized_d2d_success_ewma
            + (1.0 - d2d_throughput_ewma_decay)
            * ch_upload_3_d2d.astype(users_input__x.dtype)
        )

        def direct_attempt_drain(attempt_mask):
            # Direct polling/fixed/optimized curves pay a direct device-to-BS
            # cost for attempted transmissions.  The cost is charged on
            # attempts, not only successes, because RF energy is consumed before
            # collision/decoding outcome is known.
            return attempt_mask.astype(users_input__x.dtype) * direct_bs_energy_by_device

        def polling_direct_attempt_drain():
            drain = jnp.zeros(k_devices, dtype=users_input__x.dtype)
            return drain.at[scheduled_users].add(
                polling_compute_success.astype(users_input__x.dtype)
                * direct_bs_energy_by_device[scheduled_users]
            )

        def d2d_attempt_drain(
            cluster_attempt_counts,
            active_member_mask,
            heads,
            member_energy,
        ):
            # CHs pay for every BS-uplink attempt.  Members pay when they have
            # an active update and their cluster actually tries to use the
            # aggregate in this iteration.  This preserves the HFL hierarchy:
            # members pay D2D local cost, while the CH pays the long-range
            # aggregate uplink cost.
            attempt_counts = jnp.where(
                cluster_mask,
                cluster_attempt_counts.astype(users_input__x.dtype),
                0.0,
            )
            active_counts = jnp.sum(active_member_mask, axis=1).astype(
                users_input__x.dtype
            )
            is_scenario_ch = safe_members == heads[:, None]
            active_non_ch_counts = jnp.sum(
                active_member_mask & (~is_scenario_ch),
                axis=1,
            ).astype(users_input__x.dtype)
            ch_attempt_energy = ch_bs_energy_by_device[heads]
            if energy_model == "first_order_radio":
                ch_attempt_energy = (
                    ch_attempt_energy
                    + active_non_ch_counts * member_rx_energy
                    + active_counts * aggregate_update_energy
                )
            ch_drain = jnp.zeros(k_devices, dtype=users_input__x.dtype)
            ch_drain = ch_drain.at[heads].add(
                attempt_counts * ch_attempt_energy
            )

            member_attempt_mask = active_member_mask & (~is_scenario_ch) & (
                attempt_counts[:, None] > 0.0
            )
            member_drain = jnp.zeros(k_devices, dtype=users_input__x.dtype)
            member_drain = member_drain.at[safe_members.reshape(-1)].add(
                member_attempt_mask.reshape(-1).astype(users_input__x.dtype)
                * member_energy.reshape(-1)
            )
            return ch_drain + member_drain, ch_drain

        def rotation_control_drain(heads):
            per_cluster_cost = jnp.where(
                should_rotate_d2d_heads & cluster_mask,
                energy_rotation_control_cost,
                0.0,
            )
            drain = jnp.zeros(k_devices, dtype=users_input__x.dtype)
            return drain.at[heads].add(per_cluster_cost)

        polling_d2d_attempt_counts = jnp.zeros(
            polling_d2d_heads.shape,
            dtype=users_input__x.dtype,
        ).at[scheduled_clusters].add(
            polling_d2d_attempt.astype(users_input__x.dtype)
        )
        fixed_d2d_attempt_counts = candidates_2_d2d_mask.astype(users_input__x.dtype)
        optimized_d2d_attempt_counts = candidates_3_d2d_mask.astype(
            users_input__x.dtype
        )
        polling_d2d_drain, polling_d2d_ch_drain = d2d_attempt_drain(
            polling_d2d_attempt_counts,
            active_member_mask_1,
            polling_d2d_heads,
            member_energy_1,
        )
        fixed_d2d_drain, fixed_d2d_ch_drain = d2d_attempt_drain(
            fixed_d2d_attempt_counts,
            active_member_mask_2,
            fixed_d2d_heads,
            member_energy_2,
        )
        optimized_d2d_drain, optimized_d2d_ch_drain = d2d_attempt_drain(
            optimized_d2d_attempt_counts,
            active_member_mask_3,
            optimized_d2d_heads,
            member_energy_3,
        )
        polling_rotation_drain = rotation_control_drain(polling_d2d_heads)
        fixed_rotation_drain = rotation_control_drain(fixed_d2d_heads)
        optimized_rotation_drain = rotation_control_drain(optimized_d2d_heads)
        polling_d2d_drain = polling_d2d_drain + polling_rotation_drain
        fixed_d2d_drain = fixed_d2d_drain + fixed_rotation_drain
        optimized_d2d_drain = optimized_d2d_drain + optimized_rotation_drain
        polling_d2d_ch_drain = polling_d2d_ch_drain + polling_rotation_drain
        fixed_d2d_ch_drain = fixed_d2d_ch_drain + fixed_rotation_drain
        optimized_d2d_ch_drain = optimized_d2d_ch_drain + optimized_rotation_drain

        battery_drain = jnp.stack(
            [
                polling_direct_attempt_drain(),
                direct_attempt_drain(candidates_2_mask),
                direct_attempt_drain(candidates_3_mask),
                polling_d2d_drain,
                fixed_d2d_drain,
                optimized_d2d_drain,
            ],
            axis=0,
        )
        next_batteries = jnp.clip(
            current_batteries - energy_enabled * battery_drain,
            0.0,
            1.0,
        )
        d2d_ch_drain = jnp.stack(
            [
                polling_d2d_ch_drain,
                fixed_d2d_ch_drain,
                optimized_d2d_ch_drain,
            ],
            axis=0,
        )
        actual_d2d_ch_drain = jnp.minimum(
            energy_enabled * d2d_ch_drain,
            current_batteries[3:],
        )
        next_clusterhead_energy_totals = (
            clusterhead_energy_totals + actual_d2d_ch_drain
        )

        gradients = jnp.stack(
            [
                gradient_1,
                gradient_2,
                gradient_3,
                gradient_1_d2d,
                gradient_2_d2d,
                gradient_3_d2d,
            ],
            axis=0,
        )
        next_weights = jax.vmap(apply_gradient)(current_weights, gradients)

        iteration_uploads = jnp.asarray(
            [
                upload_1,
                upload_2,
                upload_3,
                upload_1_d2d,
                upload_2_d2d,
                upload_3_d2d,
            ],
            dtype=jnp.int32,
        )
        next_upload_totals = upload_totals + iteration_uploads
        next_clusterhead_upload_totals = clusterhead_upload_totals + jnp.asarray(
            [ch_upload_1_d2d, ch_upload_2_d2d, ch_upload_3_d2d],
            dtype=jnp.int32,
        )

        polling_device_success = jnp.zeros(k_devices, dtype=jnp.bool_).at[
            scheduled_users
        ].set(polling_success)
        polling_cluster_success = (
            jnp.zeros(cluster_mask.shape, dtype=jnp.int32)
            .at[scheduled_clusters]
            .add(polling_success_d2d.astype(jnp.int32))
            > 0
        )
        next_non_d2d_aoi = jnp.stack(
            [
                jnp.where(polling_device_success, 1.0, non_d2d_aoi[0] + 1.0),
                jnp.where(success_2, 1.0, non_d2d_aoi[1] + 1.0),
                jnp.where(success_3, 1.0, non_d2d_aoi[2] + 1.0),
            ],
            axis=0,
        )
        next_d2d_aoi = jnp.stack(
            [
                jnp.where(
                    cluster_mask,
                    jnp.where(polling_cluster_success, 1.0, d2d_aoi[0] + 1.0),
                    0.0,
                ),
                jnp.where(
                    cluster_mask,
                    jnp.where(success_2_d2d, 1.0, d2d_aoi[1] + 1.0),
                    0.0,
                ),
                jnp.where(
                    cluster_mask,
                    jnp.where(success_3_d2d, 1.0, d2d_aoi[2] + 1.0),
                    0.0,
                ),
            ],
            axis=0,
        )
        (
            mean_aoi,
            peak_aoi,
            p75_aoi,
            p90_aoi,
            p95_aoi,
            stale_fraction_50,
            stale_fraction_75,
            stale_fraction_100,
        ) = scenario_aoi_summary(
            next_non_d2d_aoi,
            next_d2d_aoi,
            iteration_index + jnp.asarray(1, dtype=jnp.int32),
        )

        error_norms = jnp.linalg.norm(next_weights - weights_vector__w[None, :], axis=1)
        next_state = (
            key,
            next_weights,
            next_batteries,
            d2d_heads,
            next_clusterhead_energy_totals,
            next_psi,
            next_psi_d2d,
            next_upload_totals,
            next_clusterhead_upload_totals,
            next_optimized_d2d_freshness,
            next_optimized_d2d_reference_direction,
            next_optimized_d2d_success_ewma,
            next_non_d2d_aoi,
            next_d2d_aoi,
        )
        mean_battery = jnp.mean(next_batteries, axis=1)
        # Gather the currently elected CH battery for each D2D scenario.
        #
        # next_batteries[3:]: float[3, K]
        #   Scenario-specific battery state for polling+D2D, fixed+D2D, and
        #   optimized+D2D.
        # d2d_heads: int[3, max_clusters]
        #   Current elected CH device id per D2D scenario and cluster row.
        #
        # ``take_along_axis`` is used instead of ``next_batteries[3:, d2d_heads]``
        # because mixed slicing and advanced indexing would not mean "take each
        # scenario row at its own CH columns" under JAX/NumPy indexing rules.
        d2d_current_head_battery = jnp.take_along_axis(
            next_batteries[3:],
            d2d_heads,
            axis=1,
        )
        d2d_clusterhead_battery = jnp.sum(
            jnp.where(
                cluster_mask[None, :],
                d2d_current_head_battery,
                0.0,
            ),
            axis=1,
        ) / jnp.maximum(
            active_clusterhead_count,
            jnp.asarray(1.0, dtype=users_input__x.dtype),
        )
        total_energy_used = jnp.sum(
            initial_device_battery[None, :] - next_batteries,
            axis=1,
        )
        mean_energy_used = total_energy_used / jnp.asarray(
            k_devices,
            dtype=users_input__x.dtype,
        )
        energy_efficiency = jnp.where(
            total_energy_used > jnp.asarray(1e-12, dtype=users_input__x.dtype),
            next_upload_totals.astype(users_input__x.dtype) / total_energy_used,
            0.0,
        )
        mean_clusterhead_energy_used = (
            jnp.sum(next_clusterhead_energy_totals, axis=1)
            / jnp.maximum(
                active_clusterhead_count,
                jnp.asarray(1.0, dtype=users_input__x.dtype),
            )
        )
        trace_row = (
            error_norms,
            next_upload_totals,
            next_clusterhead_upload_totals,
            mean_battery,
            d2d_clusterhead_battery,
            mean_energy_used,
            energy_efficiency,
            mean_clusterhead_energy_used,
            mean_aoi,
            peak_aoi,
            p75_aoi,
            p90_aoi,
            p95_aoi,
            stale_fraction_50,
            stale_fraction_75,
            stale_fraction_100,
        )
        return next_state, trace_row

    initial_psi_d2d = (
        jnp.asarray(0.5, dtype=users_input__x.dtype)
        if optimized_d2d_access_mode == "max_weight"
        else jnp.asarray(0.0, dtype=users_input__x.dtype)
    )

    initial_state = (
        scan_key,
        weights,
        jnp.stack([initial_device_battery] * 6, axis=0),
        jnp.stack([cluster_heads] * 3, axis=0),
        jnp.zeros((3, k_devices), dtype=users_input__x.dtype),
        jnp.asarray(0.0, dtype=users_input__x.dtype),
        initial_psi_d2d,
        jnp.zeros(6, dtype=jnp.int32),
        jnp.zeros(3, dtype=jnp.int32),
        jnp.where(
            cluster_mask,
            jnp.ones_like(cluster_sizes, dtype=users_input__x.dtype),
            0.0,
        ),
        jnp.zeros(data_dimension, dtype=users_input__x.dtype),
        # Start from the fixed-D2D reference throughput to avoid an artificial
        # cold-start burst of redistribution before any ACK history exists.
        initial_expected_fixed_d2d_ch_successes,
        jnp.ones((3, k_devices), dtype=users_input__x.dtype),
        jnp.where(
            cluster_mask[None, :],
            jnp.ones((3, cluster_mask.shape[0]), dtype=users_input__x.dtype),
            0.0,
        ),
    )

    (
        _,
        (
            error_norms,
            upload_totals,
            clusterhead_upload_totals,
            mean_battery,
            mean_clusterhead_battery,
            mean_energy_used,
            energy_efficiency,
            mean_clusterhead_energy_used,
            mean_aoi,
            peak_aoi,
            p75_aoi,
            p90_aoi,
            p95_aoi,
            stale_fraction_50,
            stale_fraction_75,
            stale_fraction_100,
        ),
    ) = jax.lax.scan(
        scan_iteration,
        initial_state,
        jnp.arange(max_iterations_t, dtype=jnp.int32),
    )

    if checkpoints is None:
        checkpoint_indices = jnp.arange(max_iterations_t, dtype=jnp.int32)
    else:
        checkpoint_indices = jnp.asarray(checkpoints, dtype=jnp.int32) - 1
        checkpoint_indices = jnp.clip(checkpoint_indices, 0, max_iterations_t - 1)

    return JaxTraceResult(
        clusterized_devices_rate=clusters.clusterized_devices_rate,
        error_norms=error_norms[checkpoint_indices],
        successful_uploads=upload_totals[checkpoint_indices],
        successful_clusterhead_uploads=clusterhead_upload_totals[checkpoint_indices],
        mean_battery=mean_battery[checkpoint_indices],
        mean_clusterhead_battery=mean_clusterhead_battery[checkpoint_indices],
        mean_energy_used=mean_energy_used[checkpoint_indices],
        energy_efficiency=energy_efficiency[checkpoint_indices],
        mean_clusterhead_energy_used=mean_clusterhead_energy_used[checkpoint_indices],
        mean_aoi=mean_aoi[checkpoint_indices],
        peak_aoi=peak_aoi[checkpoint_indices],
        p75_aoi=p75_aoi[checkpoint_indices],
        p90_aoi=p90_aoi[checkpoint_indices],
        p95_aoi=p95_aoi[checkpoint_indices],
        stale_fraction_50=stale_fraction_50[checkpoint_indices],
        stale_fraction_75=stale_fraction_75[checkpoint_indices],
        stale_fraction_100=stale_fraction_100[checkpoint_indices],
        checkpoints=checkpoint_indices + 1,
    )


def error_calculator(
    number_of_mobile_devices__k: int,
    data_dimension__L: int,
    number_of_parallel_channels__M: int,
    probability_that_user_can_compute_its_local_update__pcomp: float,
    number_of_iterations__t: int,
    learning_rate__u1: float,
    step_size__u: float,
    clusters_list: list,
    seed=None,
    normalize_by_k: bool = False,
    d2d_member_compute_probability: float = 1.0,
    d2d_member_link_success_probability: float = 1.0,
    d2d_ch_bs_success_mode: str = "none",
    d2d_ch_bs_min_success_probability: float = 0.20,
    d2d_ch_bs_pathloss_exponent: float = 2.0,
    d2d_ch_bs_battery_exponent: float = 0.0,
    d2d_ch_bs_reference_snr: float = 100000.0,
    d2d_ch_bs_snr_threshold: float = 1.0,
    device_bs_success_mode: str = "none",
    device_bs_min_success_probability: float = 0.20,
    device_bs_pathloss_exponent: float = 2.0,
    device_bs_battery_exponent: float = 0.0,
    device_bs_reference_snr: float = 100000.0,
    device_bs_snr_threshold: float = 1.0,
    energy_drain_mode: str = "none",
    energy_model: str = "constant",
    battery_feasibility_mode: str = "off",
    energy_direct_bs_cost: float = 0.0,
    energy_d2d_member_cost: float = 0.0,
    energy_ch_bs_cost: float = 0.0,
    energy_electronics_cost: float = 0.0002,
    energy_bs_amplifier_cost: float = 2e-8,
    energy_d2d_amplifier_cost: float = 1e-6,
    energy_bs_pathloss_exponent: float = 2.0,
    energy_d2d_pathloss_exponent: float = 2.0,
    energy_aggregation_cost: float = 0.00002,
    energy_update_size: float = 1.0,
    energy_aggregate_size: float = 1.0,
    energy_rotation_control_cost: float = 0.0,
    d2d_ch_rotation_mode: str = "static",
    d2d_ch_rotation_interval: int = 10,
    d2d_energy_efficiency_level: str = "balanced",
    device_coords=None,
    device_radius=None,
    device_distance_to_bs=None,
    device_battery=None,
    optimized_access_floor_fraction: float = 0.0,
    optimized_d2d_access_floor_fraction: float = 0.0,
    optimized_d2d_access_mode: str = "norm",
    optimized_d2d_norm_exponent: float = 1.0,
    optimized_d2d_cluster_size_exponent: float = 1.0,
    optimized_d2d_freshness_exponent: float = 0.5,
    optimized_d2d_threshold_gain: float = 8.0,
    optimized_d2d_novelty_exponent: float = 1.0,
    optimized_d2d_novelty_floor: float = 0.25,
    optimized_d2d_reference_decay: float = 0.90,
    optimized_d2d_load_target_factor: float = 1.0,
    optimized_d2d_load_allocation_mode: str = "conditional_selective_water_filling",
    optimized_d2d_redistribution_fraction: float = 0.5,
    optimized_d2d_redistribution_trigger_ratio: float = 0.95,
    optimized_d2d_density_trigger_threshold: float = 0.95,
    optimized_d2d_dense_trigger_ratio: float = 0.90,
    optimized_d2d_throughput_ewma_decay: float = 0.90,
    optimized_d2d_late_norm_exponent: float = 1.25,
    optimized_d2d_late_freshness_exponent: float = 1.0,
    optimized_d2d_adaptive_switch_fraction: float = 0.30,
    optimized_d2d_adaptive_switch_gain: float = 12.0,
    optimized_d2d_aoi_weight: float = 0.5,
    optimized_d2d_aoi_exponent: float = 1.0,
    optimized_d2d_aoi_threshold_fraction: float = 0.75,
    dtype=None,
):
    """Compatibility wrapper returning the legacy 16-value final tuple."""
    clusters = prepare_clusters_for_jax(clusters_list)
    result = error_calculator_trace_jax(
        number_of_mobile_devices__k=number_of_mobile_devices__k,
        data_dimension__L=data_dimension__L,
        number_of_parallel_channels__M=number_of_parallel_channels__M,
        probability_that_user_can_compute_its_local_update__pcomp=probability_that_user_can_compute_its_local_update__pcomp,
        max_iterations_t=number_of_iterations__t,
        learning_rate__u1=learning_rate__u1,
        step_size__u=step_size__u,
        clusters=clusters,
        seed=seed,
        normalize_by_k=normalize_by_k,
        d2d_member_compute_probability=d2d_member_compute_probability,
        d2d_member_link_success_probability=d2d_member_link_success_probability,
        d2d_ch_bs_success_mode=d2d_ch_bs_success_mode,
        d2d_ch_bs_min_success_probability=d2d_ch_bs_min_success_probability,
        d2d_ch_bs_pathloss_exponent=d2d_ch_bs_pathloss_exponent,
        d2d_ch_bs_battery_exponent=d2d_ch_bs_battery_exponent,
        d2d_ch_bs_reference_snr=d2d_ch_bs_reference_snr,
        d2d_ch_bs_snr_threshold=d2d_ch_bs_snr_threshold,
        device_bs_success_mode=device_bs_success_mode,
        device_bs_min_success_probability=device_bs_min_success_probability,
        device_bs_pathloss_exponent=device_bs_pathloss_exponent,
        device_bs_battery_exponent=device_bs_battery_exponent,
        device_bs_reference_snr=device_bs_reference_snr,
        device_bs_snr_threshold=device_bs_snr_threshold,
        energy_drain_mode=energy_drain_mode,
        energy_model=energy_model,
        battery_feasibility_mode=battery_feasibility_mode,
        energy_direct_bs_cost=energy_direct_bs_cost,
        energy_d2d_member_cost=energy_d2d_member_cost,
        energy_ch_bs_cost=energy_ch_bs_cost,
        energy_electronics_cost=energy_electronics_cost,
        energy_bs_amplifier_cost=energy_bs_amplifier_cost,
        energy_d2d_amplifier_cost=energy_d2d_amplifier_cost,
        energy_bs_pathloss_exponent=energy_bs_pathloss_exponent,
        energy_d2d_pathloss_exponent=energy_d2d_pathloss_exponent,
        energy_aggregation_cost=energy_aggregation_cost,
        energy_update_size=energy_update_size,
        energy_aggregate_size=energy_aggregate_size,
        energy_rotation_control_cost=energy_rotation_control_cost,
        d2d_ch_rotation_mode=d2d_ch_rotation_mode,
        d2d_ch_rotation_interval=d2d_ch_rotation_interval,
        d2d_energy_efficiency_level=d2d_energy_efficiency_level,
        device_coords=device_coords,
        device_radius=device_radius,
        device_distance_to_bs=device_distance_to_bs,
        device_battery=device_battery,
        optimized_access_floor_fraction=optimized_access_floor_fraction,
        optimized_d2d_access_floor_fraction=optimized_d2d_access_floor_fraction,
        optimized_d2d_access_mode=optimized_d2d_access_mode,
        optimized_d2d_norm_exponent=optimized_d2d_norm_exponent,
        optimized_d2d_cluster_size_exponent=optimized_d2d_cluster_size_exponent,
        optimized_d2d_freshness_exponent=optimized_d2d_freshness_exponent,
        optimized_d2d_threshold_gain=optimized_d2d_threshold_gain,
        optimized_d2d_novelty_exponent=optimized_d2d_novelty_exponent,
        optimized_d2d_novelty_floor=optimized_d2d_novelty_floor,
        optimized_d2d_reference_decay=optimized_d2d_reference_decay,
        optimized_d2d_load_target_factor=optimized_d2d_load_target_factor,
        optimized_d2d_load_allocation_mode=optimized_d2d_load_allocation_mode,
        optimized_d2d_redistribution_fraction=optimized_d2d_redistribution_fraction,
        optimized_d2d_redistribution_trigger_ratio=(
            optimized_d2d_redistribution_trigger_ratio
        ),
        optimized_d2d_density_trigger_threshold=(
            optimized_d2d_density_trigger_threshold
        ),
        optimized_d2d_dense_trigger_ratio=optimized_d2d_dense_trigger_ratio,
        optimized_d2d_throughput_ewma_decay=optimized_d2d_throughput_ewma_decay,
        optimized_d2d_late_norm_exponent=optimized_d2d_late_norm_exponent,
        optimized_d2d_late_freshness_exponent=optimized_d2d_late_freshness_exponent,
        optimized_d2d_adaptive_switch_fraction=optimized_d2d_adaptive_switch_fraction,
        optimized_d2d_adaptive_switch_gain=optimized_d2d_adaptive_switch_gain,
        optimized_d2d_aoi_weight=optimized_d2d_aoi_weight,
        optimized_d2d_aoi_exponent=optimized_d2d_aoi_exponent,
        optimized_d2d_aoi_threshold_fraction=optimized_d2d_aoi_threshold_fraction,
        checkpoints=[number_of_iterations__t],
        dtype=dtype,
    )
    errors = np.asarray(result.error_norms)[0]
    uploads = np.asarray(result.successful_uploads)[0]
    clusterhead_uploads = np.asarray(result.successful_clusterhead_uploads)[0]
    return (
        float(np.asarray(result.clusterized_devices_rate)),
        float(errors[0]),
        float(errors[1]),
        float(errors[2]),
        float(errors[3]),
        float(errors[4]),
        float(errors[5]),
        int(uploads[0]),
        int(uploads[1]),
        int(uploads[2]),
        int(uploads[3]),
        int(uploads[4]),
        int(uploads[5]),
        int(clusterhead_uploads[0]),
        int(clusterhead_uploads[1]),
        int(clusterhead_uploads[2]),
    )
