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
    """Metric arrays produced by ``error_calculator_trace_jax``."""

    clusterized_devices_rate: Any
    error_norms: Any
    successful_uploads: Any
    successful_clusterhead_uploads: Any
    checkpoints: Any


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


def _successful_from_draws(channel_key, random_draws, probability, eligibility, n_channels):
    """Resolve multichannel ALOHA contenders for one model.

    ``random_draws`` supplies the compute/access draw.  ``probability`` may be
    scalar or vector.  Each candidate picks one channel; a candidate succeeds
    only if no other candidate picked the same channel.
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
    successful = candidates & (channel_counts[selected_channels] == 1)
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

    dtype = jnp.float32 if dtype is None else dtype
    pcomp = jnp.asarray(
        probability_that_user_can_compute_its_local_update__pcomp,
        dtype=dtype,
    )
    learning_rate = jnp.asarray(learning_rate__u1, dtype=dtype)
    step_size = jnp.asarray(step_size__u, dtype=dtype)
    d2d_compute_probability = jnp.asarray(d2d_member_compute_probability, dtype=dtype)
    d2d_link_probability = jnp.asarray(d2d_member_link_success_probability, dtype=dtype)

    key = _key_from_seed(seed)
    data_key, true_weight_key, init_weight_key, scan_key = jax.random.split(key, 4)

    k_devices = int(number_of_mobile_devices__k)
    data_dimension = int(data_dimension__L)
    n_channels = int(number_of_parallel_channels__M)
    cluster_members = clusters.cluster_members.astype(jnp.int32)
    cluster_sizes = clusters.cluster_sizes.astype(jnp.int32)
    cluster_mask = clusters.cluster_mask
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

    member_positions = jnp.arange(max_cluster_size, dtype=jnp.int32)[None, :]
    member_mask = member_positions < cluster_sizes[:, None]
    cluster_heads = jnp.where(cluster_mask, cluster_members[:, 0], 0)

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

    def scan_iteration(state, iteration_index):
        (
            key,
            current_weights,
            psi,
            psi_d2d,
            upload_totals,
            clusterhead_upload_totals,
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
        ) = jax.random.split(key, 9)

        device_draws = jax.random.uniform(draw_key, (k_devices,), dtype=dtype)
        channel_draws = jax.random.uniform(polling_key, (n_channels,), dtype=dtype)

        # Optional member-to-CH realism.  Position 0 is the CH, so it is active
        # whenever its CH-to-BS model selects the cluster.
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
        is_cluster_head_position = member_positions == 0
        active_member_mask = (
            member_mask
            & cluster_mask[:, None]
            & (
                is_cluster_head_position
                | (
                    (active_compute_draws < d2d_compute_probability)
                    & (active_link_draws < d2d_link_probability)
                )
            )
        )
        active_member_counts = jnp.sum(active_member_mask, axis=1).astype(jnp.int32)

        local_updates = local_updates_for_weights(current_weights)
        aggregate_updates_model_1 = aggregate_cluster_updates(
            local_updates[:, 3, :],
            active_member_mask,
        )
        aggregate_updates_model_2 = aggregate_cluster_updates(
            local_updates[:, 4, :],
            active_member_mask,
        )
        aggregate_updates_model_3 = aggregate_cluster_updates(
            local_updates[:, 5, :],
            active_member_mask,
        )
        aggregate_norms_model_3 = jnp.linalg.norm(aggregate_updates_model_3, axis=1)

        # Model 1: polling without D2D.
        scheduled_users = (
            iteration_index * n_channels + jnp.arange(n_channels, dtype=jnp.int32)
        ) % k_devices
        polling_success = channel_draws < pcomp
        gradient_1 = jnp.sum(
            jnp.where(polling_success[:, None], local_updates[scheduled_users, 0, :], 0.0),
            axis=0,
        )
        upload_1 = jnp.sum(polling_success).astype(jnp.int32)

        # Model 1 with D2D: polling schedules CH rows instead of device IDs.
        scheduled_clusters = (
            iteration_index * n_channels + jnp.arange(n_channels, dtype=jnp.int32)
        ) % number_of_clusterheads
        gradient_1_d2d = jnp.sum(
            jnp.where(
                polling_success[:, None],
                aggregate_updates_model_1[scheduled_clusters],
                0.0,
            ),
            axis=0,
        )
        upload_1_d2d = jnp.sum(
            jnp.where(polling_success, active_member_counts[scheduled_clusters], 0)
        ).astype(jnp.int32)
        ch_upload_1_d2d = upload_1

        # Model 2: fixed ALOHA without and with D2D.
        threshold = jnp.minimum(
            access_probability,
            pcomp,
        )
        success_2, _ = _successful_from_draws(
            channel_key_2,
            device_draws,
            threshold,
            jnp.ones(k_devices, dtype=jnp.bool_),
            n_channels,
        )
        gradient_2 = jnp.sum(jnp.where(success_2[:, None], local_updates[:, 1, :], 0.0), axis=0)
        upload_2 = jnp.sum(success_2).astype(jnp.int32)

        cluster_draws = device_draws[cluster_heads]
        threshold_d2d = jnp.minimum(
            access_probability_d2d,
            pcomp,
        )
        success_2_d2d, _ = _successful_from_draws(
            channel_key_2_d2d,
            cluster_draws,
            threshold_d2d,
            cluster_mask,
            n_channels,
        )
        gradient_2_d2d = jnp.sum(
            jnp.where(success_2_d2d[:, None], aggregate_updates_model_2, 0.0),
            axis=0,
        )
        upload_2_d2d = jnp.sum(
            jnp.where(success_2_d2d, active_member_counts, 0)
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
        success_3, candidates_3 = _successful_from_draws(
            channel_key_3,
            device_draws,
            optimized_probability,
            jnp.ones(k_devices, dtype=jnp.bool_),
            n_channels,
        )
        gradient_3 = jnp.sum(jnp.where(success_3[:, None], local_updates[:, 2, :], 0.0), axis=0)
        upload_3 = jnp.sum(success_3).astype(jnp.int32)
        next_psi = psi + step_size * (
            candidates_3.astype(users_input__x.dtype) - n_channels
        )

        optimized_probability_d2d = jnp.where(
            iteration_index == 0,
            access_probability_d2d,
            _safe_access_probability(aggregate_norms_model_3, psi_d2d),
        )
        optimized_probability_d2d = jnp.minimum(
            optimized_probability_d2d,
            pcomp,
        )
        success_3_d2d, candidates_3_d2d = _successful_from_draws(
            channel_key_3_d2d,
            cluster_draws,
            optimized_probability_d2d,
            cluster_mask,
            n_channels,
        )
        gradient_3_d2d = jnp.sum(
            jnp.where(success_3_d2d[:, None], aggregate_updates_model_3, 0.0),
            axis=0,
        )
        upload_3_d2d = jnp.sum(
            jnp.where(success_3_d2d, active_member_counts, 0)
        ).astype(jnp.int32)
        ch_upload_3_d2d = jnp.sum(success_3_d2d).astype(jnp.int32)
        next_psi_d2d = psi_d2d + step_size * (
            candidates_3_d2d.astype(users_input__x.dtype) - n_channels
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

        error_norms = jnp.linalg.norm(next_weights - weights_vector__w[None, :], axis=1)
        next_state = (
            key,
            next_weights,
            next_psi,
            next_psi_d2d,
            next_upload_totals,
            next_clusterhead_upload_totals,
        )
        trace_row = (error_norms, next_upload_totals, next_clusterhead_upload_totals)
        return next_state, trace_row

    initial_state = (
        scan_key,
        weights,
        jnp.asarray(0.0, dtype=users_input__x.dtype),
        jnp.asarray(0.0, dtype=users_input__x.dtype),
        jnp.zeros(6, dtype=jnp.int32),
        jnp.zeros(3, dtype=jnp.int32),
    )

    _, (error_norms, upload_totals, clusterhead_upload_totals) = jax.lax.scan(
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
