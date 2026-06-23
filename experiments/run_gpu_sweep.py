"""Run GPU-first JAX sweeps for the HFL/ALOHA simulation.

The runner batches independent Monte Carlo rounds with ``jax.vmap``.  Each
round generates devices, clusters them, runs one FL trajectory to ``max_t``,
and records metrics at every requested checkpoint.  The output rows aggregate
means and 95% confidence intervals over rounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Clustering.jax_clustering_algorithm import clusterizer_jax, devices_generator_jax
from Models.jax_models_arrangement import error_calculator_trace_jax

try:  # pragma: no cover - optional backend.
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - local env may not have JAX.
    jax = None
    jnp = None
    _JAX_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial assignment.
    _JAX_IMPORT_ERROR = None


SCENARIOS = (
    "polling",
    "fixed_aloha",
    "optimized_aloha",
    "polling_d2d",
    "fixed_aloha_d2d",
    "optimized_aloha_d2d",
)

CLUSTER_QUALITY_METRICS = (
    "number_of_clusters",
    "singleton_count",
    "non_singleton_cluster_count",
    "clustered_devices_count",
    "mean_cluster_size",
    "mean_non_singleton_cluster_size",
)


def _require_jax():
    if jax is None:
        raise ImportError(
            "run_gpu_sweep requires JAX. For Colab GPU, install a matching "
            "JAX CUDA wheel, for example `pip install -U \"jax[cuda13]\"`."
        ) from _JAX_IMPORT_ERROR


def _confidence_interval_95(values):
    """Return mean and normal-approximation 95% CI half-width."""
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))


def _checkpoint_array(iterations, checkpoints):
    if checkpoints is None:
        return np.arange(1, int(iterations) + 1, dtype=np.int32)
    parsed = np.asarray(sorted({int(value) for value in checkpoints}), dtype=np.int32)
    if np.any(parsed < 1) or np.any(parsed > int(iterations)):
        raise ValueError("checkpoints must be in [1, iterations]")
    return parsed


def _configure_precision(args):
    """Set the JAX precision mode before any arrays for this sweep are created."""
    precision = getattr(args, "precision", "float32")
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")

    enable_x64 = precision == "float64"
    jax.config.update("jax_enable_x64", enable_x64)
    return jnp.float64 if enable_x64 else jnp.float32


def _cluster_quality_vector(clusters, dtype):
    """Return scalar structural metrics for one clustered deployment.

    The clustering rate tells us how many devices are not singletons, but it
    does not tell us whether those clustered devices form useful aggregates.
    These metrics let each run report both coverage and quality:

    * number of active cluster rows, equal to the CH count used by D2D ALOHA;
    * singleton count, which drives the clustered-device percentage;
    * number of non-singleton D2D clusters;
    * number of devices inside those non-singleton clusters;
    * average size across all active clusters;
    * average size only among non-singleton clusters.
    """
    active_sizes = jnp.where(clusters.cluster_mask, clusters.cluster_sizes, 0).astype(dtype)
    non_singleton_mask = clusters.cluster_mask & (clusters.cluster_sizes > 1)
    non_singleton_sizes = jnp.where(non_singleton_mask, clusters.cluster_sizes, 0).astype(dtype)

    number_of_clusters = clusters.number_of_clusters.astype(dtype)
    singleton_count = clusters.singleton_count.astype(dtype)
    non_singleton_cluster_count = jnp.sum(non_singleton_mask).astype(dtype)
    clustered_devices_count = jnp.sum(non_singleton_sizes).astype(dtype)
    total_devices = jnp.sum(active_sizes).astype(dtype)

    mean_cluster_size = total_devices / jnp.maximum(number_of_clusters, 1.0)
    mean_non_singleton_cluster_size = clustered_devices_count / jnp.maximum(
        non_singleton_cluster_count,
        1.0,
    )

    return jnp.asarray(
        [
            number_of_clusters,
            singleton_count,
            non_singleton_cluster_count,
            clustered_devices_count,
            mean_cluster_size,
            mean_non_singleton_cluster_size,
        ],
        dtype=dtype,
    )


def run_gpu_sweep(args):
    """Run a batched JAX sweep and return aggregate metric rows."""
    _require_jax()
    compute_dtype = _configure_precision(args)
    checkpoints = _checkpoint_array(args.iterations, getattr(args, "checkpoints", None))
    seeds = np.arange(args.seed, args.seed + args.rounds, dtype=np.uint32)

    def run_one_round(seed):
        devices = devices_generator_jax(
            number_of_devices=args.devices,
            bs_radius=args.bs_radius,
            seed=seed,
            uniform_area=getattr(args, "uniform_area", False),
            dtype=compute_dtype,
        )
        clusters = clusterizer_jax(
            devices=devices,
            device_radius=args.device_radius,
            max_devices_per_cluster=args.max_cluster_size,
            min_devices_per_cluster=args.min_cluster_size,
            clustering_mode=args.clustering_mode,
            strategy=args.clustering_strategy,
            tile_size=args.tile_size,
            repair_passes=args.repair_passes,
            initial_cluster_size=args.initial_cluster_size,
            rotation_repair_passes=args.rotation_repair_passes,
            merge_passes=args.merge_passes,
            cluster_head_selection_mode=args.cluster_head_selection_mode,
            cluster_head_degree_weight=args.cluster_head_degree_weight,
            cluster_head_channel_weight=args.cluster_head_channel_weight,
            cluster_head_battery_weight=args.cluster_head_battery_weight,
        )
        cluster_quality = _cluster_quality_vector(clusters, compute_dtype)
        trace = error_calculator_trace_jax(
            number_of_mobile_devices__k=args.devices,
            data_dimension__L=args.data_dimension,
            number_of_parallel_channels__M=args.channels,
            probability_that_user_can_compute_its_local_update__pcomp=args.pcomp,
            max_iterations_t=args.iterations,
            learning_rate__u1=args.learning_rate,
            step_size__u=args.step_size,
            clusters=clusters,
            seed=seed,
            normalize_by_k=args.normalize_by_k,
            d2d_member_compute_probability=args.d2d_member_compute_probability,
            d2d_member_link_success_probability=args.d2d_member_link_success_probability,
            d2d_ch_bs_success_mode=args.d2d_ch_bs_success_mode,
            d2d_ch_bs_min_success_probability=args.d2d_ch_bs_min_success_probability,
            d2d_ch_bs_pathloss_exponent=args.d2d_ch_bs_pathloss_exponent,
            d2d_ch_bs_battery_exponent=args.d2d_ch_bs_battery_exponent,
            device_bs_success_mode=args.device_bs_success_mode,
            device_bs_min_success_probability=args.device_bs_min_success_probability,
            device_bs_pathloss_exponent=args.device_bs_pathloss_exponent,
            device_bs_battery_exponent=args.device_bs_battery_exponent,
            device_distance_to_bs=devices.distance_to_bs,
            device_battery=devices.battery,
            optimized_access_floor_fraction=args.optimized_access_floor_fraction,
            optimized_d2d_access_floor_fraction=args.optimized_d2d_access_floor_fraction,
            optimized_d2d_access_mode=args.optimized_d2d_access_mode,
            optimized_d2d_norm_exponent=args.optimized_d2d_norm_exponent,
            optimized_d2d_cluster_size_exponent=args.optimized_d2d_cluster_size_exponent,
            optimized_d2d_freshness_exponent=args.optimized_d2d_freshness_exponent,
            optimized_d2d_threshold_gain=args.optimized_d2d_threshold_gain,
            optimized_d2d_novelty_exponent=args.optimized_d2d_novelty_exponent,
            optimized_d2d_novelty_floor=args.optimized_d2d_novelty_floor,
            optimized_d2d_reference_decay=args.optimized_d2d_reference_decay,
            optimized_d2d_load_target_factor=args.optimized_d2d_load_target_factor,
            optimized_d2d_load_allocation_mode=args.optimized_d2d_load_allocation_mode,
            optimized_d2d_redistribution_fraction=args.optimized_d2d_redistribution_fraction,
            optimized_d2d_redistribution_trigger_ratio=(
                args.optimized_d2d_redistribution_trigger_ratio
            ),
            optimized_d2d_density_trigger_threshold=(
                args.optimized_d2d_density_trigger_threshold
            ),
            optimized_d2d_dense_trigger_ratio=args.optimized_d2d_dense_trigger_ratio,
            optimized_d2d_throughput_ewma_decay=(
                args.optimized_d2d_throughput_ewma_decay
            ),
            optimized_d2d_late_norm_exponent=args.optimized_d2d_late_norm_exponent,
            optimized_d2d_late_freshness_exponent=(
                args.optimized_d2d_late_freshness_exponent
            ),
            optimized_d2d_adaptive_switch_fraction=(
                args.optimized_d2d_adaptive_switch_fraction
            ),
            optimized_d2d_adaptive_switch_gain=args.optimized_d2d_adaptive_switch_gain,
            checkpoints=checkpoints,
            dtype=compute_dtype,
        )
        return (
            trace.error_norms,
            trace.successful_uploads,
            trace.successful_clusterhead_uploads,
            trace.clusterized_devices_rate,
            cluster_quality,
        )

    # JIT compiles one batched program specialized to the experiment dimensions.
    batched_runner = jax.jit(jax.vmap(run_one_round))
    started = time.perf_counter()
    (
        error_norms,
        uploads,
        clusterhead_uploads,
        cluster_rates,
        cluster_quality,
    ) = batched_runner(seeds)
    jax.block_until_ready(error_norms)
    elapsed_seconds = time.perf_counter() - started

    error_norms = np.asarray(error_norms)
    uploads = np.asarray(uploads)
    clusterhead_uploads = np.asarray(clusterhead_uploads)
    cluster_rates = np.asarray(cluster_rates)
    cluster_quality = np.asarray(cluster_quality)

    rows = []
    cluster_rate_mean, cluster_rate_ci95 = _confidence_interval_95(cluster_rates)
    cluster_quality_summary = {}
    for metric_index, metric_name in enumerate(CLUSTER_QUALITY_METRICS):
        mean, ci95 = _confidence_interval_95(cluster_quality[:, metric_index])
        cluster_quality_summary[f"{metric_name}_mean"] = mean
        cluster_quality_summary[f"{metric_name}_ci95"] = ci95

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        row = {
            "t": int(checkpoint),
            "rounds": int(args.rounds),
            "devices": int(args.devices),
            "clustering_mode": args.clustering_mode,
            "clusterized_devices_rate_mean": cluster_rate_mean,
            "clusterized_devices_rate_ci95": cluster_rate_ci95,
            **cluster_quality_summary,
        }
        for scenario_index, scenario_name in enumerate(SCENARIOS):
            mean, ci95 = _confidence_interval_95(
                error_norms[:, checkpoint_index, scenario_index]
            )
            row[f"{scenario_name}_error_norm_mean"] = mean
            row[f"{scenario_name}_error_norm_ci95"] = ci95

            mean, ci95 = _confidence_interval_95(
                uploads[:, checkpoint_index, scenario_index]
            )
            row[f"{scenario_name}_uploads_mean"] = mean
            row[f"{scenario_name}_uploads_ci95"] = ci95

        for d2d_index, scenario_name in enumerate(SCENARIOS[3:]):
            mean, ci95 = _confidence_interval_95(
                clusterhead_uploads[:, checkpoint_index, d2d_index]
            )
            row[f"{scenario_name}_clusterhead_uploads_mean"] = mean
            row[f"{scenario_name}_clusterhead_uploads_ci95"] = ci95

        rows.append(row)

    dense_strategy_note = (
        "pair-first dense D2D formation, local singleton join repair, "
        "local pair CH-rotation repair, and local CH-to-CH merge repair up to Cmax; "
        "geometric mode ranks CHs by one-hop degree"
    )
    grid_strategy_note = "cell_side = R_D2D / sqrt(2)"
    metadata = {
        "backend": "jax",
        "jax_version": jax.__version__,
        "devices": int(args.devices),
        "rounds": int(args.rounds),
        "iterations": int(args.iterations),
        "checkpoints": checkpoints.astype(int).tolist(),
        "seed_start": int(args.seed),
        "seed_policy": "round_seed = seed_start + round_index",
        "clustering_mode": args.clustering_mode,
        "clustering_strategy": args.clustering_strategy,
        "gradient_normalization": "by_k" if args.normalize_by_k else "thesis_unscaled",
        "precision": getattr(args, "precision", "float32"),
        "jax_enable_x64": getattr(args, "precision", "float32") == "float64",
        "repair_passes": int(args.repair_passes),
        "rotation_repair_passes": int(args.rotation_repair_passes),
        "merge_passes": int(args.merge_passes),
        "initial_cluster_size": int(args.initial_cluster_size),
        "cluster_head_selection_mode": args.cluster_head_selection_mode,
        "cluster_head_degree_weight": float(args.cluster_head_degree_weight),
        "cluster_head_channel_weight": float(args.cluster_head_channel_weight),
        "cluster_head_battery_weight": float(args.cluster_head_battery_weight),
        "cluster_quality_metrics": list(CLUSTER_QUALITY_METRICS),
        "d2d_ch_bs_success_mode": args.d2d_ch_bs_success_mode,
        "d2d_ch_bs_min_success_probability": float(
            args.d2d_ch_bs_min_success_probability
        ),
        "d2d_ch_bs_pathloss_exponent": float(args.d2d_ch_bs_pathloss_exponent),
        "d2d_ch_bs_battery_exponent": float(args.d2d_ch_bs_battery_exponent),
        "device_bs_success_mode": args.device_bs_success_mode,
        "device_bs_min_success_probability": float(
            args.device_bs_min_success_probability
        ),
        "device_bs_pathloss_exponent": float(args.device_bs_pathloss_exponent),
        "device_bs_battery_exponent": float(args.device_bs_battery_exponent),
        "optimized_access_floor_fraction": float(args.optimized_access_floor_fraction),
        "optimized_d2d_access_floor_fraction": float(
            args.optimized_d2d_access_floor_fraction
        ),
        "optimized_d2d_access_mode": args.optimized_d2d_access_mode,
        "optimized_d2d_norm_exponent": float(args.optimized_d2d_norm_exponent),
        "optimized_d2d_cluster_size_exponent": float(
            args.optimized_d2d_cluster_size_exponent
        ),
        "optimized_d2d_freshness_exponent": float(args.optimized_d2d_freshness_exponent),
        "optimized_d2d_threshold_gain": float(args.optimized_d2d_threshold_gain),
        "optimized_d2d_novelty_exponent": float(args.optimized_d2d_novelty_exponent),
        "optimized_d2d_novelty_floor": float(args.optimized_d2d_novelty_floor),
        "optimized_d2d_reference_decay": float(args.optimized_d2d_reference_decay),
        "optimized_d2d_load_target_factor": float(args.optimized_d2d_load_target_factor),
        "optimized_d2d_load_allocation_mode": args.optimized_d2d_load_allocation_mode,
        "optimized_d2d_redistribution_fraction": float(
            args.optimized_d2d_redistribution_fraction
        ),
        "optimized_d2d_redistribution_trigger_ratio": float(
            args.optimized_d2d_redistribution_trigger_ratio
        ),
        "optimized_d2d_density_trigger_threshold": float(
            args.optimized_d2d_density_trigger_threshold
        ),
        "optimized_d2d_dense_trigger_ratio": float(
            args.optimized_d2d_dense_trigger_ratio
        ),
        "optimized_d2d_throughput_ewma_decay": float(
            args.optimized_d2d_throughput_ewma_decay
        ),
        "optimized_d2d_late_norm_exponent": float(
            args.optimized_d2d_late_norm_exponent
        ),
        "optimized_d2d_late_freshness_exponent": float(
            args.optimized_d2d_late_freshness_exponent
        ),
        "optimized_d2d_adaptive_switch_fraction": float(
            args.optimized_d2d_adaptive_switch_fraction
        ),
        "optimized_d2d_adaptive_switch_gain": float(
            args.optimized_d2d_adaptive_switch_gain
        ),
        "clustering_strategy_note": (
            dense_strategy_note
            if args.clustering_strategy == "dense"
            else grid_strategy_note
        ),
        "dense_strategy": dense_strategy_note,
        "grid_strategy": grid_strategy_note,
        "tile_size": int(args.tile_size),
        "python_version": platform.python_version(),
        "jax_devices": [str(device) for device in jax.devices()],
        "elapsed_seconds_including_compile": elapsed_seconds,
    }
    return rows, metadata


def _write_outputs(rows, metadata, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


def _timestamp_run_name():
    """Return the default folder name for a successful experiment run."""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _unique_run_dir(runs_dir, run_name):
    """Return a non-existing run directory path.

    The directory is not created here.  ``main`` calls this only after the JAX
    sweep finishes successfully, then ``_write_outputs`` creates the directory
    as part of writing the CSV.  If two runs start in the same second, numeric
    suffixes keep both outputs instead of overwriting either one.
    """
    runs_dir = Path(runs_dir)
    base = runs_dir / run_name
    if not base.exists():
        return base

    suffix = 2
    while True:
        candidate = runs_dir / f"{run_name}-{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _resolve_output_csv(args):
    """Resolve where CSV results should be written after a successful run."""
    if args.output is not None:
        return Path(args.output)

    run_name = args.run_name or _timestamp_run_name()
    run_dir = _unique_run_dir(args.runs_dir, run_name)
    return run_dir / "results.csv"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, default=1000)
    parser.add_argument("--bs-radius", type=float, default=300.0)
    parser.add_argument("--device-radius", type=float, default=15.0)
    parser.add_argument("--max-cluster-size", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None)
    parser.add_argument("--data-dimension", type=int, default=10)
    parser.add_argument("--channels", type=int, default=10)
    parser.add_argument("--pcomp", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=202406)
    parser.add_argument(
        "--clustering-mode",
        choices=("no_d2d", "geometric", "utility"),
        default="geometric",
    )
    parser.add_argument(
        "--clustering-strategy",
        choices=("dense", "grid"),
        default="dense",
        help=(
            "dense is thesis-oriented and compares each CH against all devices; "
            "grid is faster and more conservative for very large K."
        ),
    )
    parser.add_argument(
        "--normalize-by-k",
        action="store_true",
        help=(
            "Divide aggregated gradients by K. Disabled by default because the "
            "thesis notebook figure used the unnormalized SGD step."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float32",
        help=(
            "float32 is fastest for GPU sweeps; float64 is slower but needed "
            "to reproduce thesis curves that fall below about 1e-7."
        ),
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument(
        "--repair-passes",
        type=int,
        default=1,
        help=(
            "Number of local singleton join-repair passes for dense clustering. "
            "Each pass allows reachable singletons to join nearby CHs with spare Cmax."
        ),
    )
    parser.add_argument(
        "--initial-cluster-size",
        type=int,
        default=2,
        help=(
            "Initial dense D2D cluster size before local repair. The default 2 "
            "matches D2D-SRC pair formation; larger values greedily fill clusters earlier."
        ),
    )
    parser.add_argument(
        "--rotation-repair-passes",
        type=int,
        default=1,
        help=(
            "Number of local CH-rotation repair passes for size-2 clusters. "
            "A pair can rotate CH to its member when that member can absorb a singleton."
        ),
    )
    parser.add_argument(
        "--merge-passes",
        type=int,
        default=1,
        help=(
            "Number of local CH-to-CH merge passes after singleton/rotation repair. "
            "A merge is accepted only when the target CH can cover the union and Cmax holds."
        ),
    )
    parser.add_argument(
        "--cluster-head-selection-mode",
        choices=("first", "quality"),
        default="first",
        help=(
            "Post-clustering CH election. first preserves the current CH in "
            "column 0; quality rotates each cluster to the best member that "
            "still covers all members using D2D degree, BS channel quality, "
            "and battery scores."
        ),
    )
    parser.add_argument(
        "--cluster-head-degree-weight",
        type=float,
        default=0.40,
        help="Quality CH rotation weight for normalized D2D degree.",
    )
    parser.add_argument(
        "--cluster-head-channel-weight",
        type=float,
        default=0.40,
        help="Quality CH rotation weight for normalized inverse pathloss to the BS.",
    )
    parser.add_argument(
        "--cluster-head-battery-weight",
        type=float,
        default=0.20,
        help="Quality CH rotation weight for normalized battery percentage.",
    )
    parser.add_argument("--uniform-area", action="store_true")
    parser.add_argument("--d2d-member-compute-probability", type=float, default=1.0)
    parser.add_argument("--d2d-member-link-success-probability", type=float, default=1.0)
    parser.add_argument(
        "--d2d-ch-bs-success-mode",
        choices=("none", "channel_quality"),
        default="none",
        help=(
            "Optional CH-to-BS decoding realism for D2D curves. none preserves "
            "the collision-only thesis-compatible behavior; channel_quality "
            "makes a collision-free CH upload succeed according to the elected "
            "CH distance-to-BS and optional battery factor."
        ),
    )
    parser.add_argument(
        "--d2d-ch-bs-min-success-probability",
        type=float,
        default=0.20,
        help=(
            "Minimum collision-free CH-to-BS decoding probability used by "
            "channel_quality mode. Ignored when success mode is none."
        ),
    )
    parser.add_argument(
        "--d2d-ch-bs-pathloss-exponent",
        type=float,
        default=2.0,
        help=(
            "Pathloss exponent used to convert CH distance-to-BS into normalized "
            "BS channel quality for channel_quality mode."
        ),
    )
    parser.add_argument(
        "--d2d-ch-bs-battery-exponent",
        type=float,
        default=0.0,
        help=(
            "Optional battery exponent for channel_quality CH-to-BS decoding. "
            "The default 0 uses channel quality only."
        ),
    )
    parser.add_argument(
        "--device-bs-success-mode",
        choices=("none", "channel_quality"),
        default="none",
        help=(
            "Optional device-to-BS decoding realism for non-D2D curves. none "
            "preserves the thesis-compatible collision-only behavior; "
            "channel_quality makes each collision-free direct upload succeed "
            "according to the transmitting device distance-to-BS and optional "
            "battery factor."
        ),
    )
    parser.add_argument(
        "--device-bs-min-success-probability",
        type=float,
        default=0.20,
        help=(
            "Minimum collision-free device-to-BS decoding probability used by "
            "channel_quality mode. Ignored when success mode is none."
        ),
    )
    parser.add_argument(
        "--device-bs-pathloss-exponent",
        type=float,
        default=2.0,
        help=(
            "Pathloss exponent used to convert device distance-to-BS into "
            "normalized BS channel quality for direct channel_quality mode."
        ),
    )
    parser.add_argument(
        "--device-bs-battery-exponent",
        type=float,
        default=0.0,
        help=(
            "Optional battery exponent for direct device-to-BS decoding. "
            "The default 0 uses channel quality only."
        ),
    )
    parser.add_argument(
        "--optimized-access-floor-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum access probability for optimized ALOHA as a fraction of "
            "the fixed-ALOHA probability. The default 0 preserves the thesis controller."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-access-floor-fraction",
        type=float,
        default=0.0,
        help=(
            "Minimum access probability for optimized ALOHA with D2D as a fraction "
            "of the fixed D2D ALOHA probability. Use 1.0 to prevent D2D channel starvation."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-access-mode",
        choices=("norm", "utility", "max_weight", "hybrid", "adaptive_diversity"),
        default="norm",
        help=(
            "norm preserves the thesis-style optimized D2D controller; utility "
            "load-controls access by aggregate norm, active cluster size, and freshness; "
            "max_weight uses a dual-threshold gate that concentrates access on high-utility CHs; "
            "hybrid keeps utility load control and adds directional novelty; "
            "adaptive_diversity shifts from early utility to late novelty/freshness."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-norm-exponent",
        type=float,
        default=1.0,
        help="Utility-mode exponent for aggregate update norm.",
    )
    parser.add_argument(
        "--optimized-d2d-cluster-size-exponent",
        type=float,
        default=1.0,
        help="Utility-mode exponent for active D2D aggregate size.",
    )
    parser.add_argument(
        "--optimized-d2d-freshness-exponent",
        type=float,
        default=0.5,
        help="Utility-mode exponent for time since last optimized-D2D CH success.",
    )
    parser.add_argument(
        "--optimized-d2d-threshold-gain",
        type=float,
        default=8.0,
        help=(
            "Max-weight mode sigmoid gain. Larger values make the CH access "
            "decision closer to a hard utility threshold."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-novelty-exponent",
        type=float,
        default=1.0,
        help=(
            "Hybrid/adaptive-diversity exponent for directional novelty against "
            "recent optimized-D2D uploads."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-novelty-floor",
        type=float,
        default=0.25,
        help=(
            "Hybrid/adaptive-diversity minimum novelty credit for aggregates "
            "aligned with the recent optimized-D2D reference direction."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-reference-decay",
        type=float,
        default=0.90,
        help=(
            "Hybrid/adaptive-diversity exponential decay for the recent successful "
            "optimized-D2D reference direction."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-load-target-factor",
        type=float,
        default=1.0,
        help=(
            "Utility/hybrid/adaptive-diversity multiplier for the target expected "
            "CH contender load. 1.0 targets M contenders, 0.8 targets 0.8*M, "
            "and 1.2 targets 1.2*M."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-load-allocation-mode",
        choices=(
            "water_filling",
            "selective_water_filling",
            "conditional_selective_water_filling",
            "proportional_clip",
        ),
        default="conditional_selective_water_filling",
        help=(
            "Load-controlled utility allocator. water_filling redistributes "
            "all probability clipped by pcomp; selective_water_filling "
            "redistributes only --optimized-d2d-redistribution-fraction of "
            "the clipped mass; conditional_selective_water_filling only "
            "redistributes when observed optimized-D2D CH throughput falls "
            "below the trigger ratio and otherwise returns proportional_clip; "
            "proportional_clip reproduces the older behavior."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-redistribution-fraction",
        type=float,
        default=0.5,
        help=(
            "Selective-water-filling fraction of clipped target load to "
            "redistribute. 0.0 matches proportional_clip; 1.0 matches the "
            "water-filling target load."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-redistribution-trigger-ratio",
        type=float,
        default=0.95,
        help=(
            "Conditional-selective trigger. Redistribution is skipped when the "
            "EWMA of successful optimized-D2D CH uploads is at least this "
            "fraction of the expected fixed-D2D CH throughput."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-density-trigger-threshold",
        type=float,
        default=0.95,
        help=(
            "Conditional-selective density gate. When the clustered-device "
            "fraction is at least this value, the allocator uses "
            "--optimized-d2d-dense-trigger-ratio instead of the base trigger. "
            "Use 1.0 to effectively disable the dense-regime override."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-dense-trigger-ratio",
        type=float,
        default=0.90,
        help=(
            "Conditional-selective trigger ratio used in dense clusterization "
            "regimes. Lower values make redistribution less likely when the "
            "network is already highly covered by D2D clusters."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-throughput-ewma-decay",
        type=float,
        default=0.90,
        help=(
            "Conditional-selective EWMA decay for observed optimized-D2D CH "
            "throughput. Larger values react more slowly; 0.90 is a stable "
            "default for 200-round thesis-style curves."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-late-norm-exponent",
        type=float,
        default=1.25,
        help=(
            "Adaptive-diversity late-phase norm exponent. Lower values reduce "
            "late over-selection of the largest aggregate directions."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-late-freshness-exponent",
        type=float,
        default=1.0,
        help=(
            "Adaptive-diversity late-phase freshness exponent. Larger values "
            "give stale CHs more late-round access probability."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-adaptive-switch-fraction",
        type=float,
        default=0.30,
        help=(
            "Adaptive-diversity midpoint as a fraction of total iterations. "
            "The phase uses t/max_t instead of true error to stay deployable."
        ),
    )
    parser.add_argument(
        "--optimized-d2d-adaptive-switch-gain",
        type=float,
        default=12.0,
        help=(
            "Adaptive-diversity sigmoid gain. Larger values make the transition "
            "from early utility to late diversity sharper."
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("Runs"),
        help="Parent directory for automatically created timestamped run folders.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name under --runs-dir. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional direct CSV path. If omitted, writes Runs/<timestamp>/results.csv.",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        default=("png", "pdf"),
        help="Figure formats generated from the output CSV.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write CSV and metadata; skip figure generation.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    rows, metadata = run_gpu_sweep(args)
    output_csv = _resolve_output_csv(args)
    metadata_path = output_csv.with_suffix(".metadata.json")

    metadata["run_directory"] = str(output_csv.parent)
    metadata["output_csv"] = str(output_csv)
    metadata["metadata_path"] = str(metadata_path)
    metadata["plots_enabled"] = not args.no_plots
    metadata["plot_formats"] = list(args.plot_formats)
    metadata_path = _write_outputs(rows, metadata, output_csv)
    print(f"wrote {output_csv}")
    print(f"wrote {metadata_path}")
    if not args.no_plots:
        from experiments.plot_gpu_sweep import plot_sweep_csv

        generated_paths = plot_sweep_csv(output_csv, formats=args.plot_formats)
        metadata["generated_figures"] = [str(path) for path in generated_paths]
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for output_path in generated_paths:
            print(f"wrote {output_path}")


if __name__ == "__main__":
    main()


def namespace_from_defaults(**overrides):
    """Small helper for tests/notebooks that want an args-like object."""
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
