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
        )
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
            checkpoints=checkpoints,
            dtype=compute_dtype,
        )
        return (
            trace.error_norms,
            trace.successful_uploads,
            trace.successful_clusterhead_uploads,
            trace.clusterized_devices_rate,
        )

    # JIT compiles one batched program specialized to the experiment dimensions.
    batched_runner = jax.jit(jax.vmap(run_one_round))
    started = time.perf_counter()
    error_norms, uploads, clusterhead_uploads, cluster_rates = batched_runner(seeds)
    jax.block_until_ready(error_norms)
    elapsed_seconds = time.perf_counter() - started

    error_norms = np.asarray(error_norms)
    uploads = np.asarray(uploads)
    clusterhead_uploads = np.asarray(clusterhead_uploads)
    cluster_rates = np.asarray(cluster_rates)

    rows = []
    cluster_rate_mean, cluster_rate_ci95 = _confidence_interval_95(cluster_rates)
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        row = {
            "t": int(checkpoint),
            "rounds": int(args.rounds),
            "devices": int(args.devices),
            "clustering_mode": args.clustering_mode,
            "clusterized_devices_rate_mean": cluster_rate_mean,
            "clusterized_devices_rate_ci95": cluster_rate_ci95,
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
        "candidate CH absorbs closest unassigned devices within R_D2D up to Cmax; "
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
    parser.add_argument("--uniform-area", action="store_true")
    parser.add_argument("--d2d-member-compute-probability", type=float, default=1.0)
    parser.add_argument("--d2d-member-link-success-probability", type=float, default=1.0)
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
