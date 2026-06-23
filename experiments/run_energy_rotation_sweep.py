"""Compare static and energy-aware D2D CH-rotation profiles.

This runner exists because energy-aware CH rotation is a tradeoff policy, not a
pure error-minimization policy.  The normal ``run_gpu_sweep`` command can run a
single profile, but comparing ``static`` against ``performance``, ``balanced``,
and ``eco`` manually is tedious and easy to mis-document.

The sweep keeps every non-rotation parameter fixed, writes one subfolder per
candidate, then writes a compact summary focused on:

* optimized-D2D error over the full curve;
* successful device and CH uploads;
* normalized energy used;
* uploads per normalized battery unit;
* final D2D CH battery and CH energy used.

The ranking is intentionally conservative.  A candidate is considered
``energy_feasible`` when, relative to the static baseline, final optimized-D2D
error and total optimized-D2D energy are both within the configured tolerance.
Among feasible candidates, the runner prefers higher final CH battery, then
lower full-curve log-error AUC.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from experiments.run_gpu_sweep import (
    D2D_ENERGY_EFFICIENCY_PROFILES,
    _unique_run_dir,
    _write_outputs,
    build_parser as build_sweep_parser,
    run_gpu_sweep,
)


ERROR_TARGETS = (
    ("t_to_1e_minus_3", 1e-3),
    ("t_to_1e_minus_6", 1e-6),
    ("t_to_1e_minus_9", 1e-9),
    ("t_to_1e_minus_12", 1e-12),
)

SUMMARY_FIELDS = (
    "rank",
    "candidate_index",
    "candidate_id",
    "d2d_ch_rotation_mode",
    "d2d_ch_rotation_trigger_mode",
    "d2d_ch_rotation_aoi_threshold_fraction",
    "d2d_energy_efficiency_level",
    "rotation_channel_weight",
    "rotation_battery_weight",
    "rotation_stability_weight",
    "energy_feasible",
    "error_ratio_vs_static",
    "energy_ratio_vs_static",
    "efficiency_ratio_vs_static",
    "ch_battery_gain_vs_static",
    "ch_energy_ratio_vs_static",
    "clusterized_devices_rate",
    "target_time_score",
    "t_to_1e_minus_3",
    "t_to_1e_minus_6",
    "t_to_1e_minus_9",
    "t_to_1e_minus_12",
    "log_error_auc",
    "log_error_sum",
    "t100_error",
    "t200_error",
    "t200_error_ci95",
    "ch_upload_ratio",
    "device_upload_ratio",
    "optimized_d2d_ch_uploads_t200",
    "fixed_d2d_ch_uploads_t200",
    "optimized_d2d_uploads_t200",
    "fixed_d2d_uploads_t200",
    "optimized_d2d_energy_used_t200",
    "optimized_d2d_energy_efficiency_t200",
    "optimized_d2d_clusterhead_battery_t200",
    "optimized_d2d_clusterhead_energy_used_t200",
    "result_csv",
)


def _default_sweep_name():
    return "energy_rotation_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def energy_rotation_candidates(include_static=True, include_aoi_triggered=False):
    """Return the fixed rotation-profile comparison set.

    ``static`` is the control: no intra-run CH re-election.  The three
    energy-aware candidates use the named profile weights from
    ``D2D_ENERGY_EFFICIENCY_PROFILES``.  AoI-triggered candidates are optional
    because they are a newer CH-side freshness experiment and should not change
    historical static/performance/balanced/eco sweeps unless explicitly asked.
    """
    candidates = []
    if include_static:
        candidates.append(
            {
                "candidate_id": "static",
                "d2d_ch_rotation_mode": "static",
                "d2d_ch_rotation_trigger_mode": "interval",
                "d2d_ch_rotation_aoi_threshold_fraction": 0.75,
                "d2d_energy_efficiency_level": "balanced",
            }
        )
    for level in ("performance", "balanced", "eco"):
        candidates.append(
            {
                "candidate_id": f"energy_{level}",
                "d2d_ch_rotation_mode": "energy_aware",
                "d2d_ch_rotation_trigger_mode": "interval",
                "d2d_ch_rotation_aoi_threshold_fraction": 0.75,
                "d2d_energy_efficiency_level": level,
            }
        )
    if include_aoi_triggered:
        for trigger_mode in ("aoi", "interval_or_aoi"):
            candidates.append(
                {
                    "candidate_id": f"energy_performance_{trigger_mode}",
                    "d2d_ch_rotation_mode": "energy_aware",
                    "d2d_ch_rotation_trigger_mode": trigger_mode,
                    "d2d_ch_rotation_aoi_threshold_fraction": 0.75,
                    "d2d_energy_efficiency_level": "performance",
                }
            )
    return candidates


def select_candidate_slice(candidates, candidate_start=0, candidate_count=None):
    """Return a zero-based candidate slice for Colab-friendly partial runs."""
    if candidate_start < 0:
        raise ValueError("candidate_start must be non-negative")
    if candidate_count is not None and candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if candidate_count is None:
        return list(candidates[candidate_start:])
    return list(candidates[candidate_start : candidate_start + candidate_count])


def _float(row, key):
    return float(row[key])


def _safe_log10(value):
    return math.log10(max(float(value), 1e-300))


def _sorted_rows(rows):
    return sorted(rows, key=lambda row: int(row["t"]))


def _nearest_row(rows, target_t):
    return min(rows, key=lambda row: abs(int(row["t"]) - int(target_t)))


def _time_to_error_target(rows, target):
    """Return first recorded t where optimized-D2D reaches target error."""
    rows = _sorted_rows(rows)
    for row in rows:
        if _float(row, "optimized_aloha_d2d_error_norm_mean") <= target:
            return int(row["t"])
    return int(rows[-1]["t"]) + 1


def _average_log_error_auc(rows):
    """Return trapezoidal average of log10 optimized-D2D error over t."""
    rows = _sorted_rows(rows)
    points = [
        (
            int(row["t"]),
            _safe_log10(_float(row, "optimized_aloha_d2d_error_norm_mean")),
        )
        for row in rows
    ]
    if len(points) == 1:
        return points[0][1]

    area = 0.0
    for (left_t, left_y), (right_t, right_y) in zip(points, points[1:]):
        width = max(right_t - left_t, 0)
        area += width * (left_y + right_y) / 2.0
    span = max(points[-1][0] - points[0][0], 1)
    return area / span


def _log_error_sum(rows):
    return sum(
        _safe_log10(_float(row, "optimized_aloha_d2d_error_norm_mean"))
        for row in _sorted_rows(rows)
    )


def _profile_weights(candidate):
    if candidate["d2d_ch_rotation_mode"] == "static":
        return (0.0, 0.0, 1.0)
    return D2D_ENERGY_EFFICIENCY_PROFILES[
        candidate["d2d_energy_efficiency_level"]
    ]


def summarize_candidate(candidate, rows, result_csv, candidate_index=None):
    """Summarize one rotation-profile candidate into scalar metrics."""
    rows = _sorted_rows(rows)
    row_100 = _nearest_row(rows, 100)
    row_200 = _nearest_row(rows, 200)
    channel_weight, battery_weight, stability_weight = _profile_weights(candidate)

    optimized_ch = _float(row_200, "optimized_aloha_d2d_clusterhead_uploads_mean")
    fixed_ch = _float(row_200, "fixed_aloha_d2d_clusterhead_uploads_mean")
    optimized_uploads = _float(row_200, "optimized_aloha_d2d_uploads_mean")
    fixed_uploads = _float(row_200, "fixed_aloha_d2d_uploads_mean")
    target_times = {
        field: _time_to_error_target(rows, target)
        for field, target in ERROR_TARGETS
    }

    return {
        **candidate,
        "rank": 0,
        "candidate_index": "" if candidate_index is None else int(candidate_index),
        "d2d_ch_rotation_trigger_mode": candidate.get(
            "d2d_ch_rotation_trigger_mode",
            "interval",
        ),
        "d2d_ch_rotation_aoi_threshold_fraction": candidate.get(
            "d2d_ch_rotation_aoi_threshold_fraction",
            0.75,
        ),
        "rotation_channel_weight": channel_weight,
        "rotation_battery_weight": battery_weight,
        "rotation_stability_weight": stability_weight,
        "energy_feasible": "",
        "error_ratio_vs_static": "",
        "energy_ratio_vs_static": "",
        "efficiency_ratio_vs_static": "",
        "ch_battery_gain_vs_static": "",
        "ch_energy_ratio_vs_static": "",
        "clusterized_devices_rate": _float(row_200, "clusterized_devices_rate_mean"),
        "target_time_score": sum(target_times.values()),
        **target_times,
        "log_error_auc": _average_log_error_auc(rows),
        "log_error_sum": _log_error_sum(rows),
        "t100_error": _float(row_100, "optimized_aloha_d2d_error_norm_mean"),
        "t200_error": _float(row_200, "optimized_aloha_d2d_error_norm_mean"),
        "t200_error_ci95": _float(row_200, "optimized_aloha_d2d_error_norm_ci95"),
        "ch_upload_ratio": optimized_ch / fixed_ch if fixed_ch > 0.0 else math.inf,
        "device_upload_ratio": (
            optimized_uploads / fixed_uploads if fixed_uploads > 0.0 else math.inf
        ),
        "optimized_d2d_ch_uploads_t200": optimized_ch,
        "fixed_d2d_ch_uploads_t200": fixed_ch,
        "optimized_d2d_uploads_t200": optimized_uploads,
        "fixed_d2d_uploads_t200": fixed_uploads,
        "optimized_d2d_energy_used_t200": _float(
            row_200,
            "optimized_aloha_d2d_energy_used_mean",
        ),
        "optimized_d2d_energy_efficiency_t200": _float(
            row_200,
            "optimized_aloha_d2d_energy_efficiency_mean",
        ),
        "optimized_d2d_clusterhead_battery_t200": _float(
            row_200,
            "optimized_aloha_d2d_clusterhead_battery_mean",
        ),
        "optimized_d2d_clusterhead_energy_used_t200": _float(
            row_200,
            "optimized_aloha_d2d_clusterhead_energy_used_mean",
        ),
        "result_csv": str(result_csv),
    }


def annotate_static_ratios(summaries, tolerance=1.05):
    """Add ratios against the static baseline to every summary row."""
    if tolerance < 1.0:
        raise ValueError("tolerance must be at least 1.0")

    static_rows = [
        row for row in summaries if row["d2d_ch_rotation_mode"] == "static"
    ]
    if not static_rows:
        return summaries

    baseline = static_rows[0]
    base_error = float(baseline["t200_error"])
    base_energy = float(baseline["optimized_d2d_energy_used_t200"])
    base_efficiency = float(baseline["optimized_d2d_energy_efficiency_t200"])
    base_ch_battery = float(baseline["optimized_d2d_clusterhead_battery_t200"])
    base_ch_energy = float(baseline["optimized_d2d_clusterhead_energy_used_t200"])

    for row in summaries:
        error_ratio = (
            float(row["t200_error"]) / base_error if base_error > 0.0 else math.inf
        )
        energy_ratio = (
            float(row["optimized_d2d_energy_used_t200"]) / base_energy
            if base_energy > 0.0
            else math.inf
        )
        efficiency_ratio = (
            float(row["optimized_d2d_energy_efficiency_t200"]) / base_efficiency
            if base_efficiency > 0.0
            else math.inf
        )
        ch_battery_gain = (
            float(row["optimized_d2d_clusterhead_battery_t200"]) - base_ch_battery
        )
        ch_energy_ratio = (
            float(row["optimized_d2d_clusterhead_energy_used_t200"]) / base_ch_energy
            if base_ch_energy > 0.0
            else math.inf
        )
        row["error_ratio_vs_static"] = error_ratio
        row["energy_ratio_vs_static"] = energy_ratio
        row["efficiency_ratio_vs_static"] = efficiency_ratio
        row["ch_battery_gain_vs_static"] = ch_battery_gain
        row["ch_energy_ratio_vs_static"] = ch_energy_ratio
        row["energy_feasible"] = (
            error_ratio <= tolerance and energy_ratio <= tolerance
        )
    return summaries


def rank_candidate_summaries(summaries, tolerance=1.05):
    """Rank profiles by energy fairness under bounded error/energy regressions."""
    annotate_static_ratios(summaries, tolerance=tolerance)

    def sort_key(row):
        feasible = bool(row.get("energy_feasible", False))
        ch_battery_gain = row.get("ch_battery_gain_vs_static", 0.0)
        if ch_battery_gain in ("", None):
            ch_battery_gain = 0.0
        return (
            not feasible,
            -float(ch_battery_gain),
            float(row["log_error_auc"]),
            float(row["t200_error"]),
            row["candidate_id"],
        )

    ranked = sorted(summaries, key=sort_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _write_summary_csv(rows, output_path):
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _format_ratio(value):
    if value in ("", None):
        return ""
    return f"{float(value):.4f}"


def _write_markdown(rows, output_path):
    lines = [
        "# Energy-Aware CH Rotation Sweep",
        "",
        "Ranking prefers candidates that keep final optimized-D2D error and "
        "energy within the configured tolerance against the static baseline, "
        "then maximize final optimized-D2D CH battery.",
        "",
        "| Rank | Candidate | Feasible | t1e-3 | t200 error | Error/static | "
        "Energy/static | CH battery gain | Energy eff | CH uploads | Device uploads |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{candidate_id}` | {feasible} | {t3} | {t200:.3e} | "
            "{err_ratio} | {energy_ratio} | {battery_gain} | {eff:.3f} | "
            "{ch:.2f} | {uploads:.2f} |".format(
                rank=row["rank"],
                candidate_id=row["candidate_id"],
                feasible=row["energy_feasible"],
                t3=row["t_to_1e_minus_3"],
                t200=float(row["t200_error"]),
                err_ratio=_format_ratio(row["error_ratio_vs_static"]),
                energy_ratio=_format_ratio(row["energy_ratio_vs_static"]),
                battery_gain=_format_ratio(row["ch_battery_gain_vs_static"]),
                eff=float(row["optimized_d2d_energy_efficiency_t200"]),
                ch=float(row["optimized_d2d_ch_uploads_t200"]),
                uploads=float(row["optimized_d2d_uploads_t200"]),
            )
        )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clean_formats(formats):
    cleaned = []
    for item in formats:
        normalized = str(item).strip().lstrip(".").lower()
        if normalized:
            cleaned.append(normalized)
    return tuple(dict.fromkeys(cleaned))


def _read_result_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_figure(fig, output_stem, formats):
    generated_paths = []
    for file_format in formats:
        output_path = Path(output_stem).with_suffix(f".{file_format}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        generated_paths.append(output_path)
    return generated_paths


def _write_error_curve_plot(ranked_rows, output_stem, formats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axis = plt.subplots(figsize=(8.0, 5.2))
    for row in ranked_rows:
        result_rows = _read_result_rows(row["result_csv"])
        x_values = [int(result_row["t"]) for result_row in result_rows]
        y_values = [
            max(
                float(result_row["optimized_aloha_d2d_error_norm_mean"]),
                np.finfo(float).tiny,
            )
            for result_row in result_rows
        ]
        axis.plot(
            x_values,
            y_values,
            linewidth=2.0,
            label=row["candidate_id"],
        )

    axis.set_yscale("log")
    axis.set_xlabel("Iteration t")
    axis.set_ylabel("Optimized D2D error norm")
    axis.set_title("Optimized D2D error by CH rotation profile")
    axis.grid(True, which="both", alpha=0.45)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    generated_paths = _save_figure(fig, output_stem, formats)
    plt.close(fig)
    return generated_paths


def _write_tradeoff_plot(ranked_rows, output_stem, formats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.4, 5.2))
    for row in ranked_rows:
        x_value = _safe_log10(float(row["t200_error"]))
        y_value = float(row["optimized_d2d_clusterhead_battery_t200"])
        axis.scatter(
            [x_value],
            [y_value],
            s=90,
            label=row["candidate_id"],
        )
        axis.annotate(
            str(row["rank"]),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    axis.set_xlabel("log10 optimized-D2D error at t=200")
    axis.set_ylabel("Mean optimized-D2D CH battery at t=200")
    axis.set_title("Error vs CH battery tradeoff")
    axis.grid(True, alpha=0.35)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    generated_paths = _save_figure(fig, output_stem, formats)
    plt.close(fig)
    return generated_paths


def _write_summary_plots(ranked_rows, sweep_dir, formats):
    formats = _clean_formats(formats)
    if not formats:
        return []

    generated_paths = []
    generated_paths.extend(
        _write_error_curve_plot(
            ranked_rows,
            Path(sweep_dir) / "energy_rotation_optimized_d2d_error_norm",
            formats,
        )
    )
    generated_paths.extend(
        _write_tradeoff_plot(
            ranked_rows,
            Path(sweep_dir) / "energy_rotation_tradeoff",
            formats,
        )
    )
    return generated_paths


def _copy_args_with_candidate(args, candidate, output_csv):
    candidate_args = vars(args).copy()
    candidate_args.update(candidate)
    candidate_args.update(
        {
            "energy_drain_mode": "dynamic",
            "output": Path(output_csv),
            "run_name": candidate["candidate_id"],
        }
    )
    return SimpleNamespace(**candidate_args)


def build_parser():
    parser = build_sweep_parser()
    parser.description = __doc__
    parser.set_defaults(
        rounds=100,
        precision="float64",
        energy_drain_mode="dynamic",
        energy_direct_bs_cost=0.002,
        energy_d2d_member_cost=0.0005,
        energy_ch_bs_cost=0.005,
        cluster_head_selection_mode="quality",
        cluster_head_degree_weight=0.0,
        cluster_head_channel_weight=1.0,
        cluster_head_battery_weight=0.0,
        d2d_ch_bs_success_mode="channel_quality",
        d2d_ch_bs_min_success_probability=0.35,
        d2d_ch_bs_pathloss_exponent=2.0,
        d2d_ch_bs_battery_exponent=0.25,
        device_bs_success_mode="channel_quality",
        device_bs_min_success_probability=0.35,
        device_bs_pathloss_exponent=2.0,
        device_bs_battery_exponent=0.25,
        optimized_d2d_access_mode="utility",
        optimized_d2d_load_allocation_mode="conditional_selective_water_filling",
        optimized_d2d_redistribution_fraction=0.25,
        optimized_d2d_redistribution_trigger_ratio=0.95,
        optimized_d2d_density_trigger_threshold=0.95,
        optimized_d2d_dense_trigger_ratio=0.0,
        optimized_d2d_throughput_ewma_decay=0.90,
        optimized_d2d_access_floor_fraction=0.02,
        optimized_d2d_norm_exponent=3.5,
        optimized_d2d_cluster_size_exponent=1.5,
        optimized_d2d_freshness_exponent=0.25,
        optimized_d2d_load_target_factor=1.1,
    )
    parser.add_argument(
        "--candidate-start",
        type=int,
        default=0,
        help="Zero-based index of the first rotation candidate to run.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=None,
        help="Number of rotation candidates to run from --candidate-start.",
    )
    parser.add_argument(
        "--no-static-baseline",
        action="store_true",
        help=(
            "Run only energy-aware profiles. Ranking ratios against static are "
            "left blank when no static baseline is present."
        ),
    )
    parser.add_argument(
        "--include-aoi-triggered-rotation",
        action="store_true",
        help=(
            "Add performance-profile candidates triggered by AoI stale-tail "
            "rotation. Default off preserves the historical four-candidate "
            "static/performance/balanced/eco sweep."
        ),
    )
    parser.add_argument(
        "--energy-feasibility-tolerance",
        type=float,
        default=1.05,
        help=(
            "Maximum allowed final error and energy ratio versus static for a "
            "candidate to be marked energy_feasible."
        ),
    )
    parser.add_argument(
        "--per-config-plots",
        action="store_true",
        help="Generate the full plot set inside every candidate subfolder.",
    )
    return parser


def run_energy_rotation_sweep(args):
    """Run static/performance/balanced/eco CH-rotation candidates."""
    full_candidates = energy_rotation_candidates(
        include_static=not args.no_static_baseline,
        include_aoi_triggered=args.include_aoi_triggered_rotation,
    )
    selected_candidates = select_candidate_slice(
        full_candidates,
        candidate_start=args.candidate_start,
        candidate_count=args.candidate_count,
    )
    if not selected_candidates:
        raise ValueError("candidate selection is empty")

    if args.energy_feasibility_tolerance < 1.0:
        raise ValueError("energy_feasibility_tolerance must be at least 1.0")

    sweep_name = args.run_name or _default_sweep_name()
    sweep_dir = _unique_run_dir(args.runs_dir, sweep_name)

    summaries = []
    candidate_start = int(args.candidate_start)
    candidate_end = candidate_start + len(selected_candidates)

    for offset, candidate in enumerate(selected_candidates):
        candidate_index = candidate_start + offset
        candidate_dir = sweep_dir / f"{candidate_index + 1:03d}_{candidate['candidate_id']}"
        result_csv = candidate_dir / "results.csv"
        candidate_args = _copy_args_with_candidate(args, candidate, result_csv)
        rows, metadata = run_gpu_sweep(candidate_args)
        metadata["energy_rotation_candidate"] = candidate
        metadata["energy_rotation_candidate_index"] = candidate_index
        metadata["energy_rotation_parent"] = str(sweep_dir)
        metadata["run_directory"] = str(candidate_dir)
        metadata["output_csv"] = str(result_csv)
        metadata["plots_enabled"] = bool(args.per_config_plots)
        metadata["plot_formats"] = list(args.plot_formats)
        metadata_path = _write_outputs(rows, metadata, result_csv)

        if args.per_config_plots:
            from experiments.plot_gpu_sweep import plot_sweep_csv

            generated_paths = plot_sweep_csv(result_csv, formats=args.plot_formats)
            metadata["generated_figures"] = [str(path) for path in generated_paths]
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        summaries.append(
            summarize_candidate(
                candidate,
                rows,
                result_csv,
                candidate_index=candidate_index,
            )
        )
        print(
            f"[{offset + 1}/{len(selected_candidates)} | "
            f"candidate {candidate_index + 1}/{len(full_candidates)}] "
            f"wrote {result_csv}"
        )

    ranked = rank_candidate_summaries(
        summaries,
        tolerance=args.energy_feasibility_tolerance,
    )
    summary_csv = sweep_dir / "energy_rotation_summary.csv"
    summary_md = sweep_dir / "energy_rotation_summary.md"
    _write_summary_csv(ranked, summary_csv)
    _write_markdown(ranked, summary_md)
    summary_plot_paths = []
    if not args.no_plots:
        summary_plot_paths = _write_summary_plots(
            ranked,
            sweep_dir,
            args.plot_formats,
        )

    metadata = {
        "candidate_count": len(selected_candidates),
        "candidate_start": candidate_start,
        "candidate_end_exclusive": candidate_end,
        "full_candidate_count": len(full_candidates),
        "candidate_grid": full_candidates,
        "energy_feasibility_tolerance": float(args.energy_feasibility_tolerance),
        "ranking": {
            "primary": (
                "among candidates within tolerance versus static for final "
                "optimized-D2D error and energy, maximize final optimized-D2D "
                "CH battery"
            ),
            "tie_breakers": [
                "lower full-curve optimized-D2D log-error AUC",
                "lower t=200 optimized-D2D error",
                "candidate id",
            ],
        },
        "sweep_defaults": {
            "precision": args.precision,
            "rounds": args.rounds,
            "energy_drain_mode": "dynamic",
            "energy_direct_bs_cost": args.energy_direct_bs_cost,
            "energy_d2d_member_cost": args.energy_d2d_member_cost,
            "energy_ch_bs_cost": args.energy_ch_bs_cost,
            "d2d_ch_rotation_interval": args.d2d_ch_rotation_interval,
            "d2d_ch_rotation_trigger_mode": args.d2d_ch_rotation_trigger_mode,
            "d2d_ch_rotation_aoi_threshold_fraction": (
                args.d2d_ch_rotation_aoi_threshold_fraction
            ),
            "d2d_energy_efficiency_profiles": D2D_ENERGY_EFFICIENCY_PROFILES,
            "optimized_d2d_access_mode": args.optimized_d2d_access_mode,
            "optimized_d2d_load_allocation_mode": args.optimized_d2d_load_allocation_mode,
            "optimized_d2d_access_floor_fraction": args.optimized_d2d_access_floor_fraction,
            "optimized_d2d_norm_exponent": args.optimized_d2d_norm_exponent,
            "optimized_d2d_cluster_size_exponent": args.optimized_d2d_cluster_size_exponent,
            "optimized_d2d_freshness_exponent": args.optimized_d2d_freshness_exponent,
            "optimized_d2d_load_target_factor": args.optimized_d2d_load_target_factor,
        },
        "summary_csv": str(summary_csv),
        "summary_markdown": str(summary_md),
        "summary_plots": [str(path) for path in summary_plot_paths],
    }
    metadata_path = sweep_dir / "energy_rotation_sweep.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")
    for plot_path in summary_plot_paths:
        print(f"wrote {plot_path}")
    print(f"wrote {metadata_path}")
    return ranked, metadata


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_energy_rotation_sweep(args)


if __name__ == "__main__":
    main()
