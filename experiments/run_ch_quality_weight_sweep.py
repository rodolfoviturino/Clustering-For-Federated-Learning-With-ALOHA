"""Run a focused sweep over quality-CH election weights.

This runner is intentionally small.  The algorithmic pieces already exist in
``run_gpu_sweep``:

* dense one-hop D2D clustering;
* quality CH election;
* channel-aware CH-to-BS decoding;
* channel-aware direct device-to-BS decoding for fair non-D2D curves;
* utility optimized-D2D access;
* density-aware conditional selective water-filling.

The goal here is not to introduce a new controller.  It executes a fixed set of
CH election weight candidates and writes one summary so the final channel-heavy
choice can be made from paired runs instead of manually comparing long command
outputs.
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
    ("t_to_1e_minus_6", 1e-6),
    ("t_to_1e_minus_9", 1e-9),
    ("t_to_1e_minus_12", 1e-12),
)

CANDIDATE_PROFILES = {
    "finalists": (
        {
            "candidate_id": "w001000_channel_only",
            "cluster_head_degree_weight": 0.0,
            "cluster_head_channel_weight": 1.0,
            "cluster_head_battery_weight": 0.0,
        },
        {
            "candidate_id": "w009010_channel_battery",
            "cluster_head_degree_weight": 0.0,
            "cluster_head_channel_weight": 0.9,
            "cluster_head_battery_weight": 0.1,
        },
    ),
    "channel_heavy": (
        {
            "candidate_id": "w108010_reference",
            "cluster_head_degree_weight": 0.1,
            "cluster_head_channel_weight": 0.8,
            "cluster_head_battery_weight": 0.1,
        },
        {
            "candidate_id": "w009010_channel_battery",
            "cluster_head_degree_weight": 0.0,
            "cluster_head_channel_weight": 0.9,
            "cluster_head_battery_weight": 0.1,
        },
        {
            "candidate_id": "w001000_channel_only",
            "cluster_head_degree_weight": 0.0,
            "cluster_head_channel_weight": 1.0,
            "cluster_head_battery_weight": 0.0,
        },
        {
            "candidate_id": "w109000_degree_channel",
            "cluster_head_degree_weight": 0.1,
            "cluster_head_channel_weight": 0.9,
            "cluster_head_battery_weight": 0.0,
        },
    ),
}

SUMMARY_FIELDS = (
    "rank",
    "candidate_index",
    "candidate_id",
    "cluster_head_degree_weight",
    "cluster_head_channel_weight",
    "cluster_head_battery_weight",
    "clusterized_devices_rate",
    "target_time_score",
    "t_to_1e_minus_6",
    "t_to_1e_minus_9",
    "t_to_1e_minus_12",
    "log_error_auc",
    "log_error_sum",
    "t100_error",
    "t200_error",
    "ch_upload_ratio",
    "device_upload_ratio",
    "optimized_d2d_ch_uploads_t200",
    "fixed_d2d_ch_uploads_t200",
    "optimized_d2d_uploads_t200",
    "fixed_d2d_uploads_t200",
    "direct_optimized_t200_error",
    "direct_optimized_uploads_t200",
    "direct_optimized_log_error_auc",
    "optimized_d2d_log_auc_gain_vs_direct",
    "optimized_d2d_vs_direct_t200_error_ratio",
    "optimized_d2d_log10_gain_vs_direct_t200",
    "optimized_d2d_upload_ratio_vs_direct_t200",
    "result_csv",
)


def _default_sweep_name():
    return "ch_quality_weights_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def select_candidate_slice(candidates, candidate_start=0, candidate_count=None):
    """Return a zero-based candidate slice for Colab-friendly partial runs."""
    if candidate_start < 0:
        raise ValueError("candidate_start must be non-negative")
    if candidate_count is not None and candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if candidate_count is None:
        return list(candidates[candidate_start:])
    return list(candidates[candidate_start : candidate_start + candidate_count])


def quality_weight_candidates(profile="finalists"):
    """Return named CH-election weight candidates."""
    if profile not in CANDIDATE_PROFILES:
        raise ValueError(f"unknown CH quality candidate profile: {profile}")
    return [dict(candidate) for candidate in CANDIDATE_PROFILES[profile]]


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


def _scenario_log_error_points(rows, scenario):
    rows = _sorted_rows(rows)
    return [
        (
            int(row["t"]),
            _safe_log10(_float(row, f"{scenario}_error_norm_mean")),
        )
        for row in rows
    ]


def _log_error_points(rows):
    return _scenario_log_error_points(rows, "optimized_aloha_d2d")


def _average_scenario_log_error_auc(rows, scenario):
    """Return trapezoidal average of log10 error for one scenario over t."""
    points = _scenario_log_error_points(rows, scenario)
    if len(points) == 1:
        return points[0][1]

    area = 0.0
    for (left_t, left_y), (right_t, right_y) in zip(points, points[1:]):
        width = max(right_t - left_t, 0)
        area += width * (left_y + right_y) / 2.0
    span = max(points[-1][0] - points[0][0], 1)
    return area / span


def _average_log_error_auc(rows):
    """Return trapezoidal average of log10 optimized-D2D error over t."""
    return _average_scenario_log_error_auc(rows, "optimized_aloha_d2d")


def _log_error_sum(rows):
    return sum(value for _, value in _log_error_points(rows))


def summarize_candidate(candidate, rows, result_csv, candidate_index=None):
    """Summarize one CH-weight candidate into rankable scalar metrics."""
    rows = _sorted_rows(rows)
    row_100 = _nearest_row(rows, 100)
    row_200 = _nearest_row(rows, 200)

    optimized_ch = _float(row_200, "optimized_aloha_d2d_clusterhead_uploads_mean")
    fixed_ch = _float(row_200, "fixed_aloha_d2d_clusterhead_uploads_mean")
    optimized_uploads = _float(row_200, "optimized_aloha_d2d_uploads_mean")
    fixed_uploads = _float(row_200, "fixed_aloha_d2d_uploads_mean")
    direct_optimized_error = _float(row_200, "optimized_aloha_error_norm_mean")
    direct_optimized_uploads = _float(row_200, "optimized_aloha_uploads_mean")
    optimized_d2d_error = _float(row_200, "optimized_aloha_d2d_error_norm_mean")
    optimized_d2d_auc = _average_log_error_auc(rows)
    direct_optimized_auc = _average_scenario_log_error_auc(rows, "optimized_aloha")

    target_times = {
        field: _time_to_error_target(rows, target)
        for field, target in ERROR_TARGETS
    }

    return {
        **candidate,
        "rank": 0,
        "candidate_index": "" if candidate_index is None else int(candidate_index),
        "clusterized_devices_rate": _float(row_200, "clusterized_devices_rate_mean"),
        "target_time_score": sum(target_times.values()),
        **target_times,
        "log_error_auc": optimized_d2d_auc,
        "log_error_sum": _log_error_sum(rows),
        "t100_error": _float(row_100, "optimized_aloha_d2d_error_norm_mean"),
        "t200_error": optimized_d2d_error,
        "ch_upload_ratio": optimized_ch / fixed_ch if fixed_ch > 0.0 else math.inf,
        "device_upload_ratio": (
            optimized_uploads / fixed_uploads if fixed_uploads > 0.0 else math.inf
        ),
        "optimized_d2d_ch_uploads_t200": optimized_ch,
        "fixed_d2d_ch_uploads_t200": fixed_ch,
        "optimized_d2d_uploads_t200": optimized_uploads,
        "fixed_d2d_uploads_t200": fixed_uploads,
        "direct_optimized_t200_error": direct_optimized_error,
        "direct_optimized_uploads_t200": direct_optimized_uploads,
        "direct_optimized_log_error_auc": direct_optimized_auc,
        "optimized_d2d_log_auc_gain_vs_direct": (
            direct_optimized_auc - optimized_d2d_auc
        ),
        "optimized_d2d_vs_direct_t200_error_ratio": (
            optimized_d2d_error / direct_optimized_error
            if direct_optimized_error > 0.0
            else math.inf
        ),
        "optimized_d2d_log10_gain_vs_direct_t200": (
            _safe_log10(direct_optimized_error) - _safe_log10(optimized_d2d_error)
        ),
        "optimized_d2d_upload_ratio_vs_direct_t200": (
            optimized_uploads / direct_optimized_uploads
            if direct_optimized_uploads > 0.0
            else math.inf
        ),
        "result_csv": str(result_csv),
    }


def rank_candidate_summaries(summaries):
    """Rank candidates by full-curve error, then target times and final error."""
    ranked = sorted(
        summaries,
        key=lambda row: (
            float(row["log_error_auc"]),
            int(row["t_to_1e_minus_9"]),
            int(row["t_to_1e_minus_6"]),
            float(row["t200_error"]),
            abs(float(row["ch_upload_ratio"]) - 1.0),
            row["candidate_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _write_summary_csv(rows, output_path):
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_top_markdown(rows, output_path):
    lines = [
        "# CH Quality Weight Sweep",
        "",
        "Ranking uses lower average log10 optimized-D2D error over the full "
        "recorded curve as the primary criterion.  Ties prefer faster target "
        "times, lower t=200 error, and CH upload ratio closer to 1.0.",
        "",
        "| Rank | Candidate | Weights degree/channel/battery | t1e-6 | t1e-9 | "
        "t1e-12 | log AUC | logsum | D2D t200 | Direct t200 | D2D log10 gain | "
        "CH ratio | D2D/direct uploads |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        weights = (
            f"{row['cluster_head_degree_weight']:.2f}/"
            f"{row['cluster_head_channel_weight']:.2f}/"
            f"{row['cluster_head_battery_weight']:.2f}"
        )
        lines.append(
            "| {rank} | `{candidate_id}` | {weights} | {t6} | {t9} | {t12} | "
            "{auc:.3f} | {logsum:.3f} | {t200:.3e} | {direct:.3e} | "
            "{gain:.3f} | {ch:.4f} | {direct_upload:.4f} |".format(
                rank=row["rank"],
                candidate_id=row["candidate_id"],
                weights=weights,
                t6=row["t_to_1e_minus_6"],
                t9=row["t_to_1e_minus_9"],
                t12=row["t_to_1e_minus_12"],
                auc=float(row["log_error_auc"]),
                logsum=float(row["log_error_sum"]),
                t200=float(row["t200_error"]),
                direct=float(row["direct_optimized_t200_error"]),
                gain=float(row["optimized_d2d_log10_gain_vs_direct_t200"]),
                ch=float(row["ch_upload_ratio"]),
                direct_upload=float(row["optimized_d2d_upload_ratio_vs_direct_t200"]),
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


def _write_candidate_error_norm_plot(ranked_rows, output_stem, formats):
    """Plot optimized-D2D error curves for every CH-weight candidate."""
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
        weights = (
            f"{float(row['cluster_head_degree_weight']):.1f}/"
            f"{float(row['cluster_head_channel_weight']):.1f}/"
            f"{float(row['cluster_head_battery_weight']):.1f}"
        )
        axis.plot(
            x_values,
            y_values,
            linewidth=2.0,
            label=f"{row['candidate_id']} ({weights})",
        )

    axis.set_yscale("log")
    axis.set_xlabel("Iteration t")
    axis.set_ylabel("Optimized D2D error norm")
    axis.set_title("Optimized D2D error by CH election weights")
    axis.grid(True, which="both", alpha=0.45)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    generated_paths = _save_figure(fig, output_stem, formats)
    plt.close(fig)
    return generated_paths


def _write_best_fair_error_norm_plot(best_row, output_stem, formats):
    """Plot the six fair-comparison error curves for the best candidate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from experiments.plot_gpu_sweep import THESIS_FIGURE_15_STYLES

    result_rows = _read_result_rows(best_row["result_csv"])
    x_values = [int(row["t"]) for row in result_rows]
    marker_stride = 1 if len(x_values) <= 20 else max(1, len(x_values) // 10)

    fig, axis = plt.subplots(figsize=(7.6, 5.4))
    for scenario, label, color, linestyle, marker in THESIS_FIGURE_15_STYLES:
        mean_column = f"{scenario}_error_norm_mean"
        if mean_column not in result_rows[0]:
            continue
        y_values = [
            max(float(row[mean_column]), np.finfo(float).tiny)
            for row in result_rows
        ]
        axis.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_stride,
            linewidth=2.0,
            markersize=6.0,
            label=label,
        )

    axis.set_yscale("log")
    axis.set_xlabel("Iteration t")
    axis.set_ylabel("Error Norm")
    axis.set_title(f"Fair error norm comparison: {best_row['candidate_id']}")
    axis.grid(True, which="both", alpha=0.55)
    axis.legend(loc="lower left", fontsize=8)
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
        _write_candidate_error_norm_plot(
            ranked_rows,
            Path(sweep_dir) / "ch_quality_weight_optimized_d2d_error_norm",
            formats,
        )
    )
    generated_paths.extend(
        _write_best_fair_error_norm_plot(
            ranked_rows[0],
            Path(sweep_dir) / "ch_quality_weight_best_fair_error_norm",
            formats,
        )
    )
    return generated_paths


def _copy_args_with_candidate(args, candidate, output_csv):
    candidate_args = vars(args).copy()
    candidate_args.update(candidate)
    candidate_args.update(
        {
            "cluster_head_selection_mode": "quality",
            "output": Path(output_csv),
            "run_name": candidate["candidate_id"],
        }
    )
    return SimpleNamespace(**candidate_args)


def _candidate_from_metadata(result_csv):
    metadata_path = Path(result_csv).with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise ValueError(
            f"{metadata_path} is missing; cannot recover CH-quality candidate"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    candidate = metadata.get("ch_quality_weight_candidate")
    if not candidate:
        raise ValueError(
            f"{metadata_path} does not contain ch_quality_weight_candidate"
        )
    candidate_index = metadata.get("ch_quality_weight_candidate_index")
    return candidate, candidate_index


def summarize_existing_sweep(args):
    """Regenerate summaries and plots from an existing CH-quality sweep folder."""
    sweep_dir = Path(args.summarize_existing_run_dir)
    result_paths = sorted(sweep_dir.glob("*/results.csv"))
    if not result_paths:
        raise ValueError(f"{sweep_dir} does not contain candidate results.csv files")

    summaries = []
    for result_csv in result_paths:
        candidate, candidate_index = _candidate_from_metadata(result_csv)
        rows = _read_result_rows(result_csv)
        summaries.append(
            summarize_candidate(
                candidate,
                rows,
                result_csv,
                candidate_index=candidate_index,
            )
        )

    ranked = rank_candidate_summaries(summaries)
    summary_csv = sweep_dir / "ch_quality_weight_summary.csv"
    top_md = sweep_dir / "ch_quality_weight_top.md"
    _write_summary_csv(ranked, summary_csv)
    _write_top_markdown(ranked[: args.top_k], top_md)
    summary_plot_paths = []
    if not args.no_plots:
        summary_plot_paths = _write_summary_plots(
            ranked,
            sweep_dir,
            args.plot_formats,
        )

    metadata = {
        "source": "existing_ch_quality_weight_sweep",
        "candidate_count": len(ranked),
        "summary_csv": str(summary_csv),
        "top_markdown": str(top_md),
        "summary_plots": [str(path) for path in summary_plot_paths],
    }
    metadata_path = sweep_dir / "ch_quality_weight_sweep.postprocess.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {summary_csv}")
    print(f"wrote {top_md}")
    for plot_path in summary_plot_paths:
        print(f"wrote {plot_path}")
    print(f"wrote {metadata_path}")
    return ranked, metadata


def build_parser():
    parser = build_sweep_parser()
    parser.description = __doc__
    parser.set_defaults(
        precision="float64",
        cluster_head_selection_mode="quality",
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
        "--candidate-profile",
        choices=tuple(CANDIDATE_PROFILES),
        default="finalists",
        help=(
            "finalists compares channel-only against channel+battery; "
            "channel_heavy also includes the previous 0.1/0.8/0.1 reference "
            "and the 0.1/0.9/0.0 ablation."
        ),
    )
    parser.add_argument(
        "--candidate-start",
        type=int,
        default=0,
        help="Zero-based start index for partial Colab runs.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=None,
        help="Number of candidates to run from candidate-start.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked candidates to include in the Markdown summary.",
    )
    parser.add_argument(
        "--per-config-plots",
        action="store_true",
        help="Generate the full plot set for each candidate subfolder.",
    )
    parser.add_argument(
        "--summarize-existing-run-dir",
        type=Path,
        default=None,
        help=(
            "Do not run simulations. Rebuild the summary CSV, Markdown, and "
            "error-norm plots from an existing CH-quality sweep directory."
        ),
    )
    return parser


def run_ch_quality_weight_sweep(args):
    full_candidates = quality_weight_candidates(args.candidate_profile)
    selected_candidates = select_candidate_slice(
        full_candidates,
        candidate_start=args.candidate_start,
        candidate_count=args.candidate_count,
    )
    if not selected_candidates:
        raise ValueError("candidate selection is empty")

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
        metadata["ch_quality_weight_candidate"] = candidate
        metadata["ch_quality_weight_candidate_index"] = candidate_index
        metadata["ch_quality_weight_parent"] = str(sweep_dir)
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

    ranked = rank_candidate_summaries(summaries)
    summary_csv = sweep_dir / "ch_quality_weight_summary.csv"
    top_md = sweep_dir / "ch_quality_weight_top.md"
    _write_summary_csv(ranked, summary_csv)
    _write_top_markdown(ranked[: args.top_k], top_md)
    summary_plot_paths = []
    if not args.no_plots:
        summary_plot_paths = _write_summary_plots(
            ranked,
            sweep_dir,
            args.plot_formats,
        )

    metadata = {
        "candidate_profile": args.candidate_profile,
        "candidate_count": len(selected_candidates),
        "candidate_start": candidate_start,
        "candidate_end_exclusive": candidate_end,
        "full_candidate_count": len(full_candidates),
        "candidate_grid": full_candidates,
        "ranking": {
            "primary": "lowest average log10 optimized-D2D error over recorded t",
            "tie_breakers": [
                "lower time to 1e-9",
                "lower time to 1e-6",
                "lower t=200 optimized-D2D error",
                "CH upload ratio closer to 1.0",
            ],
        },
        "sweep_defaults": {
            "precision": args.precision,
            "d2d_ch_bs_success_mode": args.d2d_ch_bs_success_mode,
            "d2d_ch_bs_min_success_probability": args.d2d_ch_bs_min_success_probability,
            "d2d_ch_bs_pathloss_exponent": args.d2d_ch_bs_pathloss_exponent,
            "d2d_ch_bs_battery_exponent": args.d2d_ch_bs_battery_exponent,
            "device_bs_success_mode": args.device_bs_success_mode,
            "device_bs_min_success_probability": args.device_bs_min_success_probability,
            "device_bs_pathloss_exponent": args.device_bs_pathloss_exponent,
            "device_bs_battery_exponent": args.device_bs_battery_exponent,
            "energy_drain_mode": args.energy_drain_mode,
            "energy_direct_bs_cost": args.energy_direct_bs_cost,
            "energy_d2d_member_cost": args.energy_d2d_member_cost,
            "energy_ch_bs_cost": args.energy_ch_bs_cost,
            "d2d_ch_rotation_mode": args.d2d_ch_rotation_mode,
            "d2d_ch_rotation_interval": args.d2d_ch_rotation_interval,
            "d2d_energy_efficiency_level": args.d2d_energy_efficiency_level,
            "d2d_energy_efficiency_profile_weights": {
                "channel": D2D_ENERGY_EFFICIENCY_PROFILES[
                    args.d2d_energy_efficiency_level
                ][0],
                "battery": D2D_ENERGY_EFFICIENCY_PROFILES[
                    args.d2d_energy_efficiency_level
                ][1],
                "stability": D2D_ENERGY_EFFICIENCY_PROFILES[
                    args.d2d_energy_efficiency_level
                ][2],
            },
            "optimized_d2d_access_mode": args.optimized_d2d_access_mode,
            "optimized_d2d_load_allocation_mode": args.optimized_d2d_load_allocation_mode,
            "optimized_d2d_access_floor_fraction": args.optimized_d2d_access_floor_fraction,
            "optimized_d2d_norm_exponent": args.optimized_d2d_norm_exponent,
            "optimized_d2d_cluster_size_exponent": args.optimized_d2d_cluster_size_exponent,
            "optimized_d2d_freshness_exponent": args.optimized_d2d_freshness_exponent,
            "optimized_d2d_load_target_factor": args.optimized_d2d_load_target_factor,
        },
        "summary_csv": str(summary_csv),
        "top_markdown": str(top_md),
        "summary_plots": [str(path) for path in summary_plot_paths],
    }
    metadata_path = sweep_dir / "ch_quality_weight_sweep.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {summary_csv}")
    print(f"wrote {top_md}")
    for plot_path in summary_plot_paths:
        print(f"wrote {plot_path}")
    print(f"wrote {metadata_path}")
    return ranked, metadata


def main():
    args = build_parser().parse_args()
    if args.summarize_existing_run_dir is not None:
        summarize_existing_sweep(args)
        return
    run_ch_quality_weight_sweep(args)


if __name__ == "__main__":
    main()
