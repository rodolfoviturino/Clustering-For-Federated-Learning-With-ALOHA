"""Tune optimized-D2D utility parameters and summarize Pareto candidates.

This runner is intentionally separate from ``run_gpu_sweep``.  The normal
runner executes one experiment configuration.  This module executes a fixed
utility-mode grid, writes each candidate to its own subfolder, and creates a
small summary that ranks candidates by curve quality while keeping BS-side CH
contention close to the fixed-D2D baseline.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from experiments.run_gpu_sweep import (
    _unique_run_dir,
    _write_outputs,
    build_parser as build_sweep_parser,
    run_gpu_sweep,
)


FLOOR_VALUES = (0.05, 0.10, 0.15)
NORM_EXPONENT_VALUES = (1.5, 2.0, 2.5)
CLUSTER_SIZE_EXPONENT_VALUES = (0.75, 1.0, 1.25)
FRESHNESS_EXPONENT_VALUES = (0.5, 1.0, 1.5)
LOAD_TARGET_FACTOR_VALUES = (0.8, 1.0, 1.2)

SUMMARY_FIELDS = (
    "rank",
    "pareto_feasible",
    "candidate_id",
    "optimized_d2d_access_floor_fraction",
    "optimized_d2d_norm_exponent",
    "optimized_d2d_cluster_size_exponent",
    "optimized_d2d_freshness_exponent",
    "optimized_d2d_load_target_factor",
    "log_error_auc",
    "t100_error",
    "t200_error",
    "ch_upload_ratio",
    "device_upload_gain",
    "optimized_d2d_ch_uploads_t200",
    "fixed_d2d_ch_uploads_t200",
    "optimized_d2d_uploads_t200",
    "fixed_d2d_uploads_t200",
    "result_csv",
)


def select_candidate_slice(candidates, candidate_start=0, candidate_count=None):
    """Return the requested zero-based slice of the candidate grid."""
    if candidate_start < 0:
        raise ValueError("candidate_start must be non-negative")
    if candidate_count is not None and candidate_count < 1:
        raise ValueError("candidate_count must be positive")

    if candidate_count is None:
        return candidates[candidate_start:]
    return candidates[candidate_start : candidate_start + candidate_count]


def _candidate_tag(value):
    return f"{float(value):.2f}".replace(".", "p")


def candidate_grid():
    """Return the fixed Pareto tuning grid for optimized-D2D utility mode."""
    candidates = []
    for floor, norm_exp, size_exp, freshness_exp, load_factor in itertools.product(
        FLOOR_VALUES,
        NORM_EXPONENT_VALUES,
        CLUSTER_SIZE_EXPONENT_VALUES,
        FRESHNESS_EXPONENT_VALUES,
        LOAD_TARGET_FACTOR_VALUES,
    ):
        candidate_id = (
            f"floor{_candidate_tag(floor)}"
            f"_norm{_candidate_tag(norm_exp)}"
            f"_size{_candidate_tag(size_exp)}"
            f"_fresh{_candidate_tag(freshness_exp)}"
            f"_load{_candidate_tag(load_factor)}"
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "optimized_d2d_access_floor_fraction": floor,
                "optimized_d2d_norm_exponent": norm_exp,
                "optimized_d2d_cluster_size_exponent": size_exp,
                "optimized_d2d_freshness_exponent": freshness_exp,
                "optimized_d2d_load_target_factor": load_factor,
            }
        )
    return candidates


def _float(row, key):
    return float(row[key])


def _sorted_rows(rows):
    return sorted(rows, key=lambda row: int(row["t"]))


def _nearest_row(rows, target_t):
    return min(rows, key=lambda row: abs(int(row["t"]) - int(target_t)))


def _average_log_error_auc(rows):
    """Return trapezoidal AUC of log10 optimized-D2D error over checkpoints."""
    rows = _sorted_rows(rows)
    points = [
        (int(row["t"]), math.log10(max(_float(row, "optimized_aloha_d2d_error_norm_mean"), 1e-300)))
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


def summarize_candidate(candidate, rows, result_csv):
    """Summarize one candidate's rows into rankable scalar metrics."""
    rows = _sorted_rows(rows)
    row_100 = _nearest_row(rows, 100)
    row_200 = _nearest_row(rows, 200)

    optimized_ch = _float(row_200, "optimized_aloha_d2d_clusterhead_uploads_mean")
    fixed_ch = _float(row_200, "fixed_aloha_d2d_clusterhead_uploads_mean")
    optimized_uploads = _float(row_200, "optimized_aloha_d2d_uploads_mean")
    fixed_uploads = _float(row_200, "fixed_aloha_d2d_uploads_mean")

    ch_ratio = optimized_ch / fixed_ch if fixed_ch > 0.0 else math.inf
    device_upload_gain = (
        (optimized_uploads - fixed_uploads) / fixed_uploads if fixed_uploads > 0.0 else math.inf
    )
    feasible = 0.95 <= ch_ratio <= 1.05

    return {
        **candidate,
        "rank": 0,
        "pareto_feasible": feasible,
        "log_error_auc": _average_log_error_auc(rows),
        "t100_error": _float(row_100, "optimized_aloha_d2d_error_norm_mean"),
        "t200_error": _float(row_200, "optimized_aloha_d2d_error_norm_mean"),
        "ch_upload_ratio": ch_ratio,
        "device_upload_gain": device_upload_gain,
        "optimized_d2d_ch_uploads_t200": optimized_ch,
        "fixed_d2d_ch_uploads_t200": fixed_ch,
        "optimized_d2d_uploads_t200": optimized_uploads,
        "fixed_d2d_uploads_t200": fixed_uploads,
        "result_csv": str(result_csv),
    }


def rank_candidate_summaries(summaries):
    """Rank feasible Pareto candidates before infeasible candidates."""
    ranked = sorted(
        summaries,
        key=lambda row: (
            not row["pareto_feasible"],
            row["log_error_auc"],
            row["t100_error"],
            row["t200_error"],
            -row["device_upload_gain"],
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


def _write_top10_markdown(rows, output_path):
    lines = [
        "# Utility Pareto Top 10",
        "",
        "| Rank | Candidate | Feasible | Log AUC | t100 Error | t200 Error | CH Ratio | Device Upload Gain |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{candidate_id}` | {feasible} | {auc:.6g} | {t100:.6g} | "
            "{t200:.6g} | {ratio:.4f} | {gain:.4f} |".format(
                rank=row["rank"],
                candidate_id=row["candidate_id"],
                feasible="yes" if row["pareto_feasible"] else "no",
                auc=row["log_error_auc"],
                t100=row["t100_error"],
                t200=row["t200_error"],
                ratio=row["ch_upload_ratio"],
                gain=row["device_upload_gain"],
            )
        )
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pareto_plot(rows, output_stem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feasible = [row for row in rows if row["pareto_feasible"]]
    infeasible = [row for row in rows if not row["pareto_feasible"]]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.axvspan(0.95, 1.05, color="#d9ead3", alpha=0.5, label="CH ratio target")
    if infeasible:
        ax.scatter(
            [row["ch_upload_ratio"] for row in infeasible],
            [row["log_error_auc"] for row in infeasible],
            s=24,
            color="#b7b7b7",
            label="infeasible",
        )
    if feasible:
        ax.scatter(
            [row["ch_upload_ratio"] for row in feasible],
            [row["log_error_auc"] for row in feasible],
            s=28,
            color="#1f77b4",
            label="feasible",
        )

    for row in rows[:3]:
        ax.annotate(
            str(row["rank"]),
            (row["ch_upload_ratio"], row["log_error_auc"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xlabel("Optimized D2D CH uploads / fixed D2D CH uploads at t=200")
    ax.set_ylabel("Average log10 optimized-D2D error over the curve")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()

    output_stem = Path(output_stem)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def _copy_args_with_candidate(args, candidate, output_csv):
    candidate_args = vars(args).copy()
    candidate_args.update(candidate)
    candidate_args.update(
        {
            "optimized_d2d_access_mode": "utility",
            "optimized_access_floor_fraction": args.optimized_access_floor_fraction,
            "output": Path(output_csv),
            "run_name": candidate["candidate_id"],
        }
    )
    return SimpleNamespace(**candidate_args)


def _default_sweep_name():
    return "utility_pareto_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def build_parser():
    parser = build_sweep_parser()
    parser.description = __doc__
    parser.set_defaults(
        optimized_d2d_access_mode="utility",
        optimized_d2d_access_floor_fraction=0.10,
        optimized_d2d_norm_exponent=2.0,
        optimized_d2d_cluster_size_exponent=1.0,
        optimized_d2d_freshness_exponent=1.0,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help=(
            "Run only the first N grid candidates for smoke testing. "
            "Equivalent to --candidate-start 0 --candidate-count N."
        ),
    )
    parser.add_argument(
        "--candidate-start",
        type=int,
        default=0,
        help="Zero-based index of the first candidate to run from the fixed grid.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=None,
        help="Number of candidates to run from --candidate-start.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidates to include in the Markdown summary.",
    )
    parser.add_argument(
        "--per-config-plots",
        action="store_true",
        help="Generate standard result figures inside every candidate subfolder.",
    )
    return parser


def run_utility_pareto_sweep(args):
    """Run the fixed utility candidate grid and write summary artifacts."""
    if args.top_k < 1:
        raise ValueError("top_k must be positive")

    if args.max_candidates is not None and args.candidate_count is not None:
        raise ValueError("use either max_candidates or candidate_count, not both")

    candidate_start = int(args.candidate_start)
    candidate_count = args.candidate_count
    if args.max_candidates is not None:
        if args.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        candidate_start = 0
        candidate_count = args.max_candidates

    full_grid = candidate_grid()
    candidates = select_candidate_slice(full_grid, candidate_start, candidate_count)
    if not candidates:
        raise ValueError("candidate slice is empty")

    run_name = args.run_name or _default_sweep_name()
    sweep_dir = _unique_run_dir(args.runs_dir, run_name)
    summaries = []

    for offset, candidate in enumerate(candidates):
        grid_index = candidate_start + offset
        candidate_dir = sweep_dir / f"{grid_index + 1:03d}_{candidate['candidate_id']}"
        result_csv = candidate_dir / "results.csv"
        candidate_args = _copy_args_with_candidate(args, candidate, result_csv)
        rows, metadata = run_gpu_sweep(candidate_args)

        metadata["utility_pareto_candidate"] = candidate
        metadata["utility_pareto_candidate_index"] = grid_index
        metadata["utility_pareto_parent"] = str(sweep_dir)
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

        summaries.append(summarize_candidate(candidate, rows, result_csv))
        print(f"[{offset + 1}/{len(candidates)}] wrote {result_csv}")

    ranked = rank_candidate_summaries(summaries)
    summary_csv = sweep_dir / "utility_sweep_summary.csv"
    top10_md = sweep_dir / "utility_sweep_top10.md"
    pareto_stem = sweep_dir / "utility_sweep_pareto"

    _write_summary_csv(ranked, summary_csv)
    _write_top10_markdown(ranked[: args.top_k], top10_md)
    plot_paths = _write_pareto_plot(ranked, pareto_stem)

    metadata = {
        "candidate_count": len(candidates),
        "candidate_start": candidate_start,
        "candidate_end_exclusive": candidate_start + len(candidates),
        "full_grid_candidate_count": len(full_grid),
        "candidate_grid": {
            "optimized_d2d_access_floor_fraction": list(FLOOR_VALUES),
            "optimized_d2d_norm_exponent": list(NORM_EXPONENT_VALUES),
            "optimized_d2d_cluster_size_exponent": list(CLUSTER_SIZE_EXPONENT_VALUES),
            "optimized_d2d_freshness_exponent": list(FRESHNESS_EXPONENT_VALUES),
            "optimized_d2d_load_target_factor": list(LOAD_TARGET_FACTOR_VALUES),
        },
        "ranking": {
            "primary": "lowest average log10 optimized-D2D error AUC",
            "constraint": "0.95 <= final optimized/fixed D2D CH upload ratio <= 1.05",
            "tie_breakers": [
                "lower t=100 optimized-D2D error",
                "lower t=200 optimized-D2D error",
                "higher final device-upload gain",
            ],
        },
        "summary_csv": str(summary_csv),
        "top10_markdown": str(top10_md),
        "pareto_plots": [str(path) for path in plot_paths],
    }
    metadata_path = sweep_dir / "utility_sweep.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {summary_csv}")
    print(f"wrote {top10_md}")
    for plot_path in plot_paths:
        print(f"wrote {plot_path}")
    print(f"wrote {metadata_path}")
    return ranked, metadata


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_utility_pareto_sweep(args)


if __name__ == "__main__":
    main()
