"""Compare completed experiment runs without rerunning JAX.

The simulation runner writes one ``results.csv`` per run folder.  This module
loads two or more such CSV files and produces a compact comparison report for a
chosen scenario, with metrics that have repeatedly mattered during the research
iteration:

* threshold times for error norm, e.g. first ``t`` reaching ``1e-12``;
* log-error AUC over the saved checkpoints, which is less fragile than only
  reading the final point;
* final uploads, CH uploads, energy, energy efficiency, and AoI summaries;
* stale-tail and member-level freshness fractions when the source CSV contains
  the newer AoI columns.

It intentionally does not import JAX.  The comparison is post-processing only,
so it is cheap to run locally after CPU or Colab experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path


DEFAULT_SCENARIO = "optimized_aloha_d2d"
ERROR_THRESHOLDS = (1e-6, 1e-9, 1e-12)
FINAL_METRICS = (
    "error_norm",
    "uploads",
    "clusterhead_uploads",
    "energy_used",
    "energy_efficiency",
    "aoi",
    "peak_aoi",
    "p75_aoi",
    "p90_aoi",
    "p95_aoi",
    "stale_fraction_50",
    "stale_fraction_75",
    "stale_fraction_100",
    "member_aoi",
    "member_peak_aoi",
    "member_p75_aoi",
    "member_p90_aoi",
    "member_p95_aoi",
    "member_stale_fraction_50",
    "member_stale_fraction_75",
    "member_stale_fraction_100",
    "member_participation_p05",
    "member_zero_participation_fraction",
    "member_stale_compute_failure_fraction",
    "member_stale_link_failure_fraction",
    "member_stale_member_energy_failure_fraction",
    "member_stale_ch_no_attempt_fraction",
    "member_stale_collision_fraction",
    "member_stale_ch_bs_failure_fraction",
    "member_stale_other_failure_fraction",
)


def _resolve_results_csv(path):
    """Return a concrete CSV path from either a run directory or a CSV path."""
    path = Path(path)
    if path.is_dir():
        candidate = path / "results.csv"
        if candidate.exists():
            return candidate
        csv_files = sorted(path.glob("*.csv"))
        if len(csv_files) == 1:
            return csv_files[0]
        raise FileNotFoundError(
            f"{path} is a directory but does not contain a unique results CSV"
        )
    if path.exists():
        return path
    raise FileNotFoundError(f"{path} does not exist")


def _load_rows(csv_path):
    """Load numeric CSV rows and sort them by iteration t."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    for row in rows:
        if "t" not in row:
            raise ValueError(f"{csv_path} does not contain a 't' column")
    return sorted(rows, key=lambda row: int(float(row["t"])))


def _float_or_none(value):
    """Parse a CSV value as float while tolerating absent optional columns."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _column(rows, name):
    """Return an optional numeric column from loaded rows."""
    values = [_float_or_none(row.get(name)) for row in rows]
    if any(value is None for value in values):
        return None
    return values


def _trapezoid_auc(x_values, y_values):
    """Compute trapezoidal AUC for saved checkpoints."""
    if len(x_values) == 1:
        return y_values[0]
    total = 0.0
    for index in range(1, len(x_values)):
        width = x_values[index] - x_values[index - 1]
        total += 0.5 * width * (y_values[index] + y_values[index - 1])
    return total


def _first_t_at_or_below(rows, scenario, threshold):
    """Return first checkpoint where error norm reaches a target threshold."""
    column_name = f"{scenario}_error_norm_mean"
    for row in rows:
        value = _float_or_none(row.get(column_name))
        if value is not None and value <= threshold:
            return int(float(row["t"]))
    return None


def summarize_run(csv_path, scenario=DEFAULT_SCENARIO):
    """Build one comparison summary row for a completed run CSV."""
    csv_path = _resolve_results_csv(csv_path)
    rows = _load_rows(csv_path)
    t_values = [int(float(row["t"])) for row in rows]
    final_row = rows[-1]
    summary = {
        "run": csv_path.parent.name,
        "csv_path": str(csv_path),
        "scenario": scenario,
        "first_t": t_values[0],
        "final_t": t_values[-1],
        "checkpoint_count": len(rows),
    }

    error_values = _column(rows, f"{scenario}_error_norm_mean")
    if error_values is not None:
        log_error_values = [math.log10(max(value, 1e-300)) for value in error_values]
        summary["log_error_auc"] = _trapezoid_auc(t_values, log_error_values)
        summary["log_error_sum"] = sum(log_error_values)
    for threshold in ERROR_THRESHOLDS:
        summary[f"t_to_{threshold:.0e}"] = _first_t_at_or_below(
            rows,
            scenario,
            threshold,
        )

    for metric_name in FINAL_METRICS:
        column_name = f"{scenario}_{metric_name}_mean"
        value = _float_or_none(final_row.get(column_name))
        if value is not None:
            summary[f"final_{metric_name}"] = value
        values = _column(rows, column_name)
        if values is not None:
            summary[f"{metric_name}_auc"] = _trapezoid_auc(t_values, values)

    metadata_path = csv_path.with_suffix(".metadata.json")
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        for key in (
            "devices",
            "rounds",
            "precision",
            "cluster_split_mode",
            "cluster_split_max_size",
            "optimized_d2d_access_mode",
            "optimized_d2d_load_allocation_mode",
            "optimized_d2d_aoi_weight",
            "optimized_d2d_aoi_exponent",
            "optimized_d2d_aoi_threshold_fraction",
            "optimized_d2d_aoi_channel_exponent",
            "optimized_d2d_aoi_battery_exponent",
            "optimized_d2d_member_refresh_floor_fraction",
            "optimized_d2d_member_quota_cap_fraction",
            "optimized_d2d_member_deficit_decay",
            "optimized_d2d_member_deficit_weight",
            "optimized_d2d_member_collision_target_fraction",
            "optimized_d2d_member_collision_gain",
            "optimized_d2d_member_collision_min_quota_scale",
            "optimized_d2d_member_schedule_fraction",
            "optimized_d2d_member_schedule_deficit_weight",
            "optimized_d2d_member_schedule_control_cost",
            "d2d_member_link_success_mode",
            "d2d_member_link_success_probability",
            "d2d_member_pathloss_exponent",
            "d2d_member_reference_snr",
            "d2d_member_snr_threshold",
            "energy_model",
            "battery_feasibility_mode",
            "d2d_ch_rotation_mode",
            "d2d_ch_rotation_interval",
            "d2d_ch_rotation_trigger_mode",
            "d2d_ch_rotation_aoi_threshold_fraction",
            "d2d_ch_rotation_member_threshold_fraction",
            "d2d_ch_rotation_member_link_weight",
            "d2d_ch_bs_success_mode",
            "device_bs_success_mode",
        ):
            if key in metadata:
                summary[key] = metadata[key]
    return summary


def _fieldnames(rows):
    """Return stable CSV field order while preserving optional metric columns."""
    priority = [
        "run",
        "scenario",
        "devices",
        "rounds",
        "precision",
        "cluster_split_mode",
        "cluster_split_max_size",
        "optimized_d2d_access_mode",
        "optimized_d2d_load_allocation_mode",
        "final_t",
        "checkpoint_count",
        "log_error_auc",
        "log_error_sum",
        "t_to_1e-06",
        "t_to_1e-09",
        "t_to_1e-12",
        "final_error_norm",
        "final_uploads",
        "final_clusterhead_uploads",
        "final_energy_used",
        "final_energy_efficiency",
        "final_aoi",
        "final_p75_aoi",
        "final_p90_aoi",
        "final_p95_aoi",
        "final_peak_aoi",
        "final_stale_fraction_50",
        "final_stale_fraction_75",
        "final_stale_fraction_100",
        "final_member_aoi",
        "final_member_p75_aoi",
        "final_member_p90_aoi",
        "final_member_p95_aoi",
        "final_member_stale_fraction_75",
        "final_member_zero_participation_fraction",
        "final_member_participation_p05",
        "csv_path",
    ]
    seen = set()
    ordered = []
    for field in priority:
        if any(field in row for row in rows):
            ordered.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                ordered.append(field)
                seen.add(field)
    return ordered


def write_comparison(summaries, output_dir):
    """Write CSV and Markdown comparison artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(summaries)
    csv_path = output_dir / "run_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    markdown_path = output_dir / "run_comparison_summary.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Run Comparison Summary\n\n")
        handle.write(
            "Lower `log_error_auc` is better because it integrates log10(error) "
            "over the saved checkpoints. Lower AoI/stale metrics are better; "
            "higher uploads and energy efficiency are usually better.\n\n"
        )
        handle.write(
            "| Run | Mode | log-error AUC | t<=1e-12 | final error | "
            "final AoI | member AoI | member stale75 | member zero | "
            "final energy efficiency |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            handle.write(
                (
                    "| {run} | {mode} | {auc} | {t12} | {error} | {aoi} | "
                    "{member_aoi} | {member_stale75} | {member_zero} | {eff} |\n"
                ).format(
                    run=row.get("run", ""),
                    mode=row.get("optimized_d2d_access_mode", ""),
                    auc=_format_markdown_number(row.get("log_error_auc")),
                    t12=row.get("t_to_1e-12", ""),
                    error=_format_markdown_number(row.get("final_error_norm")),
                    aoi=_format_markdown_number(row.get("final_aoi")),
                    member_aoi=_format_markdown_number(
                        row.get("final_member_aoi")
                    ),
                    member_stale75=_format_markdown_number(
                        row.get("final_member_stale_fraction_75")
                    ),
                    member_zero=_format_markdown_number(
                        row.get("final_member_zero_participation_fraction")
                    ),
                    eff=_format_markdown_number(row.get("final_energy_efficiency")),
                )
            )
    return csv_path, markdown_path


def _format_markdown_number(value):
    """Format optional numbers compactly for the Markdown report."""
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories or results.csv paths to compare.",
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help=f"Scenario prefix to compare. Default: {DEFAULT_SCENARIO}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for comparison artifacts. Defaults to "
            "Runs/comparison_YYYY-MM-DD-HH-MM-SS."
        ),
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.output_dir = Path("Runs") / f"comparison_{timestamp}"
    summaries = [summarize_run(path, scenario=args.scenario) for path in args.runs]
    csv_path, markdown_path = write_comparison(summaries, args.output_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
