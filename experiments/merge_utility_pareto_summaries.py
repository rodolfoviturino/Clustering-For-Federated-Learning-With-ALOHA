"""Merge utility Pareto sweep parts into one ranked summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.run_utility_pareto_sweep import (
    SUMMARY_FIELDS,
    _write_pareto_plot,
    _write_summary_csv,
    _write_top10_markdown,
    rank_candidate_summaries,
)


NUMERIC_FIELDS = {
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
}


def _parse_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_summary_csv(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            parsed["rank"] = 0
            parsed.setdefault("candidate_grid_index", "")
            parsed["pareto_feasible"] = _parse_bool(parsed["pareto_feasible"])
            for field in NUMERIC_FIELDS:
                parsed[field] = float(parsed[field])
            rows.append(parsed)
    return rows


def _summary_path(input_path):
    path = Path(input_path)
    if path.is_dir():
        return path / "utility_sweep_summary.csv"
    return path


def merge_summary_files(inputs):
    rows = []
    seen = set()
    for input_path in inputs:
        summary_path = _summary_path(input_path)
        for row in _read_summary_csv(summary_path):
            key = row["candidate_id"]
            if key in seen:
                raise ValueError(f"duplicate candidate_id in summaries: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("no summary rows found")
    return rank_candidate_summaries(rows)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Sweep part directories or utility_sweep_summary.csv files to merge.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Runs") / "utility_pareto_merged",
        help="Directory where merged summary artifacts are written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidates to include in the Markdown summary.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        raise ValueError("top_k must be positive")

    ranked = merge_summary_files(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.output_dir / "utility_sweep_summary.csv"
    top10_md = args.output_dir / "utility_sweep_top10.md"
    pareto_stem = args.output_dir / "utility_sweep_pareto"
    metadata_path = args.output_dir / "utility_sweep.metadata.json"

    _write_summary_csv(ranked, summary_csv)
    _write_top10_markdown(ranked[: args.top_k], top10_md)
    plot_paths = _write_pareto_plot(ranked, pareto_stem)

    metadata = {
        "input_count": len(args.inputs),
        "candidate_count": len(ranked),
        "inputs": [str(input_path) for input_path in args.inputs],
        "summary_csv": str(summary_csv),
        "top10_markdown": str(top10_md),
        "pareto_plots": [str(path) for path in plot_paths],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {summary_csv}")
    print(f"wrote {top10_md}")
    for plot_path in plot_paths:
        print(f"wrote {plot_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
