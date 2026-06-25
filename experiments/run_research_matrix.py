"""Run or compare canonical research experiment matrices.

This module is intentionally a thin orchestration layer.  It does not define
new models or metrics; it gathers existing run configurations into named
matrices and reuses the normal GPU sweep/comparison code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from experiments.compare_runs import DEFAULT_SCENARIO, summarize_run, write_comparison


@dataclass(frozen=True)
class MatrixRun:
    """One canonical run entry inside a research matrix."""

    run_name: str
    claim_role: str
    claim_summary: str
    overrides: dict = field(default_factory=dict)


K3000_PHYSICAL_BASE = {
    "devices": 3000,
    "rounds": 100,
    "iterations": 100,
    "seed": 202406,
    "channels": 10,
    "bs_radius": 300.0,
    "device_radius": 15.0,
    "max_cluster_size": 10,
    "min_cluster_size": 4,
    "data_dimension": 10,
    "pcomp": 0.1,
    "learning_rate": 0.01,
    "step_size": 0.1,
    "normalize_by_k": False,
    "uniform_area": False,
    "precision": "float64",
    "energy_drain_mode": "dynamic",
    "energy_model": "first_order_radio",
    "battery_feasibility_mode": "required_energy",
    "d2d_member_link_success_mode": "rayleigh_outage",
    "d2d_ch_bs_success_mode": "rayleigh_outage",
    "device_bs_success_mode": "rayleigh_outage",
    "cluster_head_selection_mode": "quality",
    "cluster_head_channel_score_mode": "rayleigh_outage",
    "cluster_head_degree_weight": 0.0,
    "cluster_head_channel_weight": 1.0,
    "cluster_head_battery_weight": 0.0,
    "optimized_d2d_access_mode": "member_quota_utility",
    "optimized_d2d_load_allocation_mode": "conditional_selective_water_filling",
    "optimized_d2d_access_floor_fraction": 0.02,
    "optimized_d2d_norm_exponent": 3.5,
    "optimized_d2d_cluster_size_exponent": 1.5,
    "optimized_d2d_freshness_exponent": 0.25,
    "optimized_d2d_load_target_factor": 1.1,
    "optimized_d2d_aoi_weight": 0.15,
    "optimized_d2d_aoi_exponent": 1.0,
    "optimized_d2d_aoi_threshold_fraction": 0.70,
    "optimized_d2d_member_refresh_floor_fraction": 0.0,
}


MATRICES = {
    "k3000_core": [
        MatrixRun(
            run_name="member_quota_k3000_w015_floor000_physical_r100",
            claim_role="main pure-ALOHA baseline",
            claim_summary=(
                "Primary member-freshness baseline for physical/Rayleigh K=3000."
            ),
            overrides={},
        ),
        MatrixRun(
            run_name="member_collision_quota_k3000_w015_t002_g2_min050_physical_r100",
            claim_role="pure-ALOHA convergence/energy ablation",
            claim_summary=(
                "Collision-aware quota improves convergence/energy but is not "
                "the main member-freshness winner."
            ),
            overrides={
                "optimized_d2d_access_mode": "member_collision_aware_quota",
                "optimized_d2d_member_collision_target_fraction": 0.02,
                "optimized_d2d_member_collision_gain": 2.0,
                "optimized_d2d_member_collision_min_quota_scale": 0.50,
            },
        ),
        MatrixRun(
            run_name="member_capped_quota_k3000_w015_cap010_physical_r100",
            claim_role="pure-ALOHA convergence/energy ablation",
            claim_summary=(
                "Low capped-quota point improves convergence/energy but worsens "
                "zero participation."
            ),
            overrides={
                "optimized_d2d_access_mode": "member_capped_quota_utility",
                "optimized_d2d_member_quota_cap_fraction": 0.10,
            },
        ),
        MatrixRun(
            run_name="member_safe_split_k3000_s8_min2_b005_w015_physical_r100",
            claim_role="negative structural ablation",
            claim_summary=(
                "Size-only safe split avoids singleton explosion but does not "
                "improve convergence or member freshness."
            ),
            overrides={
                "cluster_split_mode": "safe_max_size",
                "cluster_split_max_size": 8,
                "cluster_split_min_subcluster_size": 2,
                "cluster_split_budget_fraction": 0.05,
            },
        ),
        MatrixRun(
            run_name=(
                "member_pressure_split_k3000_s8_min2_b005_mw100_ch050_w015_"
                "physical_r100"
            ),
            claim_role="structural convergence ablation",
            claim_summary=(
                "Pressure-guided safe split recovers convergence versus "
                "size-only split but still does not improve member freshness."
            ),
            overrides={
                "cluster_split_mode": "pressure_safe_max_size",
                "cluster_split_max_size": 8,
                "cluster_split_min_subcluster_size": 2,
                "cluster_split_budget_fraction": 0.05,
                "cluster_split_pressure_member_weight": 1.0,
                "cluster_split_pressure_ch_weight": 0.5,
            },
        ),
        MatrixRun(
            run_name="member_split_k3000_s8_w015_physical_r100",
            claim_role="negative structural ablation",
            claim_summary=(
                "Global max-size split with max size 8 worsens useful uploads, "
                "energy efficiency, and member freshness."
            ),
            overrides={
                "cluster_split_mode": "max_size",
                "cluster_split_max_size": 8,
            },
        ),
        MatrixRun(
            run_name="member_split_k3000_s5_w015_physical_r100",
            claim_role="negative structural ablation",
            claim_summary=(
                "Aggressive global max-size split creates many contenders and "
                "is a clear negative ablation."
            ),
            overrides={
                "cluster_split_mode": "max_size",
                "cluster_split_max_size": 5,
            },
        ),
        MatrixRun(
            run_name="member_semischedule_k3000_s030_cc0010_physical_r100",
            claim_role="coordinated upper-bound / future work",
            claim_summary=(
                "Semi-scheduled refresh shows the value of coordination but is "
                "not part of the pure-ALOHA claim."
            ),
            overrides={
                "optimized_d2d_access_mode": "semi_scheduled_member_refresh",
                "optimized_d2d_member_schedule_fraction": 0.30,
                "optimized_d2d_member_schedule_deficit_weight": 0.0,
                "optimized_d2d_member_schedule_control_cost": 0.001,
                "optimized_d2d_aoi_weight": 0.5,
                "optimized_d2d_member_refresh_floor_fraction": 0.05,
            },
        ),
    ],
}


def get_matrix(name):
    """Return a matrix by name, validating duplicate run names."""
    try:
        matrix = MATRICES[name]
    except KeyError as exc:
        available = ", ".join(sorted(MATRICES))
        raise ValueError(f"unknown matrix {name!r}; available matrices: {available}") from exc
    _validate_unique_run_names(matrix)
    return matrix


def _validate_unique_run_names(matrix):
    seen = set()
    duplicates = []
    for entry in matrix:
        if entry.run_name in seen:
            duplicates.append(entry.run_name)
        seen.add(entry.run_name)
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"matrix contains duplicate run names: {duplicate_text}")


def _run_dir(runs_dir, entry):
    return Path(runs_dir) / entry.run_name


def _results_csv(runs_dir, entry):
    return _run_dir(runs_dir, entry) / "results.csv"


def _missing_entries(matrix, runs_dir):
    return [entry for entry in matrix if not _results_csv(runs_dir, entry).exists()]


def _args_for_entry(entry, runs_dir):
    from experiments.run_gpu_sweep import namespace_from_defaults

    overrides = dict(K3000_PHYSICAL_BASE)
    overrides.update(entry.overrides)
    output = _results_csv(runs_dir, entry)
    overrides.update(
        {
            "run_name": entry.run_name,
            "runs_dir": Path(runs_dir),
            "output": output,
            "no_plots": True,
        }
    )
    return namespace_from_defaults(**overrides)


def _execute_entry(entry, runs_dir, matrix_name):
    from experiments.run_gpu_sweep import _write_outputs, run_gpu_sweep

    args = _args_for_entry(entry, runs_dir)
    rows, metadata = run_gpu_sweep(args)
    output_csv = _results_csv(runs_dir, entry)
    metadata_path = output_csv.with_suffix(".metadata.json")
    metadata["run_directory"] = str(output_csv.parent)
    metadata["output_csv"] = str(output_csv)
    metadata["metadata_path"] = str(metadata_path)
    metadata["plots_enabled"] = False
    metadata["plot_formats"] = list(args.plot_formats)
    metadata["research_matrix"] = matrix_name
    metadata["research_matrix_claim_role"] = entry.claim_role
    return _write_outputs(rows, metadata, output_csv)


def execute_missing(matrix, runs_dir, matrix_name="k3000_core"):
    """Run missing matrix entries and return generated metadata paths."""
    generated = []
    for entry in _missing_entries(matrix, runs_dir):
        generated.append(_execute_entry(entry, runs_dir, matrix_name))
    return generated


def compare_matrix(matrix, runs_dir, output_dir, scenario=DEFAULT_SCENARIO):
    """Write comparison and paper-claim summaries for a matrix."""
    summaries = []
    by_name = {entry.run_name: entry for entry in matrix}
    for entry in matrix:
        csv_path = _results_csv(runs_dir, entry)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} is missing; rerun with --execute-missing or create it first"
            )
        summary = summarize_run(csv_path, scenario=scenario)
        summary["claim_role"] = entry.claim_role
        summary["claim_summary"] = entry.claim_summary
        summaries.append(summary)

    csv_path, markdown_path = write_comparison(summaries, output_dir)
    paper_path = write_paper_claim_summary(summaries, by_name, output_dir)
    return csv_path, markdown_path, paper_path


def write_paper_claim_summary(summaries, matrix_by_name, output_dir):
    """Write a compact Markdown summary oriented toward paper claims."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_path = output_dir / "paper_claim_summary.md"
    baseline = summaries[0] if summaries else {}

    with paper_path.open("w", encoding="utf-8") as handle:
        handle.write("# Paper Claim Summary\n\n")
        handle.write(
            "This summary classifies canonical K=3000 physical/Rayleigh runs "
            "for paper writing. Lower member AoI/stale/zero is better; "
            "`semi_scheduled_member_refresh` is an upper-bound reference, not "
            "a pure-ALOHA contribution. Metric cells use `mean +/- ci95` when "
            "the comparison CSV contains CI95 columns.\n\n"
        )
        handle.write(
            "| Run | Claim role | t<=1e-12 | final error | member AoI | "
            "member stale75 | member zero | energy efficiency | Interpretation |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in summaries:
            entry = matrix_by_name[row["run"]]
            handle.write(
                "| {run} | {role} | {t12} | {error} | {member_aoi} | "
                "{stale75} | {zero} | {eff} | {summary} |\n".format(
                    run=row["run"],
                    role=entry.claim_role,
                    t12=_format_value(row.get("t_to_1e-12")),
                    error=_format_value_with_ci(row, "final_error_norm"),
                    member_aoi=_format_value_with_ci(row, "final_member_aoi"),
                    stale75=_format_value_with_ci(
                        row,
                        "final_member_stale_fraction_75",
                    ),
                    zero=_format_value_with_ci(
                        row,
                        "final_member_zero_participation_fraction",
                    ),
                    eff=_format_value_with_ci(row, "final_energy_efficiency"),
                    summary=entry.claim_summary,
                )
            )

        if baseline:
            handle.write("\n## Main Baseline\n\n")
            handle.write(
                f"`{baseline['run']}` remains the pure-ALOHA baseline for "
                "member-level D2D freshness. Other pure-ALOHA points should be "
                "described as ablations unless they improve member freshness "
                "without damaging convergence/energy.\n"
            )
    return paper_path


def _format_value(value):
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_value_with_ci(row, metric):
    value = _format_value(row.get(metric))
    ci95 = _format_value(row.get(f"{metric}_ci95"))
    if value and ci95:
        return f"{value} +/- {ci95}"
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        choices=sorted(MATRICES),
        default="k3000_core",
        help="Canonical research matrix to compare or execute.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing run CSVs. This is the default.",
    )
    mode.add_argument(
        "--execute-missing",
        action="store_true",
        help="Run missing matrix entries before writing comparison artifacts.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("Runs"),
        help="Parent directory containing one folder per run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Comparison output directory. Defaults to Runs/comparison_<matrix>.",
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help=f"Scenario prefix to compare. Default: {DEFAULT_SCENARIO}.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    matrix = get_matrix(args.matrix)
    output_dir = args.output_dir or (args.runs_dir / f"comparison_{args.matrix}")

    if args.execute_missing:
        generated = execute_missing(matrix, args.runs_dir, args.matrix)
        for metadata_path in generated:
            print(f"wrote {metadata_path}")

    csv_path, markdown_path, paper_path = compare_matrix(
        matrix=matrix,
        runs_dir=args.runs_dir,
        output_dir=output_dir,
        scenario=args.scenario,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")
    print(f"wrote {paper_path}")


if __name__ == "__main__":
    main()
