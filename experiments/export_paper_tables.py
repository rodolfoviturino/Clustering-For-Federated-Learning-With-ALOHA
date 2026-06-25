"""Export paper-oriented tables from a completed comparison CSV.

The canonical research matrix already produces ``run_comparison_summary.csv``.
This module is a cheap post-processing step for writing: it converts that CSV
into a compact Markdown table, a LaTeX table, and claim bullets with deltas
against the baseline. It does not import JAX and does not rerun simulations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_COMPARISON_CSV = (
    Path("Runs") / "comparison_k3000_core" / "run_comparison_summary.csv"
)
DEFAULT_CAPTION = (
    "Research-matrix physical/Rayleigh comparison for optimized D2D ALOHA."
)
DEFAULT_LABEL = "tab:k3000-core"

LOWER_BETTER_MEMBER_METRICS = (
    "final_member_aoi",
    "final_member_stale_fraction_75",
    "final_member_zero_participation_fraction",
)


def load_comparison_rows(comparison_csv):
    """Load comparison rows from ``run_comparison_summary.csv``."""
    comparison_csv = Path(comparison_csv)
    with comparison_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{comparison_csv} is empty")
    if "run" not in rows[0]:
        raise ValueError(f"{comparison_csv} does not contain a 'run' column")
    return rows


def write_paper_exports(
    rows,
    output_dir,
    *,
    baseline_run=None,
    caption=DEFAULT_CAPTION,
    table_label=DEFAULT_LABEL,
):
    """Write Markdown, LaTeX, and claim-bullet paper artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = select_baseline(rows, baseline_run=baseline_run)
    export_rows = build_export_rows(rows, baseline)

    markdown_path = output_dir / "paper_results_table.md"
    latex_path = output_dir / "paper_results_table.tex"
    claims_path = output_dir / "paper_claim_bullets.md"

    write_markdown_table(export_rows, baseline, markdown_path)
    write_latex_table(export_rows, latex_path, caption=caption, table_label=table_label)
    write_claim_bullets(rows, baseline, claims_path)
    return markdown_path, latex_path, claims_path


def select_baseline(rows, *, baseline_run=None):
    """Select the baseline row, preferring an explicit run name."""
    if baseline_run is not None:
        for row in rows:
            if row.get("run") == baseline_run:
                return row
        raise ValueError(f"baseline run {baseline_run!r} was not found")

    for row in rows:
        if "baseline" in row.get("claim_role", "").lower():
            return row
    return rows[0]


def build_export_rows(rows, baseline):
    """Build formatted rows and baseline deltas for paper tables."""
    baseline_member_aoi = _float_or_none(baseline.get("final_member_aoi"))
    baseline_stale75 = _float_or_none(
        baseline.get("final_member_stale_fraction_75")
    )
    baseline_zero = _float_or_none(
        baseline.get("final_member_zero_participation_fraction")
    )
    baseline_energy = _float_or_none(baseline.get("final_energy_efficiency"))

    export_rows = []
    for row in rows:
        member_aoi = _float_or_none(row.get("final_member_aoi"))
        member_aoi_ci95 = _float_or_none(row.get("final_member_aoi_ci95"))
        stale75 = _float_or_none(row.get("final_member_stale_fraction_75"))
        stale75_ci95 = _float_or_none(
            row.get("final_member_stale_fraction_75_ci95")
        )
        zero = _float_or_none(row.get("final_member_zero_participation_fraction"))
        zero_ci95 = _float_or_none(
            row.get("final_member_zero_participation_fraction_ci95")
        )
        energy = _float_or_none(row.get("final_energy_efficiency"))
        energy_ci95 = _float_or_none(row.get("final_energy_efficiency_ci95"))
        final_error = _float_or_none(row.get("final_error_norm"))
        final_error_ci95 = _float_or_none(row.get("final_error_norm_ci95"))
        export_rows.append(
            {
                "label": short_label(row),
                "run": row.get("run", ""),
                "role": short_role(row),
                "t_to_1e-12": _format_t(row.get("t_to_1e-12")),
                "final_error": _format_scientific_with_ci(
                    final_error,
                    final_error_ci95,
                ),
                "member_aoi": _format_fixed_with_ci(member_aoi, member_aoi_ci95, 3),
                "member_stale75": _format_fixed_with_ci(stale75, stale75_ci95, 3),
                "member_zero": _format_fixed_with_ci(zero, zero_ci95, 3),
                "energy_efficiency": _format_fixed_with_ci(
                    energy,
                    energy_ci95,
                    1,
                ),
                "member_aoi_delta": _format_delta_percent(
                    _relative_delta(member_aoi, baseline_member_aoi)
                ),
                "member_stale75_delta": _format_delta_percent(
                    _relative_delta(stale75, baseline_stale75)
                ),
                "member_zero_delta": _format_delta_percent(
                    _relative_delta(zero, baseline_zero)
                ),
                "energy_efficiency_delta": _format_delta_percent(
                    _relative_delta(energy, baseline_energy)
                ),
            }
        )
    return export_rows


def write_markdown_table(export_rows, baseline, output_path):
    """Write a compact Markdown table for manuscript drafts."""
    with Path(output_path).open("w", encoding="utf-8") as handle:
        handle.write("# Paper Results Table\n\n")
        handle.write(
            "Research-matrix physical/Rayleigh comparison. Deltas are relative "
            f"to `{baseline.get('run', '')}`. Lower member AoI, stale75, and "
            "zero-participation are better; higher energy efficiency is better. "
            "Metric cells use `mean +/- ci95` when the comparison CSV contains "
            "CI95 columns.\n\n"
        )
        handle.write(
            "| Variant | Role | t<=1e-12 | Final error | Member AoI | "
            "Stale75 | Zero part. | Energy eff. | AoI vs base | "
            "Stale75 vs base | Zero vs base | Energy vs base |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in export_rows:
            handle.write(
                "| {label} | {role} | {t} | {error} | {aoi} | {stale} | "
                "{zero} | {eff} | {aoi_delta} | {stale_delta} | "
                "{zero_delta} | {eff_delta} |\n".format(
                    label=row["label"],
                    role=row["role"],
                    t=row["t_to_1e-12"],
                    error=row["final_error"],
                    aoi=row["member_aoi"],
                    stale=row["member_stale75"],
                    zero=row["member_zero"],
                    eff=row["energy_efficiency"],
                    aoi_delta=row["member_aoi_delta"],
                    stale_delta=row["member_stale75_delta"],
                    zero_delta=row["member_zero_delta"],
                    eff_delta=row["energy_efficiency_delta"],
                )
            )


def write_latex_table(export_rows, output_path, *, caption, table_label):
    """Write a LaTeX table that can be pasted into the paper."""
    with Path(output_path).open("w", encoding="utf-8") as handle:
        handle.write("% Generated by python -m experiments.export_paper_tables\n")
        handle.write("\\begin{table}[t]\n")
        handle.write("\\centering\n")
        handle.write(f"\\caption{{{_latex_escape(caption)}}}\n")
        handle.write(f"\\label{{{_latex_escape(table_label)}}}\n")
        handle.write("\\small\n")
        handle.write("\\begin{tabular}{llrrrrrrrr}\n")
        handle.write("\\hline\n")
        handle.write(
            "Variant & Role & $t_{\\le 10^{-12}}$ & Error & Member AoI & "
            "Stale75 & Zero & Eff. & Stale75 $\\Delta$ & Zero $\\Delta$ \\\\\n"
        )
        handle.write("\\hline\n")
        for row in export_rows:
            values = [
                _latex_escape(row["label"]),
                _latex_escape(row["role"]),
                row["t_to_1e-12"],
                _latex_value(row["final_error"]),
                _latex_value(row["member_aoi"]),
                _latex_value(row["member_stale75"]),
                _latex_value(row["member_zero"]),
                _latex_value(row["energy_efficiency"]),
                _latex_escape(row["member_stale75_delta"]),
                _latex_escape(row["member_zero_delta"]),
            ]
            handle.write(" & ".join(values) + " \\\\\n")
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")
        handle.write("\\end{table}\n")


def write_claim_bullets(rows, baseline, output_path):
    """Write concise paper-claim bullets from the comparison rows."""
    baseline_run = baseline.get("run", "")
    pure_rows = [
        row
        for row in rows
        if row is not baseline
        and not _is_upper_bound(row)
        and not _is_negative_ablation(row)
    ]
    upper_bound_rows = [row for row in rows if _is_upper_bound(row)]
    negative_rows = [row for row in rows if _is_negative_ablation(row)]

    dominating_rows = [
        row for row in pure_rows if _dominates_baseline_member_freshness(row, baseline)
    ]
    best_t_row = _best_min_row(pure_rows, "t_to_1e-12")
    best_energy_row = _best_max_row(pure_rows, "final_energy_efficiency")

    with Path(output_path).open("w", encoding="utf-8") as handle:
        handle.write("# Paper Claim Bullets\n\n")
        handle.write(
            f"- Baseline: `{baseline_run}` remains the main pure-ALOHA "
            "member-freshness reference for the K=3000 physical/Rayleigh setup.\n"
        )
        if dominating_rows:
            names = ", ".join(f"`{row.get('run', '')}`" for row in dominating_rows)
            handle.write(
                "- Pure ALOHA: the following ablation(s) dominate the baseline "
                f"on member AoI, stale75, and zero-participation: {names}.\n"
            )
        else:
            handle.write(
                "- Pure ALOHA: no tested ablation dominates the baseline across "
                "member AoI, stale75, and zero-participation, so ablations should "
                "be claimed narrowly.\n"
            )

        if best_t_row is not None:
            handle.write(
                "- Convergence ablation: "
                f"`{best_t_row.get('run', '')}` reaches `1e-12` at round "
                f"{_format_t(best_t_row.get('t_to_1e-12'))}"
            )
            baseline_t = _float_or_none(baseline.get("t_to_1e-12"))
            best_t = _float_or_none(best_t_row.get("t_to_1e-12"))
            if baseline_t is not None and best_t is not None:
                handle.write(f", {int(baseline_t - best_t)} rounds before baseline")
            handle.write(".\n")

        if best_energy_row is not None:
            gain = _relative_gain_text(
                _float_or_none(best_energy_row.get("final_energy_efficiency")),
                _float_or_none(baseline.get("final_energy_efficiency")),
            )
            handle.write(
                "- Energy ablation: "
                f"`{best_energy_row.get('run', '')}` has the highest pure-ALOHA "
                f"energy efficiency among ablations ({gain} versus baseline).\n"
            )

        if negative_rows:
            names = ", ".join(f"`{row.get('run', '')}`" for row in negative_rows)
            handle.write(
                "- Negative structural ablations: "
                f"{names} should be used to show that naive cluster splitting is "
                "not the source of the main gain.\n"
            )

        for row in upper_bound_rows:
            stale_reduction = _relative_reduction_text(
                _float_or_none(row.get("final_member_stale_fraction_75")),
                _float_or_none(baseline.get("final_member_stale_fraction_75")),
            )
            zero_reduction = _relative_reduction_text(
                _float_or_none(row.get("final_member_zero_participation_fraction")),
                _float_or_none(
                    baseline.get("final_member_zero_participation_fraction")
                ),
            )
            handle.write(
                "- Coordinated upper-bound: "
                f"`{row.get('run', '')}` reduces member stale75 by "
                f"{stale_reduction} and zero-participation by {zero_reduction}, "
                "but it is outside the pure-ALOHA contribution.\n"
            )


def short_label(row):
    """Return a compact human label for a run."""
    run = row.get("run", "")
    role = row.get("claim_role", "").lower()
    if "baseline" in role:
        return "Quota baseline"
    if run.startswith("member_collision_quota"):
        return "Collision-aware quota"
    if run.startswith("member_capped_quota"):
        cap = row.get("optimized_d2d_member_quota_cap_fraction", "")
        return f"Capped quota ({cap})" if cap not in ("", None) else "Capped quota"
    if run.startswith("member_safe_split"):
        return "Size-only safe split"
    if run.startswith("member_pressure_split"):
        return "Pressure-guided split"
    if run.startswith("member_split_") and "_s8_" in run:
        return "Global split max 8"
    if run.startswith("member_split_") and "_s5_" in run:
        return "Global split max 5"
    if run.startswith("member_semischedule"):
        return "Semi-scheduled upper bound"
    return run


def short_role(row):
    """Return a short paper role label."""
    role = row.get("claim_role", "")
    lowered = role.lower()
    if "baseline" in lowered:
        return "baseline"
    if "upper-bound" in lowered:
        return "upper-bound"
    if "negative" in lowered:
        return "negative ablation"
    if "structural" in lowered:
        return "structural ablation"
    if "pure-aloha" in lowered:
        return "pure-ALOHA ablation"
    return role


def _is_upper_bound(row):
    role = row.get("claim_role", "").lower()
    mode = row.get("optimized_d2d_access_mode", "").lower()
    return "upper-bound" in role or "semi_scheduled" in mode


def _is_negative_ablation(row):
    return "negative" in row.get("claim_role", "").lower()


def _dominates_baseline_member_freshness(row, baseline):
    comparisons = []
    for metric in LOWER_BETTER_MEMBER_METRICS:
        value = _float_or_none(row.get(metric))
        baseline_value = _float_or_none(baseline.get(metric))
        if value is None or baseline_value is None:
            return False
        comparisons.append(value <= baseline_value)
    return all(comparisons) and any(
        _float_or_none(row.get(metric)) < _float_or_none(baseline.get(metric))
        for metric in LOWER_BETTER_MEMBER_METRICS
    )


def _best_min_row(rows, key):
    rows = [row for row in rows if _float_or_none(row.get(key)) is not None]
    if not rows:
        return None
    return min(rows, key=lambda row: _float_or_none(row.get(key)))


def _best_max_row(rows, key):
    rows = [row for row in rows if _float_or_none(row.get(key)) is not None]
    if not rows:
        return None
    return max(rows, key=lambda row: _float_or_none(row.get(key)))


def _float_or_none(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative_delta(value, baseline):
    if value is None or baseline in (None, 0.0):
        return None
    return 100.0 * (value / baseline - 1.0)


def _format_delta_percent(value):
    if value is None:
        return ""
    return f"{value:+.1f}%"


def _format_fixed(value, digits):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _format_fixed_with_ci(value, ci95, digits):
    formatted_value = _format_fixed(value, digits)
    formatted_ci95 = _format_fixed(ci95, digits)
    if formatted_value and formatted_ci95:
        return f"{formatted_value} +/- {formatted_ci95}"
    return formatted_value


def _format_scientific(value):
    if value is None:
        return ""
    return f"{value:.3e}"


def _format_scientific_with_ci(value, ci95):
    formatted_value = _format_scientific(value)
    formatted_ci95 = _format_scientific(ci95)
    if formatted_value and formatted_ci95:
        return f"{formatted_value} +/- {formatted_ci95}"
    return formatted_value


def _format_t(value):
    numeric = _float_or_none(value)
    if numeric is None:
        return "n/a"
    return str(int(numeric))


def _relative_reduction_text(value, baseline):
    if value is None or baseline in (None, 0.0):
        return "n/a"
    return f"{100.0 * (baseline - value) / baseline:.1f}%"


def _relative_gain_text(value, baseline):
    if value is None or baseline in (None, 0.0):
        return "n/a"
    return f"{100.0 * (value - baseline) / baseline:+.1f}%"


def _latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _latex_value(text):
    """Escape a numeric table cell while preserving +/- as a LaTeX pm symbol."""
    return _latex_escape(text).replace("+/-", r"$\pm$")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=DEFAULT_COMPARISON_CSV,
        help=(
            "Comparison CSV produced by experiments.run_research_matrix or "
            "experiments.compare_runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for paper artifacts. Defaults to the comparison CSV folder.",
    )
    parser.add_argument(
        "--baseline-run",
        default=None,
        help="Optional run name to use as the delta baseline.",
    )
    parser.add_argument(
        "--caption",
        default=DEFAULT_CAPTION,
        help="Caption for the generated LaTeX table.",
    )
    parser.add_argument(
        "--table-label",
        default=DEFAULT_LABEL,
        help="LaTeX label for the generated table.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = load_comparison_rows(args.comparison_csv)
    output_dir = args.output_dir or args.comparison_csv.parent
    paths = write_paper_exports(
        rows,
        output_dir,
        baseline_run=args.baseline_run,
        caption=args.caption,
        table_label=args.table_label,
    )
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
