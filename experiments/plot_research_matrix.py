"""Plot canonical research-matrix figures from a comparison CSV.

This is the figure companion to ``experiments.export_paper_tables``. It reads a
completed ``run_comparison_summary.csv`` and produces paper-facing bar/scatter
figures without importing JAX or rerunning simulations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.export_paper_tables import (
    DEFAULT_COMPARISON_CSV,
    load_comparison_rows,
    select_baseline,
    short_label,
    short_role,
)


ROLE_STYLES = {
    "baseline": ("#2f6f9f", ""),
    "pure-ALOHA ablation": ("#4c9f70", ""),
    "structural ablation": ("#7f6bb2", "//"),
    "negative ablation": ("#c44e52", "\\\\"),
    "upper-bound": ("#dd8a34", ".."),
}

BAR_SPECS = (
    (
        "final_member_stale_fraction_75",
        "research_matrix_member_stale75",
        "Member stale75 fraction",
        "Member stale75 fraction",
        "lower",
    ),
    (
        "final_member_zero_participation_fraction",
        "research_matrix_member_zero_participation",
        "Zero-participation fraction",
        "Member zero-participation fraction",
        "lower",
    ),
    (
        "final_energy_efficiency",
        "research_matrix_energy_efficiency",
        "Energy efficiency",
        "Energy efficiency",
        "higher",
    ),
    (
        "t_to_1e-12",
        "research_matrix_t_to_1e12",
        "Rounds to error <= 1e-12",
        "FL round",
        "lower",
    ),
)


def plot_research_matrix(
    comparison_csv=DEFAULT_COMPARISON_CSV,
    *,
    output_dir=None,
    formats=("png", "pdf"),
    baseline_run=None,
):
    """Generate canonical figures for a completed research comparison."""
    comparison_csv = Path(comparison_csv)
    output_dir = Path(output_dir) if output_dir is not None else comparison_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = _clean_formats(formats)
    rows = load_comparison_rows(comparison_csv)
    baseline = select_baseline(rows, baseline_run=baseline_run)

    generated = []
    for metric, stem, title, ylabel, direction in BAR_SPECS:
        generated.extend(
            _plot_bar_metric(
                rows,
                baseline,
                metric,
                output_dir / stem,
                title=title,
                ylabel=ylabel,
                direction=direction,
                formats=formats,
            )
        )

    generated.extend(
        _plot_pareto(
            rows,
            baseline,
            output_dir / "research_matrix_stale75_energy_pareto",
            formats=formats,
        )
    )
    return generated


def _plot_bar_metric(
    rows,
    baseline,
    metric,
    output_stem,
    *,
    title,
    ylabel,
    direction,
    formats,
):
    values = [_float_or_none(row.get(metric)) for row in rows]
    ci95_values = [_float_or_none(row.get(f"{metric}_ci95")) for row in rows]
    indexed = [
        (index, row, value, ci95)
        for index, (row, value, ci95) in enumerate(zip(rows, values, ci95_values))
        if value is not None
    ]
    if not indexed:
        return []

    labels = [_plot_label(row) for _, row, _, _ in indexed]
    plotted_values = [value for _, _, value, _ in indexed]
    error_values = [0.0 if ci95 is None else ci95 for _, _, _, ci95 in indexed]
    yerr = error_values if any(value > 0.0 for value in error_values) else None
    colors = [_style_for_row(row)[0] for _, row, _, _ in indexed]
    hatches = [_style_for_row(row)[1] for _, row, _, _ in indexed]

    fig_width = max(8.0, 0.82 * len(labels) + 2.0)
    fig, axis = plt.subplots(figsize=(fig_width, 4.9))
    bars = axis.bar(
        range(len(labels)),
        plotted_values,
        yerr=yerr,
        capsize=4 if yerr is not None else 0,
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
    )
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    baseline_value = _float_or_none(baseline.get(metric))
    if baseline_value is not None:
        axis.axhline(
            baseline_value,
            color="#333333",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
        )

    for bar, value in zip(bars, plotted_values):
        axis.annotate(
            _format_annotation(value),
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=32, ha="right")
    axis.set_ylabel(ylabel)
    title_suffix = "with CI95" if yerr is not None else "no CI95 in CSV"
    axis.set_title(f"{title} ({direction} is better; {title_suffix})")
    axis.grid(True, axis="y", alpha=0.22)
    axis.legend(
        handles=_legend_handles(rows),
        loc="best",
        fontsize=8.5,
        frameon=True,
    )
    fig.tight_layout()
    return _save_figure(fig, output_stem, formats)


def _plot_pareto(rows, baseline, output_stem, *, formats):
    x_metric = "final_member_stale_fraction_75"
    y_metric = "final_energy_efficiency"
    points = []
    for row in rows:
        x_value = _float_or_none(row.get(x_metric))
        x_ci95 = _float_or_none(row.get(f"{x_metric}_ci95"))
        y_value = _float_or_none(row.get(y_metric))
        y_ci95 = _float_or_none(row.get(f"{y_metric}_ci95"))
        if x_value is not None and y_value is not None:
            points.append((row, x_value, x_ci95, y_value, y_ci95))
    if not points:
        return []

    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    for index, (row, x_value, x_ci95, y_value, y_ci95) in enumerate(points):
        color, _ = _style_for_row(row)
        marker = "*" if short_role(row) == "upper-bound" else "o"
        size = 165 if row is baseline else 105
        axis.errorbar(
            [x_value],
            [y_value],
            xerr=None if x_ci95 is None else [[x_ci95], [x_ci95]],
            yerr=None if y_ci95 is None else [[y_ci95], [y_ci95]],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            alpha=0.75,
            zorder=2,
        )
        axis.scatter(
            [x_value],
            [y_value],
            s=size,
            marker=marker,
            color=color,
            edgecolor="#333333",
            linewidth=0.8,
            label=short_role(row),
            zorder=3,
        )
        offset_x, offset_y = _pareto_label_offset(row, index)
        axis.annotate(
            _plot_label(row),
            xy=(x_value, y_value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            ha="right" if offset_x < 0 else "left",
            fontsize=8,
        )

    baseline_x = _float_or_none(baseline.get(x_metric))
    baseline_y = _float_or_none(baseline.get(y_metric))
    if baseline_x is not None:
        axis.axvline(baseline_x, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
    if baseline_y is not None:
        axis.axhline(baseline_y, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)

    axis.margins(x=0.05, y=0.14)
    axis.set_xlabel("Member stale75 fraction (lower is better)")
    axis.set_ylabel("Energy efficiency (higher is better)")
    axis.set_title("Member freshness vs energy-efficiency tradeoff")
    axis.grid(True, alpha=0.22)
    axis.legend(
        handles=_legend_handles([row for row, _, _, _, _ in points]),
        loc="best",
        fontsize=8.5,
        frameon=True,
    )
    fig.tight_layout()
    return _save_figure(fig, output_stem, formats)


def _pareto_label_offset(row, index):
    """Return deterministic label offsets for dense K=3000 tradeoff points."""
    label = _plot_label(row)
    offsets_by_label = {
        "Baseline": (28, 16),
        "Collision quota": (28, -18),
        "Capped quota": (28, 30),
        "Safe split": (-64, -22),
        "Pressure split": (28, -34),
        "Split max 8": (28, 8),
        "Split max 5": (28, 8),
        "Upper bound": (28, 12),
    }
    fallback_offsets = ((10, 8), (10, -14), (10, 20), (-28, -12))
    return offsets_by_label.get(label, fallback_offsets[index % len(fallback_offsets)])


def _plot_label(row):
    """Return compact labels for figures."""
    label = short_label(row)
    replacements = {
        "Quota baseline": "Baseline",
        "Collision-aware quota": "Collision quota",
        "Capped quota (0.1)": "Capped quota",
        "Size-only safe split": "Safe split",
        "Pressure-guided split": "Pressure split",
        "Global split max 8": "Split max 8",
        "Global split max 5": "Split max 5",
        "Semi-scheduled upper bound": "Upper bound",
    }
    return replacements.get(label, label)


def _style_for_row(row):
    return ROLE_STYLES.get(short_role(row), ("#777777", ""))


def _legend_handles(rows):
    ordered_roles = []
    for row in rows:
        role = short_role(row)
        if role not in ordered_roles:
            ordered_roles.append(role)
    handles = []
    for role in ordered_roles:
        color, hatch = ROLE_STYLES.get(role, ("#777777", ""))
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="#333333",
                hatch=hatch,
                label=role,
            )
        )
    return handles


def _save_figure(fig, output_stem, formats):
    generated_paths = []
    for file_format in formats:
        output_path = output_stem.with_suffix(f".{file_format}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        generated_paths.append(output_path)
    plt.close(fig)
    return generated_paths


def _clean_formats(formats):
    cleaned = []
    for item in formats:
        normalized = str(item).strip().lstrip(".").lower()
        if normalized:
            cleaned.append(normalized)
    return tuple(dict.fromkeys(cleaned))


def _float_or_none(value):
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_annotation(value):
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-csv",
        type=Path,
        default=DEFAULT_COMPARISON_CSV,
        help="Comparison CSV produced by experiments.run_research_matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures. Defaults to the comparison CSV folder.",
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        default=("png", "pdf"),
        help="Figure formats to write. Default: png pdf.",
    )
    parser.add_argument(
        "--baseline-run",
        default=None,
        help="Optional run name to use as the visual baseline.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    generated = plot_research_matrix(
        args.comparison_csv,
        output_dir=args.output_dir,
        formats=args.plot_formats,
        baseline_run=args.baseline_run,
    )
    for path in generated:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
