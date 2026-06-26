"""Generate figures from a JAX sweep CSV.

The simulation runner writes tabular outputs because CSV is easy to audit,
diff, copy into a thesis artifact, and reload from notebooks.  This module is
the notebook-free plotting layer: it turns those CSV files into image/vector
figures with confidence intervals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "polling": "Polling",
    "fixed_aloha": "Fixed ALOHA",
    "optimized_aloha": "Optimized ALOHA",
    "polling_d2d": "Polling + D2D",
    "fixed_aloha_d2d": "Fixed ALOHA + D2D",
    "optimized_aloha_d2d": "Optimized ALOHA + D2D",
}

ALL_SCENARIOS = tuple(SCENARIO_LABELS)
D2D_SCENARIOS = ("polling_d2d", "fixed_aloha_d2d", "optimized_aloha_d2d")

SCENARIO_PLOT_STYLES = {
    "polling": ("#d62728", "-", "o"),
    "fixed_aloha": ("#2ca02c", "--", "s"),
    "optimized_aloha": ("#1f77b4", "-.", "^"),
    "polling_d2d": ("#1f77b4", "-", "o"),
    "fixed_aloha_d2d": ("#ff7f0e", "--", "s"),
    "optimized_aloha_d2d": ("#2ca02c", "-.", "^"),
}

THESIS_FIGURE_15_STYLES = (
    ("polling", "Polling", "red", "-", "o"),
    ("polling_d2d", "Polling with D2D", "red", "--", "o"),
    ("fixed_aloha", "ALOHA with fixed access prob", "green", "-", "x"),
    ("fixed_aloha_d2d", "ALOHA with fixed access prob with D2D", "green", "--", "x"),
    ("optimized_aloha", "ALOHA with optimized access prob", "blue", "-", None),
    ("optimized_aloha_d2d", "ALOHA with optimized access prob with D2D", "blue", "--", None),
)

MEMBER_FAILURE_BREAKDOWN = (
    ("member_stale_compute_failure_fraction", "Member compute", "#4c78a8"),
    ("member_stale_link_failure_fraction", "Member link", "#f58518"),
    (
        "member_stale_member_energy_failure_fraction",
        "Member energy",
        "#54a24b",
    ),
    ("member_stale_ch_no_attempt_fraction", "CH no attempt", "#b279a2"),
    ("member_stale_collision_fraction", "ALOHA collision", "#e45756"),
    ("member_stale_ch_bs_failure_fraction", "CH-BS decode", "#72b7b2"),
    ("member_stale_other_failure_fraction", "Other", "#bab0ac"),
)

MEMBER_FAILURE_BREAKDOWN_WITH_CH_DETAIL = (
    ("member_stale_compute_failure_fraction", "Member compute", "#4c78a8"),
    ("member_stale_link_failure_fraction", "Member link", "#f58518"),
    (
        "member_stale_member_energy_failure_fraction",
        "Member energy",
        "#54a24b",
    ),
    ("member_stale_ch_compute_failure_fraction", "CH compute", "#9467bd"),
    ("member_stale_ch_energy_failure_fraction", "CH energy", "#8c564b"),
    ("member_stale_ch_access_no_draw_fraction", "CH access draw", "#b279a2"),
    ("member_stale_ch_not_scheduled_fraction", "CH not scheduled", "#ff9da6"),
    ("member_stale_ch_other_no_attempt_fraction", "CH no-attempt other", "#9d755d"),
    ("member_stale_collision_fraction", "ALOHA collision", "#e45756"),
    ("member_stale_ch_bs_failure_fraction", "CH-BS decode", "#72b7b2"),
    ("member_stale_other_failure_fraction", "Other", "#bab0ac"),
)


def _clean_formats(formats):
    """Normalize plot format strings passed by the CLI or another script."""
    cleaned = []
    for item in formats:
        normalized = str(item).strip().lstrip(".").lower()
        if normalized:
            cleaned.append(normalized)
    return tuple(dict.fromkeys(cleaned))


def _available_scenarios(frame, metric_name, scenarios):
    """Return scenarios that have mean columns for the requested metric."""
    return [
        scenario
        for scenario in scenarios
        if f"{scenario}_{metric_name}_mean" in frame.columns
    ]


def _save_figure(fig, output_stem, formats):
    """Save one Matplotlib figure to every requested format."""
    generated_paths = []
    for file_format in formats:
        output_path = output_stem.with_suffix(f".{file_format}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        generated_paths.append(output_path)
    plt.close(fig)
    return generated_paths


def _plot_metric(frame, metric_name, scenarios, ylabel, title, output_stem, formats):
    """Plot one metric over t with optional 95 percent confidence bands."""
    x_values = frame["t"].to_numpy(dtype=float)
    available_scenarios = _available_scenarios(frame, metric_name, scenarios)
    if not available_scenarios:
        return []

    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    marker_stride = 1 if len(x_values) <= 20 else max(1, len(x_values) // 10)

    for scenario_index, scenario in enumerate(available_scenarios):
        mean_column = f"{scenario}_{metric_name}_mean"
        ci_column = f"{scenario}_{metric_name}_ci95"
        y_values = frame[mean_column].to_numpy(dtype=float)
        color, linestyle, marker = SCENARIO_PLOT_STYLES.get(
            scenario,
            (None, "-", None),
        )
        markevery = None
        if marker is not None:
            markevery = (scenario_index % marker_stride, marker_stride)

        line = axis.plot(
            x_values,
            y_values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=markevery,
            linewidth=2.0,
            markersize=4.5,
            label=SCENARIO_LABELS.get(scenario, scenario),
        )[0]

        if ci_column in frame.columns:
            ci_values = frame[ci_column].to_numpy(dtype=float)
            axis.fill_between(
                x_values,
                y_values - ci_values,
                y_values + ci_values,
                color=line.get_color(),
                alpha=0.14,
                linewidth=0.0,
            )

    axis.set_xlabel("FL iterations (t)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _save_figure(fig, output_stem, formats)


def _plot_thesis_figure_15(frame, output_stem, formats):
    """Recreate the thesis-style error-norm comparison figure.

    Figure 15 in the thesis compares the three CH/device selection models with
    and without D2D-SRC on a log-scaled error-norm axis.  This plot intentionally
    uses the thesis color/style convention: red polling, green fixed ALOHA, blue
    optimized ALOHA, with dashed lines for the D2D variants.

    The notebook version used ``LIST_OF_FL_ITERATIONS_PER_ROUND = [1, 200]``.
    For better visual diagnosis, this version plots every checkpoint saved in
    the CSV.  Omit ``--checkpoints`` in the sweep command to save every
    iteration ``t = 1..max_t``.
    """
    thesis_frame = frame
    x_values = thesis_frame["t"].to_numpy(dtype=float)
    marker_stride = 1 if len(x_values) <= 20 else max(1, len(x_values) // 10)
    fig, axis = plt.subplots(figsize=(7.6, 5.4))

    for scenario, label, color, linestyle, marker in THESIS_FIGURE_15_STYLES:
        mean_column = f"{scenario}_error_norm_mean"
        if mean_column not in frame.columns:
            continue

        y_values = thesis_frame[mean_column].to_numpy(dtype=float)
        y_values = np.maximum(y_values, np.finfo(float).tiny)
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
    axis.grid(True, which="both", alpha=0.55)
    axis.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return _save_figure(fig, output_stem, formats)


def _plot_cluster_rate(frame, output_stem, formats):
    """Plot the clustered-device rate for the sweep."""
    mean_value = float(frame["clusterized_devices_rate_mean"].iloc[0])
    ci_value = float(frame["clusterized_devices_rate_ci95"].iloc[0])
    mode = str(frame["clustering_mode"].iloc[0])

    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.bar([mode], [mean_value], yerr=[ci_value], capsize=7, color="#3b6ea8")
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Clustered devices (%)")
    axis.set_title("D2D clustering rate")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_stem, formats)


def _plot_cluster_quality(frame, output_stem, formats):
    """Plot structural clustering metrics when the CSV contains them."""
    required_columns = {
        "number_of_clusters_mean",
        "singleton_count_mean",
        "non_singleton_cluster_count_mean",
        "clustered_devices_count_mean",
        "mean_cluster_size_mean",
        "mean_non_singleton_cluster_size_mean",
    }
    if not required_columns.issubset(frame.columns):
        return []

    first_row = frame.iloc[0]
    count_metrics = (
        ("clustered_devices_count", "Clustered devices"),
        ("singleton_count", "Singletons"),
        ("non_singleton_cluster_count", "D2D clusters"),
        ("number_of_clusters", "Total CH rows"),
    )
    size_metrics = (
        ("mean_cluster_size", "Mean cluster size"),
        ("mean_non_singleton_cluster_size", "Mean D2D cluster size"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))

    count_values = [float(first_row[f"{metric}_mean"]) for metric, _ in count_metrics]
    count_errors = [float(first_row.get(f"{metric}_ci95", 0.0)) for metric, _ in count_metrics]
    axes[0].bar(
        [label for _, label in count_metrics],
        count_values,
        yerr=count_errors,
        capsize=5,
        color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"],
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title("Cluster structure")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)

    size_values = [float(first_row[f"{metric}_mean"]) for metric, _ in size_metrics]
    size_errors = [float(first_row.get(f"{metric}_ci95", 0.0)) for metric, _ in size_metrics]
    axes[1].bar(
        [label for _, label in size_metrics],
        size_values,
        yerr=size_errors,
        capsize=5,
        color=["#72b7b2", "#e45756"],
    )
    axes[1].set_ylabel("Devices per cluster")
    axes[1].set_title("Aggregate quality")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    return _save_figure(fig, output_stem, formats)


def _plot_member_failure_breakdown(
    frame,
    output_stem,
    formats,
    scenario="optimized_aloha_d2d",
):
    """Plot the reason stale D2D members were not refreshed."""
    breakdown = MEMBER_FAILURE_BREAKDOWN
    if all(
        f"{scenario}_{metric_name}_mean" in frame.columns
        for metric_name, _, _ in MEMBER_FAILURE_BREAKDOWN_WITH_CH_DETAIL
    ):
        breakdown = MEMBER_FAILURE_BREAKDOWN_WITH_CH_DETAIL
    available_metrics = [
        (metric_name, label, color)
        for metric_name, label, color in breakdown
        if f"{scenario}_{metric_name}_mean" in frame.columns
    ]
    if not available_metrics:
        return []

    x_values = frame["t"].to_numpy(dtype=float)
    y_values = [
        frame[f"{scenario}_{metric_name}_mean"].to_numpy(dtype=float)
        for metric_name, _, _ in available_metrics
    ]
    labels = [label for _, label, _ in available_metrics]
    colors = [color for _, _, color in available_metrics]

    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.stackplot(
        x_values,
        y_values,
        labels=labels,
        colors=colors,
        alpha=0.88,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("FL iterations (t)")
    axis.set_ylabel("Fraction of stale D2D-clustered devices")
    axis.set_title(
        f"Member Stale Failure Breakdown: {SCENARIO_LABELS.get(scenario, scenario)}"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return _save_figure(fig, output_stem, formats)


def plot_sweep_csv(csv_path, output_dir=None, formats=("png", "pdf")):
    """Generate all standard figures for one sweep CSV.

    Returns a list of generated figure paths.  The expected CSV schema is the
    one written by ``experiments.run_gpu_sweep``.
    """
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"{csv_path} is empty")
    if "t" not in frame.columns:
        raise ValueError(f"{csv_path} does not contain a 't' column")

    formats = _clean_formats(formats)
    if not formats:
        raise ValueError("at least one plot format must be requested")

    if output_dir is None:
        output_dir = csv_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_stem = output_dir / csv_path.stem

    generated_paths = []
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="error_norm",
            scenarios=ALL_SCENARIOS,
            ylabel="Mean error norm",
            title="Error norm over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_error_norm"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_thesis_figure_15(
            frame,
            output_stem=base_stem.with_name(f"{base_stem.name}_figure_15_error_norm"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="uploads",
            scenarios=ALL_SCENARIOS,
            ylabel="Successful device updates",
            title="Successful uploads over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_uploads"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="clusterhead_uploads",
            scenarios=D2D_SCENARIOS,
            ylabel="Successful CH uploads",
            title="Cluster-head uploads over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_clusterhead_uploads"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="battery",
            scenarios=ALL_SCENARIOS,
            ylabel="Mean battery level",
            title="Mean battery over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_battery"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="clusterhead_battery",
            scenarios=D2D_SCENARIOS,
            ylabel="Mean CH battery level",
            title="Cluster-head battery over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_clusterhead_battery"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="energy_used",
            scenarios=ALL_SCENARIOS,
            ylabel="Mean normalized energy used",
            title="Mean energy used over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_energy_used"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="energy_efficiency",
            scenarios=ALL_SCENARIOS,
            ylabel="Uploads per normalized battery unit",
            title="Energy efficiency over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_energy_efficiency"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="clusterhead_energy_used",
            scenarios=D2D_SCENARIOS,
            ylabel="Mean CH normalized energy used",
            title="Cluster-head energy used over FL iterations",
            output_stem=base_stem.with_name(
                f"{base_stem.name}_clusterhead_energy_used"
            ),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="aoi",
            scenarios=ALL_SCENARIOS,
            ylabel="Mean AoI",
            title="Mean Age of Information over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_aoi"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="peak_aoi",
            scenarios=ALL_SCENARIOS,
            ylabel="Peak AoI",
            title="Peak Age of Information over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_peak_aoi"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="p95_aoi",
            scenarios=ALL_SCENARIOS,
            ylabel="95th percentile AoI",
            title="95th Percentile Age of Information over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_p95_aoi"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="p75_aoi",
            scenarios=ALL_SCENARIOS,
            ylabel="75th percentile AoI",
            title="75th Percentile Age of Information over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_p75_aoi"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="p90_aoi",
            scenarios=ALL_SCENARIOS,
            ylabel="90th percentile AoI",
            title="90th Percentile Age of Information over FL iterations",
            output_stem=base_stem.with_name(f"{base_stem.name}_p90_aoi"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="stale_fraction_50",
            scenarios=ALL_SCENARIOS,
            ylabel="Fraction with AoI > 1 + 50% of elapsed t",
            title="Stale-tail fraction over FL iterations",
            output_stem=base_stem.with_name(
                f"{base_stem.name}_stale_fraction_50"
            ),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="stale_fraction_75",
            scenarios=ALL_SCENARIOS,
            ylabel="Fraction with AoI > 1 + 75% of elapsed t",
            title="Severe stale-tail fraction over FL iterations",
            output_stem=base_stem.with_name(
                f"{base_stem.name}_stale_fraction_75"
            ),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="stale_fraction_100",
            scenarios=ALL_SCENARIOS,
            ylabel="Fraction with AoI > 100 rounds",
            title="Long-stale fraction over FL iterations",
            output_stem=base_stem.with_name(
                f"{base_stem.name}_stale_fraction_100"
            ),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="member_aoi",
            scenarios=D2D_SCENARIOS,
            ylabel="Mean AoI for D2D-clustered devices",
            title="D2D-Clustered Device Age of Information",
            output_stem=base_stem.with_name(f"{base_stem.name}_member_aoi"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="member_p95_aoi",
            scenarios=D2D_SCENARIOS,
            ylabel="95th percentile AoI for D2D-clustered devices",
            title="D2D-Clustered Device 95th Percentile AoI",
            output_stem=base_stem.with_name(f"{base_stem.name}_member_p95_aoi"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="member_stale_fraction_75",
            scenarios=D2D_SCENARIOS,
            ylabel="Fraction of D2D-clustered devices with AoI > 1 + 75% of elapsed t",
            title="D2D-Clustered Device Stale Fraction",
            output_stem=base_stem.with_name(
                f"{base_stem.name}_member_stale_fraction_75"
            ),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_metric(
            frame,
            metric_name="member_zero_participation_fraction",
            scenarios=D2D_SCENARIOS,
            ylabel="Fraction of D2D-clustered devices with zero delivered updates",
            title="D2D-Clustered Device Zero-Participation Fraction",
            output_stem=base_stem.with_name(
                f"{base_stem.name}_member_zero_participation_fraction"
            ),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_member_failure_breakdown(
            frame,
            output_stem=base_stem.with_name(
                f"{base_stem.name}_member_failure_breakdown"
            ),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_cluster_rate(
            frame,
            output_stem=base_stem.with_name(f"{base_stem.name}_cluster_rate"),
            formats=formats,
        )
    )
    generated_paths.extend(
        _plot_cluster_quality(
            frame,
            output_stem=base_stem.with_name(f"{base_stem.name}_cluster_quality"),
            formats=formats,
        )
    )
    return generated_paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=("png", "pdf"))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    generated_paths = plot_sweep_csv(
        args.csv_path,
        output_dir=args.output_dir,
        formats=args.formats,
    )
    for output_path in generated_paths:
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
