# Project Architecture

The repository is organized around a `.py`-first workflow. Notebooks are useful
for analysis and inspection, but they are not required to generate research
outputs.

## Entry Points

`main.py`

- Human-friendly root command.
- Delegates to `experiments.run_gpu_sweep`.
- Use this when you want the default experiment pipeline from the repository
  root.

`experiments/run_gpu_sweep.py`

- Main experiment runner.
- Generates devices, clusters them, simulates HFL/ALOHA, writes CSV/metadata,
  and creates figures.
- Records both outcome metrics and cluster-quality metrics, including CH row
  count, singleton count, and average D2D cluster size.
- Records optional dynamic-energy metrics, including per-scenario mean battery,
  D2D cluster-head mean battery, energy used, energy efficiency, and D2D
  cluster-head energy used.
- Records AoI metrics for all six scenarios: mean AoI, peak AoI, p75/p90/p95
  AoI, and stale-tail fractions.
- Creates `Runs/<timestamp>/` by default after the simulation succeeds.

`experiments/plot_gpu_sweep.py`

- Regenerates figures from an existing sweep CSV.
- Generates battery, D2D cluster-head battery, energy-used, energy-efficiency,
  D2D cluster-head energy-used, mean-AoI, peak-AoI, p75/p90/p95-AoI, and
  stale-tail fraction plots only when the CSV has the corresponding optional
  columns.
- Does not rerun the simulation.

`experiments/compare_runs.py`

- Compares completed `results.csv` files or run folders without importing JAX.
- Writes `run_comparison_summary.csv` and `run_comparison_summary.md`.
- Reports target times, log-error AUC, final uploads, CH uploads, energy,
  energy efficiency, AoI percentiles, and stale-tail fractions for a selected
  scenario, defaulting to `optimized_aloha_d2d`.

`experiments/run_utility_pareto_sweep.py`

- Runs optimized-D2D utility candidate grids. The default `refined` grid
  searches the high-performing neighborhood found by the first full K=3000
  sweep; `--candidate-grid coarse` keeps the older 243-point grid available.
- Writes per-candidate result folders plus `utility_sweep_summary.csv`,
  `utility_sweep_top10.md`, an AUC Pareto plot, and a target-time Pareto plot.
- Ranks candidates by time to reach error targets `1e-6`, `1e-9`, and `1e-12`,
  with log-error AUC as a tie-breaker, while constraining CH uploads near fixed
  D2D.
- Supports `--candidate-start` and `--candidate-count` for Colab-friendly
  partial grid runs.

`experiments/run_ch_quality_weight_sweep.py`

- Runs focused quality-CH election weight comparisons using the current
  channel-aware enhanced defaults.
- The default `finalists` profile compares the current two strongest
  channel-heavy CH election candidates: channel-only and channel-plus-battery.
- Writes per-candidate result folders plus `ch_quality_weight_summary.csv`,
  `ch_quality_weight_top.md`, metadata, a best-candidate fair error-norm plot,
  and a candidate overlay plot for optimized-D2D error norm.
- The summary includes fair D2D-vs-direct metrics, including direct optimized
  error at `t=200`, D2D/direct error ratio, D2D log10 gain, and D2D/direct
  upload ratio.
- Use `--summarize-existing-run-dir Runs/<name>` to regenerate those summaries
  and error-norm plots from an already completed sweep without rerunning JAX.

`experiments/run_energy_rotation_sweep.py`

- Runs the fixed static/performance/balanced/eco D2D CH-rotation comparison.
- Can optionally add AoI-triggered performance-profile candidates with
  `--include-aoi-triggered-rotation`.
- Uses the current channel-aware utility optimized-D2D defaults and dynamic
  energy drain, then writes one subfolder per rotation profile.
- Writes `energy_rotation_summary.csv`, `energy_rotation_summary.md`, an
  optimized-D2D error overlay, and an error-vs-CH-battery tradeoff plot.
- Ranks profiles conservatively: keep final optimized-D2D error and total
  optimized-D2D energy within tolerance versus static, then maximize final CH
  battery.

`experiments/merge_utility_pareto_summaries.py`

- Combines summary CSVs from multiple utility Pareto parts.
- Re-ranks all candidates and regenerates the consolidated top-10 and Pareto
  plot.

## Strategy Documentation

`docs/optimized_d2d_strategy_report.md`

- Records the optimized-D2D strategy evolution, representative results,
  required control information, and plausible real-world deployment
  arrangement for the strategies tested so far.

`docs/current_architecture_considerations.md`

- Explains the current enhanced architecture in terms of implementable wireless
  signals: battery, BS channel quality, D2D degree, freshness, CH election,
  member-to-CH availability, direct device-to-BS decoding, CH-to-BS decoding,
  and load allocation.
- Records which assumptions are thesis-compatible, which are enhanced
  ablations, and which limitations remain before claiming real-world
  comparability.

`docs/current_status_and_future_work.md`

- Summarizes what has already been implemented, what the current experimental
  results mean, which pieces remain heuristic, and the recommended future-work
  order for physical energy modeling, battery feasibility, outage/SINR-based
  channel success, AoI metrics, and hyperparameter validation.

## Core Modules

`Clustering/jax_clustering_algorithm.py`

- JAX device generation and GPU-oriented one-hop D2D clustering.
- Produces fixed-shape padded cluster arrays.
- Dense mode includes local singleton repair, pair CH-rotation repair, and
  CH-to-CH merge repair while preserving one-hop CH coverage and `Cmax`.
- Optional cluster splitting re-clusters large rows locally into valid one-hop
  subclusters before the FL simulation, preserving ALOHA access while changing
  D2D membership. `max_size` is the global split ablation; `safe_max_size`
  limits the split budget and rejects tiny-tail split proposals;
  `pressure_safe_max_size` uses the same guard but ranks candidates by static
  member-to-CH and CH-to-BS risk.
- Optional quality CH election keeps cluster membership fixed but rotates the
  CH role to the best valid member according to D2D degree, BS channel quality,
  and battery.
- Quality CH election can score the BS channel with normalized inverse
  pathloss or Rayleigh outage probability.

`Models/jax_models_arrangement.py`

- JAX HFL/ALOHA simulation.
- Uses `jax.lax.scan` to run one trajectory and record all requested
  checkpoints.
- Supports optional member-to-CH D2D decoding, direct device-to-BS decoding for
  non-D2D curves, and CH-to-BS decoding for D2D curves. Devices and CHs still
  contend through ALOHA first; collision-free packets may then fail under
  either legacy scalar/channel-quality abstractions or the enhanced
  `rayleigh_outage` model. Member-to-CH Rayleigh mode uses each member's
  distance to the currently elected CH.
- Supports optional dynamic battery drain with independent battery state for
  each of the six counterfactual curves. Energy can use legacy constant costs
  or the enhanced first-order radio model that separates direct BS transmit,
  D2D member transmit, CH receive, CH aggregation, CH-BS transmit, and optional
  CH-rotation control overhead.
- Supports optional battery feasibility, where a device or CH only attempts a
  role if its current battery can pay the required energy.
- Tracks mean, peak, p75, p90, p95, and stale-tail Age of Information metrics
  for every scenario.
- Supports optional energy-aware intra-run CH rotation for the three D2D curves.
  The selected CH must be an existing cluster member and preserve one-hop
  coverage of the fixed cluster membership. Rotation can be periodic,
  AoI-triggered, or both.
- Contains thesis-compatible optimized D2D plus enhanced `utility`,
  `max_weight`, `hybrid`, `adaptive_diversity`, `aoi_aware_utility`,
  `aoi_floor_utility`, `aoi_tail_utility`, and `aoi_quality_tail_utility` CH
  access policies.
- Load-controlled enhanced policies share the same allocator family:
  `proportional_clip`, `water_filling`, `selective_water_filling`, and the
  default `conditional_selective_water_filling`, which only redistributes lost
  target load when ACK-observed optimized-D2D CH throughput is materially below
  the expected fixed-D2D CH throughput; otherwise it returns the exact
  proportional clipped allocator. The conditional allocator also has a
  density-aware trigger: highly clusterized deployments can use a lower trigger
  ratio so redistribution does not add collision pressure when D2D coverage is
  already near-complete.

`Models/models_arrangement.py`

- Compatibility facade for older imports.
- Re-exports the JAX model API.

## Notebooks

`notebooks/explore_results.ipynb`

- Optional analysis notebook.
- Should read CSV, metadata, and figures from `Runs/`.
- Should not be required to execute the simulation.

`notebooks/proposed_clustering_step_by_step.ipynb`

- Optional algorithm-inspection notebook.
- Useful for explaining or visualizing the clustering flow.

## Recommended Workflow

Run an experiment:

```bash
python main.py --run-name local_smoke --devices 100 --rounds 5 --iterations 20 --checkpoints 5 10 20
```

Regenerate figures:

```bash
python -m experiments.plot_gpu_sweep Runs/local_smoke/results.csv
```

Tune utility parameters:

```bash
python -m experiments.run_utility_pareto_sweep --run-name utility_pareto_smoke --devices 1000 --rounds 20 --precision float64 --max-candidates 5
```

Run the adaptive-diversity optimized-D2D smoke experiment:

```bash
python main.py --run-name k3000_adaptive_diversity_smoke --devices 1000 --rounds 20 --precision float64 --optimized-d2d-access-mode adaptive_diversity --optimized-d2d-load-allocation-mode conditional_selective_water_filling --optimized-d2d-redistribution-fraction 0.25 --optimized-d2d-redistribution-trigger-ratio 0.95 --optimized-d2d-density-trigger-threshold 0.95 --optimized-d2d-dense-trigger-ratio 0.90 --optimized-d2d-throughput-ewma-decay 0.90 --optimized-d2d-access-floor-fraction 0.02 --optimized-d2d-norm-exponent 3.5 --optimized-d2d-cluster-size-exponent 1.5 --optimized-d2d-freshness-exponent 0.25 --optimized-d2d-late-norm-exponent 1.25 --optimized-d2d-late-freshness-exponent 1.0 --optimized-d2d-novelty-exponent 1.5 --optimized-d2d-load-target-factor 1.1
```

Merge utility tuning parts:

```bash
python -m experiments.merge_utility_pareto_summaries Runs/utility_pareto_k3000_part* --output-dir Runs/utility_pareto_k3000_merged
```

Explore results:

```text
Open notebooks/explore_results.ipynb and load files from Runs/<run-name>/.
```
