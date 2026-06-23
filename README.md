# Clustering for Federated Learning with ALOHA

This repository contains the simulation code used for the master's thesis
["A distributed D2D clustering algorithm tailored for hierarchical federated learning in a multichannel ALOHA network"](https://repositorio.utfpr.edu.br/jspui/handle/1/34706).

The project simulates a geometry-based D2D Short Range Clustering (D2D-SRC)
heuristic and evaluates it inside a hierarchical federated learning (HFL)
system that uses multichannel ALOHA for cluster-head-to-BS communication.

## What Is Implemented

- Device deployment around a BS using the thesis default model:
  `r ~ U(1, R_BS)` and `theta ~ U(0, 2*pi)`.
- D2D-SRC clustering with one-hop CH constraints and maximum cluster size.
- Three CH/device selection models:
  - polling;
  - ALOHA with fixed access probability;
  - ALOHA with optimized access probability based on update norm.
- HFL first-tier aggregation at CHs and the thesis-figure BS update
  `w <- w - u1 * gradient` by default. Use `--normalize-by-k` for the
  conservative `gradient / K` ablation.
- Seeded JAX simulation paths for reproducible figures and logs.
- GPU-first JAX kernels using fixed-shape arrays, `jax.vmap` for Monte Carlo
  batching, and `jax.lax.scan` for one trajectory that records every requested
  `t` checkpoint.
- JAX clustering variants:
  - `geometric`, a one-hop D2D-SRC-style mode using dense radius clustering by
    default and grid clustering as the large-`K` option;
  - `utility`, a degree/battery/channel-quality ranking mode;
  - `no_d2d`, singleton clusters for baseline comparison.

The default `geometric` clustering mode uses geometry only. The `utility` mode
adds degree, battery, and BS channel-quality scoring. Neither mode currently
uses data distribution or learning similarity. Live energy evolution is
available only when `--energy-drain-mode dynamic` is explicitly enabled; the
default keeps battery fixed for thesis-compatible comparisons.

## Repository Layout

```text
Clustering/
  proposed_clustering_algorithm.py          D2D-SRC generator, clustering, validation
  jax_clustering_algorithm.py               GPU-oriented JAX device generation and clustering
Models/
  models_arrangement.py                     Compatibility facade for the JAX model backend
  jax_models_arrangement.py                 JAX HFL/ALOHA trace simulation
docs/
  project_architecture.md                   Where entry points, modules, and notebooks belong
  current_architecture_considerations.md     Current architecture signals, assumptions, and limits
  gpu_jax_backend.md                        GPU backend notes, validation, and Colab guidance
  modeling_assumptions.md                   Thesis defaults and realism switches
  optimized_d2d_strategy_report.md          Strategy evolution, results, and deployment notes
experiments/
  run_gpu_sweep.py                          Batched JAX experiment runner
  run_utility_pareto_sweep.py               Utility parameter Pareto tuning runner
  run_ch_quality_weight_sweep.py            Focused quality-CH weight comparison and plots
  merge_utility_pareto_summaries.py         Merge utility Pareto partial runs
  plot_gpu_sweep.py                         CSV-to-figure plotting CLI
  run_ablation.py                           Compatibility wrapper around the GPU sweep runner
tests/
  test_clustering.py
  test_enhanced_clustering.py
  test_enhanced_models.py
  test_models.py
notebooks/
  explore_results.ipynb                     Optional CSV/metadata/figure analysis notebook
  proposed_clustering_step_by_step.ipynb    Optional clustering explanation notebook
main.py                                     Root CLI entry point for the full .py workflow
requirements.txt
requirements-colab-gpu.txt
```

Generated run outputs are intentionally ignored by Git. Recreate them by running
`main.py` or the experiment scripts. See `docs/project_architecture.md` for the
intended module boundaries.

## Environment

For local Windows development, install the base requirements. This gives you a
JAX CPU runtime for syntax checks, unit tests, and small smoke runs:

```powershell
pip install -r requirements.txt
```

For Colab or Linux GPU runs, install the GPU requirements:

```bash
pip install -r requirements-colab-gpu.txt
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Native Windows does not have supported JAX NVIDIA GPU wheels. Use Colab, Linux,
or WSL2 for GPU experiments. The Colab requirements file intentionally does not
pin or reinstall JAX, because Colab often already provides a CUDA-enabled JAX
build matched to the current runtime. If `jax.devices()` prints `CudaDevice`,
keep that JAX installation. If it prints only `CpuDevice`, install the matching
`jax[cuda12]` or `jax[cuda13]` extra for the active Colab image, restart the
runtime, and verify `jax.devices()` again.

## Reproducibility

The main APIs accept seeds:

```python
from Clustering.jax_clustering_algorithm import devices_generator_jax, clusterizer_jax
from Models.models_arrangement import error_calculator_trace_jax

devices = devices_generator_jax(1000, 300, seed=202406)
clusters = clusterizer_jax(
    devices=devices,
    device_radius=15,
    max_devices_per_cluster=10,
    min_devices_per_cluster=4,
    clustering_mode="geometric",
    strategy="dense",
)
result = error_calculator_trace_jax(
    number_of_mobile_devices__k=1000,
    data_dimension__L=10,
    number_of_parallel_channels__M=10,
    probability_that_user_can_compute_its_local_update__pcomp=0.1,
    max_iterations_t=200,
    learning_rate__u1=0.01,
    step_size__u=0.1,
    clusters=clusters,
    seed=202406,
    checkpoints=[50, 100, 200],
)
```

The default D2D behavior is thesis-compatible: when a CH succeeds, all cluster
members are available to the CH. More realistic member-to-CH behavior can be
enabled with:

```python
error_calculator_trace_jax(
    ...,
    d2d_member_compute_probability=0.8,
    d2d_member_link_success_probability=0.9,
)
```

Run the complete `.py` workflow from local CPU, Colab, or a CUDA-enabled Linux
environment. The root entry point writes CSV results, metadata JSON, and
standard figures:

```bash
python main.py --devices 1000 --rounds 1000 --iterations 200 --checkpoints 50 100 200
```

The default run uses `--clustering-strategy dense`, pair-first local D2D
formation (`--initial-cluster-size 2`), one local singleton join-repair pass,
one local pair CH-rotation repair pass, one local CH-to-CH merge pass,
thesis-style unscaled aggregation, and `--precision float32` for speed. Add
`--precision float64` for thesis
reproduction when optimized ALOHA curves need to go below about `1e-7`. Add
`--clustering-strategy grid` for more conservative large-`K` clustering,
`--repair-passes 0 --rotation-repair-passes 0 --merge-passes 0` to disable
local repair, `--initial-cluster-size 10` to recover the earlier greedy-fill
behavior, or `--normalize-by-k` for the smaller normalized SGD update.
Use `--cluster-head-selection-mode quality` as an enhanced ablation that keeps
cluster membership fixed but elects, inside each cluster, the best valid CH
according to D2D degree, BS channel quality, and battery. A member can become
CH only when it still directly covers all cluster members. This becomes most
meaningful when paired with channel-aware CH-to-BS decoding or energy-aware
uplink ablations, because the selected CH then affects physical delivery or
cost, not only cluster structure.
Enable that channel-aware uplink ablation with
`--d2d-ch-bs-success-mode channel_quality`. In that mode a CH still attempts
ALOHA normally and can still collide, but a collision-free CH packet is decoded
with probability based on the elected CH distance-to-BS and optional battery
factor.
For physically symmetric comparisons, also enable
`--device-bs-success-mode channel_quality`. That applies the same second-stage
decoding model to the non-D2D polling, fixed ALOHA, and optimized ALOHA curves:
direct devices still contend through ALOHA first, and only collision-free
packets draw a device-to-BS decoding success probability. Keep both modes at
`none` when reproducing the original collision-only thesis behavior.

Dynamic energy drain can be enabled as a separate enhanced ablation. The
simulator keeps one battery vector per curve, so polling/fixed/optimized and
their D2D variants do not drain each other's batteries inside the same run.
Direct device-to-BS attempts pay `--energy-direct-bs-cost`; active non-CH D2D
members pay `--energy-d2d-member-cost` when their cluster attempts an
aggregate; CHs pay `--energy-ch-bs-cost` for every CH-to-BS aggregate attempt,
including collided or undecoded attempts. The runner writes mean battery and
mean D2D-CH battery columns, plus energy-used and energy-efficiency columns,
and plots them when present.

Energy-aware D2D CH rotation is another enhanced ablation. It is disabled by
default with `--d2d-ch-rotation-mode static`. Enable it with
`--d2d-ch-rotation-mode energy_aware` together with
`--energy-drain-mode dynamic`. The simulator then periodically re-elects a CH
inside each existing D2D cluster for each D2D curve. The candidate must still
cover all members in one hop, so the cluster membership and `Cmax` constraint
do not change. The selectable profiles are `performance`, `balanced`, and
`eco`, trading BS-channel quality against current battery and a small stability
bonus for keeping the current CH.

Recommended fair channel-aware comparison:

```bash
python main.py --run-name k3000_channel_fair \
  --devices 3000 \
  --rounds 200 \
  --precision float64 \
  --cluster-head-selection-mode quality \
  --cluster-head-degree-weight 0.0 \
  --cluster-head-channel-weight 1.0 \
  --cluster-head-battery-weight 0.0 \
  --d2d-ch-bs-success-mode channel_quality \
  --d2d-ch-bs-min-success-probability 0.35 \
  --d2d-ch-bs-pathloss-exponent 2.0 \
  --d2d-ch-bs-battery-exponent 0.25 \
  --device-bs-success-mode channel_quality \
  --device-bs-min-success-probability 0.35 \
  --device-bs-pathloss-exponent 2.0 \
  --device-bs-battery-exponent 0.25 \
  --optimized-d2d-access-mode utility \
  --optimized-d2d-load-allocation-mode conditional_selective_water_filling \
  --optimized-d2d-redistribution-fraction 0.25 \
  --optimized-d2d-redistribution-trigger-ratio 0.95 \
  --optimized-d2d-density-trigger-threshold 0.95 \
  --optimized-d2d-dense-trigger-ratio 0.0 \
  --optimized-d2d-throughput-ewma-decay 0.90 \
  --optimized-d2d-access-floor-fraction 0.02 \
  --optimized-d2d-norm-exponent 3.5 \
  --optimized-d2d-cluster-size-exponent 1.5 \
  --optimized-d2d-freshness-exponent 0.25 \
  --optimized-d2d-load-target-factor 1.1
```

Recommended short dynamic-energy smoke:

```bash
python main.py --run-name k1000_energy_smoke \
  --devices 1000 \
  --rounds 20 \
  --precision float64 \
  --cluster-head-selection-mode quality \
  --cluster-head-degree-weight 0.0 \
  --cluster-head-channel-weight 1.0 \
  --cluster-head-battery-weight 0.0 \
  --d2d-ch-bs-success-mode channel_quality \
  --d2d-ch-bs-min-success-probability 0.35 \
  --d2d-ch-bs-pathloss-exponent 2.0 \
  --d2d-ch-bs-battery-exponent 0.25 \
  --device-bs-success-mode channel_quality \
  --device-bs-min-success-probability 0.35 \
  --device-bs-pathloss-exponent 2.0 \
  --device-bs-battery-exponent 0.25 \
  --energy-drain-mode dynamic \
  --energy-direct-bs-cost 0.001 \
  --energy-d2d-member-cost 0.0002 \
  --energy-ch-bs-cost 0.002 \
  --optimized-d2d-access-mode utility \
  --optimized-d2d-load-allocation-mode conditional_selective_water_filling \
  --optimized-d2d-redistribution-fraction 0.25 \
  --optimized-d2d-redistribution-trigger-ratio 0.95 \
  --optimized-d2d-density-trigger-threshold 0.95 \
  --optimized-d2d-dense-trigger-ratio 0.0 \
  --optimized-d2d-throughput-ewma-decay 0.90 \
  --optimized-d2d-access-floor-fraction 0.02 \
  --optimized-d2d-norm-exponent 3.5 \
  --optimized-d2d-cluster-size-exponent 1.5 \
  --optimized-d2d-freshness-exponent 0.25 \
  --optimized-d2d-load-target-factor 1.1
```

Recommended short energy-aware CH-rotation smoke:

```bash
python main.py --run-name k1000_energy_rotation_smoke \
  --devices 1000 \
  --rounds 20 \
  --precision float64 \
  --energy-drain-mode dynamic \
  --energy-direct-bs-cost 0.002 \
  --energy-d2d-member-cost 0.0005 \
  --energy-ch-bs-cost 0.005 \
  --d2d-ch-rotation-mode energy_aware \
  --d2d-ch-rotation-interval 10 \
  --d2d-energy-efficiency-level balanced \
  --cluster-head-selection-mode quality \
  --cluster-head-degree-weight 0.0 \
  --cluster-head-channel-weight 1.0 \
  --cluster-head-battery-weight 0.0 \
  --d2d-ch-bs-success-mode channel_quality \
  --d2d-ch-bs-min-success-probability 0.35 \
  --d2d-ch-bs-pathloss-exponent 2.0 \
  --d2d-ch-bs-battery-exponent 0.25 \
  --device-bs-success-mode channel_quality \
  --device-bs-min-success-probability 0.35 \
  --device-bs-pathloss-exponent 2.0 \
  --device-bs-battery-exponent 0.25 \
  --optimized-d2d-access-mode utility \
  --optimized-d2d-load-allocation-mode conditional_selective_water_filling \
  --optimized-d2d-redistribution-fraction 0.25 \
  --optimized-d2d-redistribution-trigger-ratio 0.95 \
  --optimized-d2d-density-trigger-threshold 0.95 \
  --optimized-d2d-dense-trigger-ratio 0.0 \
  --optimized-d2d-throughput-ewma-decay 0.90 \
  --optimized-d2d-access-floor-fraction 0.02 \
  --optimized-d2d-norm-exponent 3.5 \
  --optimized-d2d-cluster-size-exponent 1.5 \
  --optimized-d2d-freshness-exponent 0.25 \
  --optimized-d2d-load-target-factor 1.1
```

To compare all CH-rotation energy profiles with the same controlled setup, use
the dedicated sweep runner:

```bash
python -m experiments.run_energy_rotation_sweep \
  --run-name k1000_energy_rotation_profiles_r100 \
  --devices 1000 \
  --rounds 100
```

That runner evaluates `static`, `performance`, `balanced`, and `eco`, writes one
subfolder per profile, and creates `energy_rotation_summary.csv`,
`energy_rotation_summary.md`, `energy_rotation_optimized_d2d_error_norm.*`, and
`energy_rotation_tradeoff.*`.  Its ranking prioritizes profiles that keep final
optimized-D2D error and energy close to the static baseline while preserving
more CH battery.

When optimized ALOHA with D2D underuses the channel after fast convergence,
enable the guarded optimized-D2D access floor:

```bash
python main.py --run-name k3000_x64_pair_merge_opt_floor --devices 3000 --precision float64 --optimized-d2d-access-floor-fraction 1.0
```

That keeps the optimized-D2D access probability at least as large as the fixed
D2D ALOHA baseline while still allowing norm-based priority above the floor.

For a stronger optimized-D2D ablation, use the utility mode. It keeps the
expected CH contender load near the number of channels, but redistributes access
toward CHs with larger aggregate updates, larger active aggregates, and longer
freshness:

```bash
python main.py --run-name k3000_x64_pair_merge_utility_opt --devices 3000 --precision float64 --optimized-d2d-access-mode utility --optimized-d2d-access-floor-fraction 0.25 --optimized-d2d-norm-exponent 1.0 --optimized-d2d-cluster-size-exponent 1.0 --optimized-d2d-freshness-exponent 0.5
```

Use `--optimized-d2d-access-floor-fraction 0.25` or `0.5` for utility mode.
Using `1.0` makes the mode nearly identical to fixed D2D because the full load
budget is already assigned to the baseline floor.
Use `--optimized-d2d-load-target-factor` to tune the expected CH contender
target for utility mode: `1.0` targets `M`, `0.8` targets `0.8M`, and `1.2`
targets `1.2M`.
Load-controlled enhanced modes use `--optimized-d2d-load-allocation-mode
conditional_selective_water_filling` by default. This starts from the old
proportional clipped allocator, checks whether the EWMA of successful
optimized-D2D CH uploads is below `--optimized-d2d-redistribution-trigger-ratio`
of the expected fixed-D2D CH throughput, and only then redistributes
`--optimized-d2d-redistribution-fraction` of the missing load toward the
configured load target. If the throughput trigger is not active, it returns
exactly the legacy `proportional_clip` probabilities. This uses ACK-observable
throughput rather than true model error or raw attempted contenders.
The conditional allocator is also density-aware by default: when the
clusterized-device fraction is at least
`--optimized-d2d-density-trigger-threshold` the trigger ratio is replaced by
`--optimized-d2d-dense-trigger-ratio`. The current defaults use `0.95` as the
dense-regime cutoff and `0.90` as the dense trigger. This was added because the
larger `K=3000` runs showed that highly clusterized deployments can be hurt by
extra redistribution: the channel receives more contenders, but the additional
successful CH uploads do not necessarily carry enough new information to offset
the collision risk. The rule is still deployable because the BS only needs the
clusterization summary and ACK-observed CH throughput.
Use `proportional_clip` to reproduce the older enhanced-mode allocator exactly,
`selective_water_filling` to always redistribute a partial amount, and
`water_filling` to force the full target load when testing high-usage ablations.
For the full strategy history, observed results, and a deployment-oriented
control-plane discussion, see `docs/optimized_d2d_strategy_report.md`.

The more selective optimized-D2D ablation is `max_weight`. It keeps the same
utility terms, but maps them through an adaptive threshold. This concentrates
access on high-value CH aggregates while the threshold controller tries to keep
the CH contention load near `M` channels:

```bash
python main.py --run-name k3000_x64_max_weight --devices 3000 --precision float64 --optimized-d2d-access-mode max_weight --optimized-d2d-access-floor-fraction 0.10 --optimized-d2d-norm-exponent 2.0 --optimized-d2d-cluster-size-exponent 1.0 --optimized-d2d-freshness-exponent 1.0 --optimized-d2d-threshold-gain 8.0
```

This is an enhanced strategy, not thesis-exact behavior. It is intended for
testing whether optimized D2D can beat fixed D2D by choosing better CH
transmissions, rather than by simply increasing BS contention.

The hybrid optimized-D2D ablation keeps the smoother utility allocation but
adds a directional novelty term. The BS maintains a recent successful
optimized-D2D update direction and can broadcast it with the FL model; each CH
discounts aggregates that are mostly aligned with that recent direction:

```bash
python main.py --run-name k3000_x64_hybrid_b2_floor010 --devices 3000 --precision float64 --optimized-d2d-access-mode hybrid --optimized-d2d-access-floor-fraction 0.10 --optimized-d2d-norm-exponent 2.0 --optimized-d2d-cluster-size-exponent 1.0 --optimized-d2d-freshness-exponent 1.0 --optimized-d2d-novelty-exponent 1.0 --optimized-d2d-novelty-floor 0.25 --optimized-d2d-reference-decay 0.90
```

This tests whether optimized D2D improves by preserving update diversity, not
by hard-thresholding access or increasing the expected CH load.

The adaptive-diversity optimized-D2D ablation is the next algorithmic test. It
keeps the same load-controlled CH contention target as `utility`/`hybrid`, but
uses a two-phase score. Early rounds favor large useful aggregates so the model
moves quickly; later rounds reduce the raw-norm emphasis and add novelty plus
freshness so the BS receives less redundant cluster directions. The phase is a
sigmoid of `t / max_t`, not of the true error, because real deployments do not
know the true optimization error:

```bash
python main.py --run-name k3000_adaptive_diversity \
  --devices 3000 \
  --rounds 100 \
  --precision float64 \
  --optimized-d2d-access-mode adaptive_diversity \
  --optimized-d2d-access-floor-fraction 0.02 \
  --optimized-d2d-norm-exponent 3.5 \
  --optimized-d2d-cluster-size-exponent 1.5 \
  --optimized-d2d-freshness-exponent 0.25 \
  --optimized-d2d-late-norm-exponent 1.25 \
  --optimized-d2d-late-freshness-exponent 1.0 \
  --optimized-d2d-novelty-exponent 1.5 \
  --optimized-d2d-novelty-floor 0.25 \
  --optimized-d2d-reference-decay 0.90 \
  --optimized-d2d-load-target-factor 1.1 \
  --optimized-d2d-load-allocation-mode conditional_selective_water_filling \
  --optimized-d2d-redistribution-fraction 0.25 \
  --optimized-d2d-redistribution-trigger-ratio 0.95 \
  --optimized-d2d-density-trigger-threshold 0.95 \
  --optimized-d2d-dense-trigger-ratio 0.90 \
  --optimized-d2d-throughput-ewma-decay 0.90 \
  --optimized-d2d-adaptive-switch-fraction 0.30 \
  --optimized-d2d-adaptive-switch-gain 12.0
```

This is not thesis-exact behavior. It is a scientifically motivated ablation
for testing whether a deployable CH-level policy can beat pure utility tuning:
local CH signals are aggregate norm, active aggregate size, freshness, and
aggregate direction; the BS can broadcast only scalar normalizers, the phase
value, and the recent optimized-D2D reference direction.

To tune the utility policy as a Pareto problem, use the utility sweep runner.
It executes a grid of floors, exponents, and load-target factors, then writes
per-candidate results plus a ranked summary:

```bash
python -m experiments.run_utility_pareto_sweep --run-name utility_pareto_smoke --devices 1000 --rounds 20 --precision float64 --max-candidates 5
```

For the full refined `K = 3000` grid:

```bash
python -m experiments.run_utility_pareto_sweep --run-name utility_pareto_k3000 --devices 3000 --rounds 100 --precision float64
```

The default refined grid has 162 candidates and focuses on the neighborhood
that performed best in the first full K=3000 sweep: lower access floor, higher
norm exponent, higher cluster-size exponent, and lower freshness exponent. The
older 243-candidate grid is still available with `--candidate-grid coarse`.
New sweeps use conditional selective water-filling allocation by default; add
`--optimized-d2d-load-allocation-mode proportional_clip` when reproducing older
utility-sweep results, or `selective_water_filling` plus
`--optimized-d2d-redistribution-fraction` when forcing partial redistribution
even when the proportional clipped load is already near the target.
Each candidate writes
`Runs/<run-name>/<candidate>/results.csv`, and the parent folder writes
`utility_sweep_summary.csv`, `utility_sweep_top10.md`, and
`utility_sweep_pareto.png`/`.pdf` plus
`utility_sweep_target_time.png`/`.pdf`. The summary includes
`t_to_1e_minus_6`, `t_to_1e_minus_9`, and `t_to_1e_minus_12` because final
error near `1e-16` is often dominated by float64 numerical saturation.

On free Colab, run the grid in smaller slices to avoid runtime disconnects:

```bash
python -m experiments.run_utility_pareto_sweep --run-name utility_pareto_k3000_part01 --devices 3000 --rounds 100 --precision float64 --candidate-start 0 --candidate-count 30
python -m experiments.run_utility_pareto_sweep --run-name utility_pareto_k3000_part02 --devices 3000 --rounds 100 --precision float64 --candidate-start 30 --candidate-count 30
python -m experiments.run_utility_pareto_sweep --run-name utility_pareto_k3000_part03 --devices 3000 --rounds 100 --precision float64 --candidate-start 60 --candidate-count 30
```

For the refined grid, continue with starts `90`, `120`, and `150` to cover all
162 candidates. For the old coarse grid, use starts `0`, `30`, `60`, `90`,
`120`, `150`, `180`, `210`, and `240`. The last slice automatically contains
only the remaining candidates. Each slice prints the selected zero-based
interval, for example `candidate slice 30:60 of 162`, and the summary CSV
includes `candidate_grid_index` so partial runs can be audited before merging.

Merge completed parts after downloading or keeping them in the same runtime:

```bash
python -m experiments.merge_utility_pareto_summaries Runs/utility_pareto_k3000_part* --output-dir Runs/utility_pareto_k3000_merged
```

To generate a thesis Figure 15-style run with the full `t = 1..200` curve,
omit `--checkpoints`:

```bash
python main.py --run-name thesis_figure_15_x64 --precision float64
```

Use `--checkpoints 1 100 200` only when you intentionally want a lighter
start/middle/end CSV and figure.

By default the runner creates a fresh timestamped folder only after the
simulation finishes successfully:

- `Runs/YYYY-MM-DD-HH-MM-SS/results.csv`;
- `Runs/YYYY-MM-DD-HH-MM-SS/results.metadata.json`;
- `Runs/YYYY-MM-DD-HH-MM-SS/results_error_norm.png` and `.pdf`;
- `Runs/YYYY-MM-DD-HH-MM-SS/results_figure_15_error_norm.png` and `.pdf`;
- `Runs/YYYY-MM-DD-HH-MM-SS/results_uploads.png` and `.pdf`;
- `Runs/YYYY-MM-DD-HH-MM-SS/results_clusterhead_uploads.png` and `.pdf`;
- `Runs/YYYY-MM-DD-HH-MM-SS/results_cluster_rate.png` and `.pdf`;
- `Runs/YYYY-MM-DD-HH-MM-SS/results_cluster_quality.png` and `.pdf`.

Use a stable folder name when you want a recognizable run:

```bash
python main.py --run-name local_smoke --devices 100 --rounds 5 --iterations 20 --checkpoints 5 10 20
```

If the folder already exists, the runner appends a numeric suffix such as
`local_smoke-02` instead of overwriting it. Use `--output Runs/some_file.csv`
only when you explicitly want the older single-file layout.

You can also regenerate figures from an existing CSV without rerunning the
simulation:

```bash
python -m experiments.plot_gpu_sweep Runs/local_smoke/results.csv
```

Notebooks live under `notebooks/` and are optional analysis frontends. They
should read the CSV/metadata and figure files generated by the `.py` workflow
rather than containing the required simulation path.

## Validation

Run the automated tests with:

```bash
python -m unittest discover
```

The clustering tests verify device uniqueness, one-hop CH distance, maximum
cluster size, deterministic seeds, and the merge overflow case. The JAX tests
cover padded cluster preparation, one-hop GPU cluster validity, singleton
accounting, finite traces, and GPU sweep row generation. Tests that require JAX
skip cleanly when JAX is not installed in the local interpreter.

## Known Limitations

- The clustering heuristic is a centralized simulator of the proposed distributed
  protocol. It does not model every message exchange explicitly.
- D2D member-to-CH collisions, packet loss, delay, and energy cost are optional
  abstractions, not a full link-layer simulator.
- Dynamic battery drain is optional and coarse-grained. It models normalized
  per-attempt energy costs, not a calibrated radio power model, recharge model,
  thermal effect, or time-varying battery voltage.
- The optimized ALOHA model uses aggregate CH update norms by default, matching
  the thesis model. Mean-normalized or utility-weighted CH access is an ablation
  candidate, not the default implementation.
- The JAX dense/grid clusterizers are statistically validated against the thesis
  idea, not byte-identical to the earlier CPU ordering. Compare curves,
  confidence intervals, upload counts, and clustering-rate distributions rather
  than exact per-seed values.
- Very large `K` values are bounded by accelerator memory. Increase `K`,
  `rounds`, and batch size gradually and record the metadata JSON produced by
  the sweep runner.
