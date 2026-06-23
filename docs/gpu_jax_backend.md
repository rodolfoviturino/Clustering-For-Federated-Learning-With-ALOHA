# GPU JAX Backend

This document describes the JAX-only accelerated backend used for large
experiments. The older CPU clustering code remains useful as a readable
description of the original D2D-SRC idea, but it is no longer the runtime
backend for new sweeps.

## Why JAX

JAX is used because the experiment has three performance requirements that fit
JAX well:

1. Many independent Monte Carlo rounds can be batched with `jax.vmap`.
2. The FL time loop can run as one compiled loop with `jax.lax.scan`.
3. Fixed-shape arrays can stay on GPU across device generation, clustering, and
   model simulation.

The official installation guide lists `pip install -U "jax[cuda13]"` for
NVIDIA CUDA 13, `jax[cuda12]` as the CUDA 12 alternative, and `jax[tpu]` for
Google Cloud TPU VMs. It also notes that NVIDIA GPU wheels are available for
Linux, while native Windows has CPU support but no NVIDIA GPU support:
https://docs.jax.dev/en/latest/installation.html

For that reason this repository uses two requirement files:

- `requirements.txt`: local development, Windows CPU, and small smoke tests.
- `requirements-colab-gpu.txt`: Colab/Linux GPU helper dependencies. It
  intentionally does not pin or reinstall JAX, because Colab often ships with a
  CUDA-enabled JAX build already matched to the current driver/runtime image.

TPU execution is not the first target. The simulation has random access,
branching, and sparse cluster membership, which are easier to tune first on GPU.
Once the GPU path is statistically validated, TPU experiments can reuse the same
fixed-shape arrays with additional profiling.

## Main Modules

`main.py`

- Root command-line entry point for the full `.py` workflow.
- Delegates to `experiments.run_gpu_sweep`.
- Use it when running experiments from the repository root.

`Clustering/jax_clustering_algorithm.py`

- Generates device arrays with `devices_generator_jax`.
- Builds padded one-hop cluster arrays with `clusterizer_jax`.
- Provides validation helpers for one-hop coverage, `Cmax`, singleton count, and
  overflow diagnostics.

`Models/jax_models_arrangement.py`

- Runs the HFL/ALOHA simulation with `error_calculator_trace_jax`.
- Uses `jax.lax.scan` to advance one trajectory from `t = 1` to `max_t`.
- Returns metrics at all requested checkpoints without rerunning earlier
  iterations.
- Can keep the thesis-compatible collision-only uplink model, or apply
  optional channel-aware decoding to direct device-to-BS uploads and D2D
  CH-to-BS uploads after ALOHA collision resolution.

`experiments/run_gpu_sweep.py`

- Creates one compiled batched program with `jax.jit(jax.vmap(...))`.
- Aggregates mean values and 95 percent confidence interval half-widths.
- Creates a fresh `Runs/YYYY-MM-DD-HH-MM-SS/` folder after the simulation
  completes successfully, unless `--output` is used for a direct CSV path.
- Writes run metadata, including backend, JAX version, device type, seed policy,
  dimensions, clustering mode, output paths, and elapsed time.
- Generates standard figures from the output CSV unless `--no-plots` is passed.

`experiments/plot_gpu_sweep.py`

- Converts a saved sweep CSV into figures without rerunning the simulation.
- Uses a non-interactive Matplotlib backend, so it works in terminals, CI,
  Windows shells, and Colab.
- Produces error norm, device upload, cluster-head upload, and clustering-rate
  plots.

## Device Arrays

`devices_generator_jax(K, R_BS, seed)` returns a `JaxDeviceBatch`.

Array contracts:

- `device_ids`: `int32[K]`. Stable IDs, normally `0..K-1`.
- `coords`: `float32/float64[K, 2]`. Device `(x, y)` coordinates in meters.
- `device_angle`: `float32/float64[K]`. Polar angle in radians.
- `distance_to_bs`: `float32/float64[K]`. Distance from each device to the BS in meters.
- `battery`: `int32[K]`. Battery percentage in `[1, 100]`.
- `stats_product`: `float32/float64[K]`. Product `distance_to_bs * battery`,
  kept for compatibility with the thesis-style descriptive statistics.
- `bs_radius`: scalar `float32/float64`. BS coverage radius in meters.

The default radius model is thesis-compatible:

```text
r ~ U(1, R_BS)
theta ~ U(0, 2*pi)
x = r cos(theta)
y = r sin(theta)
```

Set `uniform_area=True` only for an explicit disk-uniform ablation:

```text
r = sqrt(U(1, R_BS^2))
```

## Cluster Arrays

`clusterizer_jax(...)` returns a `JaxClusterResult`.

Array contracts:

- `cluster_members`: `int32[K, Cmax]`. Padded cluster rows. Unused entries are
  `-1`.
- `cluster_sizes`: `int32[K]`. Number of valid devices in each row.
- `cluster_heads`: `int32[K]`. The selected CH for each row, or `-1` for empty
  rows.
- `cluster_mask`: `bool[K]`. True only for active cluster rows.
- `number_of_clusters`: scalar `int32`. Count of active cluster rows.
- `clusterized_devices_rate`: scalar `float32/float64`. Percentage of devices
  in non-singleton clusters.
- `singleton_count`: scalar `int32`. Number of one-device clusters.
- `overflow_count`: scalar `int32`. Number of rows split because a grid cell had
  more than `Cmax` devices.
- `mode_code` and `strategy_code`: scalar `int32` metadata for saved runs.

Column 0 of every active row is the cluster head. For example, if one row is
`[8, 21, 32, -1, -1]`, device 8 is the CH and devices 21 and 32 are one-hop
members.

## GPU-Native Clustering

The default production strategy is `strategy="dense"` because it is closest to
the original thesis experiment at `K=1000`.

Dense strategy:

- ranks candidate CHs by the selected clustering mode;
- in `geometric` mode, ranks CHs by one-hop degree and then device ID, using
  only the D2D radius graph;
- forms initial local pairs by default (`--initial-cluster-size 2`), matching
  the D2D-SRC pair-formation stage;
- then lets CHs absorb reachable singletons through local repair, up to `Cmax`;
- runs local singleton repair passes after the greedy phase: a singleton can
  join only a CH it reaches directly and only if that CH has spare capacity;
- runs local CH-rotation repair for two-device clusters: if a singleton reaches
  the member but not the current CH, the member may become CH and admit it;
- reruns singleton join repair after CH rotations so newly promoted CHs can
  accept additional reachable singletons;
- runs local CH-to-CH merge repair after singleton/rotation repair: two
  non-singleton clusters may merge only when the target CH can cover the whole
  union and the merged cluster remains within `Cmax`;
- can optionally run quality CH election with
  `--cluster-head-selection-mode quality`: cluster membership stays fixed, but
  the CH position rotates to the highest-scoring member that can still directly
  cover all members. The score combines normalized D2D degree, normalized BS
  channel quality, and battery;
- preserves one-hop CH coverage directly from the radius test;
- is more expensive than grid clustering, but gives a clustering-rate behavior
  much closer to the old D2D-SRC notebook.

The scalable large-`K` option is `strategy="grid"`.

The grid side length is:

```text
cell_side = R_D2D / sqrt(2)
```

Any two points inside the same square cell are at most:

```text
sqrt(cell_side^2 + cell_side^2) = R_D2D
```

This gives a direct one-hop guarantee from the CH to every member when the CH is
chosen from inside the cell. The method avoids a full `K x K` neighbor matrix,
which would be the main memory bottleneck for very large `K`.

Dense cells are split into balanced chunks, so a cell with 12 devices and
`Cmax = 10` becomes `6 + 6` rather than `10 + 2`. If a whole cell has size in
`2..Cmin-1`, it falls back to singleton rows. That conservative fallback keeps
the clustered-device rate from counting weak tiny groups as successful D2D
clusters.

Modes:

- `geometric`: ranks devices by one-hop degree, with device ID as the
  deterministic tie-breaker. This keeps the mode geometry-only while usually
  reducing singleton clusters compared with arbitrary ID ordering.
- `utility`: ranks devices by degree, battery, normalized BS channel quality,
  and ID. Degree is computed with tiled distance comparisons to avoid materializing
  a full all-pairs tensor.
- `no_d2d`: returns singleton clusters for the baseline comparison.

`neighbor_counts_tiled_jax(coords, R_D2D, tile_size)` is available for profiling
and utility scoring. It compares all devices against fixed-size candidate tiles
on the accelerator.

The dense repair pass is intentionally distributed-style.  It is equivalent to
nearby singletons broadcasting join requests and reachable CHs with spare
capacity accepting requests. It does not require the BS to solve a global
assignment problem.

The pair-rotation repair is also local: it is limited to two-device clusters
where the promoted member can directly reach both the old CH and the singleton.
No global reassignment is used.

The CH-to-CH merge repair is a cluster-quality pass, not a clustering-rate
shortcut. It does not turn singletons into clustered devices by itself. Instead,
it consolidates nearby valid D2D clusters when one existing CH can cover the
combined member set. The practical interpretation is that neighboring CHs
exchange compact cluster summaries and accept a merge only when the target CH
can serve every member in one hop. This reduces the number of CH rows competing
on the BS uplink and increases the average aggregate size seen by HFL.

Use `--initial-cluster-size Cmax` to recover the earlier greedy-fill behavior.
The default `2` prioritizes covering more devices with at least one D2D partner
before growing clusters, which is closer to the original D2D-SRC sequence.

Quality CH election is a separate ablation from singleton/merge repair. Repair
changes which devices belong to each cluster; quality CH election changes only
which member represents an already formed cluster on the BS uplink. It is
plausible as an intra-cluster control step: members exchange local degree,
battery, and BS reference-signal quality, then elect the best candidate that
preserves one-hop coverage. It does not require the BS to assign CHs globally.
In the thesis-compatible collision-only ALOHA model, this is mainly a
structural uplink-quality ablation: because the member set is unchanged,
systematic curve gains require a physical model where CH-to-BS success, energy
cost, or retry behavior depends on the elected CH's BS channel and battery.

The channel-aware decoding ablation is available as
`--d2d-ch-bs-success-mode channel_quality`.  In this mode, an attempted CH still
enters the same multichannel ALOHA contention process.  If the CH avoids
collision, the BS decodes the aggregate with probability derived from normalized
inverse pathloss and optional battery weighting.  This makes elected-CH
identity matter while preserving distributed ALOHA decisions.

Use `--device-bs-success-mode channel_quality` as the direct device-to-BS
counterpart for non-D2D curves.  This keeps D2D and non-D2D comparisons under
the same channel-quality abstraction: first ALOHA contention, then physical
decoding for collision-free packets.

SciPy `cKDTree` is not used in the GPU backend. It is CPU-side and remains a
useful validation/profiling reference only:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

## FL And ALOHA Trace

`error_calculator_trace_jax(...)` simulates a synthetic linear-regression FL
task.

Data model:

```text
X_i in R^L
w_true in R^L
y_i = X_i dot w_true
```

At each round, every device has a local gradient:

```text
grad_i(w) = X_i * (X_i dot w - y_i)
```

The BS update uses the thesis figure's unscaled SGD step by default:

```text
w <- w - u1 * sum(successful_gradients)
```

Use `--normalize-by-k` or `normalize_by_k=True` for the conservative normalized
ablation:

```text
w <- w - u1 * sum(successful_gradients) / K
```

Use `--precision float64` for thesis reproduction when optimized ALOHA reaches
very small error norms. `float32` is the speed default for GPU sweeps, but it
usually floors norms around single-precision accuracy.

Optimized ALOHA can optionally apply an access floor:

```text
p_opt_guarded = max(p_opt_norm_based, floor_fraction * p_fixed)
```

For D2D, `p_fixed` is the fixed D2D ALOHA access probability
`min(M / number_of_clusterheads, pcomp)`. The recommended stress test for the
merge-enhanced clustering path is `--optimized-d2d-access-floor-fraction 1.0`.
This prevents optimized D2D from starving the BS channels after aggregate norms
become very small, while still letting CHs with larger aggregate norms use
higher access probabilities. The thesis-exact behavior is preserved by the
default floor fraction `0.0`.

The enhanced optimized-D2D utility mode is selected with
`--optimized-d2d-access-mode utility`. It uses:

```text
utility_h =
  norm_h^beta * active_cluster_size_h^delta * freshness_h^gamma
```

and then allocates access probabilities so the expected number of CH contenders
stays close to the channel count `M`. The access floor reserves a fraction of
the fixed-D2D load before the remaining probability mass is distributed by
utility. For this reason, a floor of `1.0` intentionally collapses the utility
mode toward fixed D2D; useful exploratory values are usually `0.25` and `0.5`.
`--optimized-d2d-load-target-factor` adjusts the same load target for
load-controlled policies: `1.0` targets `M`, `0.8` targets `0.8M`, and `1.2`
targets `1.2M`.

Load-controlled policies can choose how the target probability mass is assigned
with `--optimized-d2d-load-allocation-mode`. The base water-filling allocator
keeps the floor, computes each CH's remaining capacity `pcomp - floor`, and
finds a scalar water level `lambda` such that:

```text
p_h = floor_h + min(pcomp - floor_h, lambda * utility_h)
```

This replaces the older `proportional_clip` behavior, where clipped probability
mass disappeared whenever a high-utility CH hit `pcomp`. Water-filling is still
not centralized scheduling: the BS can broadcast the scalar water level or
equivalent normalizer, while every CH keeps making its own ALOHA decision.
Use `proportional_clip` only when reproducing older enhanced-mode runs or when
the clipped load already matches the desired channel contention.

`selective_water_filling` is the intermediate ablation. It first computes the
older clipped load, then redistributes only:

```text
redistributed_load =
  redistribution_fraction * max(target_load - proportional_clip_load, 0)
```

`--optimized-d2d-redistribution-fraction 0.0` matches `proportional_clip`;
`1.0` targets the same total load as water-filling. Values such as `0.25`,
`0.50`, and `0.75` test whether some extra CH usage improves convergence
without activating too many low-marginal-value CHs.

The default enhanced allocator is `conditional_selective_water_filling`. It
uses the same partial redistribution only when observed optimized-D2D CH
throughput is below the expected fixed-D2D CH throughput:

```text
target_load = min(M * load_target_factor, pcomp * active_clusterhead_count)

fixed_success_target =
  active_clusterhead_count *
  fixed_access_probability *
  (1 - fixed_access_probability / M)^(active_clusterhead_count - 1)

optimized_success_ewma[t] =
  decay * optimized_success_ewma[t - 1]
  + (1 - decay) * successful_optimized_d2d_ch_uploads[t - 1]

throughput_ratio = optimized_success_ewma / fixed_success_target

clusterized_fraction = clusterized_devices_rate / 100

effective_trigger_ratio =
  dense_trigger_ratio
    if clusterized_fraction >= density_trigger_threshold
  else redistribution_trigger_ratio

if throughput_ratio < effective_trigger_ratio:
  target_load_conditional =
    proportional_clip_load
    + redistribution_fraction * max(target_load - proportional_clip_load, 0)
  p_h = floor_h + min(pcomp - floor_h, lambda * utility_h)
else:
  p_h = proportional_clip_probability_h
```

The trigger is based on ACK-observable successful CH uploads, not on attempted
contenders, the inflated exploratory target, or the true model error. The
additional density gate uses the clusterization summary produced during D2D
formation. Dense deployments use a lower trigger by default because extra
redistribution can add collision pressure after almost all devices already have
one-hop D2D coverage. This avoids forcing extra redistribution when
proportional clipping already delivers near-fixed-D2D useful throughput. It
remains plausible in a deployed wireless FL system: the BS can estimate the
EWMA from successful CH uploads, know or estimate the clusterized-device
fraction from cluster formation, and broadcast the target, effective trigger
ratio, and water-level normalizer, while each CH uses its local utility score
and a local random ALOHA draw. No individual CH is centrally selected or forced
to transmit.

The EWMA is initialized at `fixed_success_target`. This prevents every run from
starting with a forced redistribution burst before the BS has any ACK history.
When the trigger is off, the allocator returns the legacy proportional clipped
probabilities exactly; it does not run water-filling with an equivalent total
load, because that would still reshuffle CH access probabilities.

The enhanced max-weight mode is selected with
`--optimized-d2d-access-mode max_weight`. It uses the same utility expression,
but maps the utility through:

```text
p_h = floor + (pcomp - floor) * sigmoid(gain * (utility_h - threshold))
```

The threshold is a scalar dual variable updated from the previous CH contender
count. This keeps the policy compatible with a realistic control channel: the
BS broadcasts the threshold and normalizing constants, while each CH computes
its own access probability locally. Larger `--optimized-d2d-threshold-gain`
values make the policy closer to hard max-weight scheduling; smaller values
make it closer to the smooth utility mode.

The hybrid mode is selected with `--optimized-d2d-access-mode hybrid`. It keeps
the smooth utility allocation, but multiplies the utility by a directional
novelty term:

```text
hybrid_utility_h = utility_h * novelty_h^eta
```

`novelty_h` is the fraction of the CH aggregate direction that is not aligned
with a recent successful optimized-D2D reference direction, with
`--optimized-d2d-novelty-floor` preserving some credit for aligned but still
large aggregates. The reference is updated by exponential decay using
`--optimized-d2d-reference-decay`. This is still a realistic distributed
control signal: the BS broadcasts the reference direction, while each CH
computes its own novelty score from its local aggregate update.

The adaptive-diversity mode is selected with
`--optimized-d2d-access-mode adaptive_diversity`. It is deliberately more than
a hyperparameter retune of `utility`: it changes the CH priority equation over
the course of the run while keeping the same load-controlled ALOHA conversion.
The motivation is consistent with three common ideas in the literature:
utility-guided participant selection in FL ([Oort](https://arxiv.org/abs/2010.06081)),
adaptive wireless scheduling ([MAB client scheduling](https://arxiv.org/abs/2007.02315)),
and freshness/Age-of-Information when stale information loses value
([WiFresh/AoI](https://arxiv.org/abs/2012.14337)).

```text
phase(t) = sigmoid(gain * ((t / max_t) - switch_fraction))

early_utility_h =
  norm_h^early_norm *
  active_cluster_size_h^size_exp *
  freshness_h^early_freshness

late_utility_h =
  norm_h^late_norm *
  active_cluster_size_h^size_exp *
  freshness_h^late_freshness *
  novelty_h^novelty_exp

adaptive_utility_h =
  (1 - phase(t)) * early_utility_h + phase(t) * late_utility_h
```

The early phase prioritizes high-norm and large active aggregates because the
model is far from convergence and a large useful aggregate can move the global
state quickly. The late phase lowers the norm exponent and introduces novelty
plus stronger freshness pressure because repeatedly uploading aligned CH
directions can waste CH contention once the main error has already fallen.
`phase(t)` depends on `t / max_t`, not on the true error norm, because the true
error is unavailable in a real deployment; the BS can know the planned horizon
and broadcast the scalar phase with the FL model.

The required signals preserve the CH-level plausibility of the experiment:

- local at the CH: aggregate update norm, aggregate direction, active aggregate
  size, and freshness since the CH last uploaded;
- broadcast or slowly updated by the BS: normalizers, the recent successful
  optimized-D2D reference direction, `switch_fraction`, `switch_gain`, and the
  current phase scalar.

Policy differences:

- `norm` is the thesis-compatible optimized-D2D controller based on aggregate
  norm and the original dual variable.
- `utility` keeps expected CH contenders near `M` and redistributes access by
  norm, active size, and freshness.
- `max_weight` maps the utility score through a threshold gate and is more
  selective, but can become sensitive to threshold dynamics.
- `hybrid` keeps smooth utility load control and multiplies the score by
  directional novelty.
- `adaptive_diversity` starts from aggressive utility and gradually shifts
  toward the hybrid diversity/freshness objective using time-normalized phase.

The utility Pareto tuning runner is:

```bash
python -m experiments.run_utility_pareto_sweep \
  --run-name utility_pareto_k3000 \
  --devices 3000 \
  --rounds 100 \
  --precision float64
```

It runs the refined 162-candidate utility grid by default and ranks candidates
by time to reach optimized-D2D error targets `1e-6`, `1e-9`, and `1e-12`,
constrained to a final optimized/fixed D2D CH-upload ratio in `[0.95, 1.05]`.
The older 243-candidate sweep remains available with `--candidate-grid coarse`.
For local smoke tests, add `--max-candidates 5`. The runner writes both the
original AUC-vs-CH-ratio Pareto plot and a target-time-vs-CH-ratio Pareto plot.

Free Colab sessions may not stay connected long enough for every candidate.
Use zero-based slices:

```bash
python -m experiments.run_utility_pareto_sweep \
  --run-name utility_pareto_k3000_part01 \
  --devices 3000 \
  --rounds 100 \
  --precision float64 \
  --candidate-start 0 \
  --candidate-count 30
```

For 30-candidate chunks on the refined grid, run starts `0`, `30`, `60`, `90`,
`120`, and `150`. For the coarse grid, also use `180`, `210`, and `240`. The
final chunk contains only the remaining candidates. The runner prints the
selected interval and writes `candidate_grid_index` to
`utility_sweep_summary.csv`, so the partial output can be checked before
merging.

Then merge finished parts:

```bash
python -m experiments.merge_utility_pareto_summaries \
  Runs/utility_pareto_k3000_part* \
  --output-dir Runs/utility_pareto_k3000_merged
```

Six scenario columns are returned:

0. Polling without D2D.
1. Fixed ALOHA without D2D.
2. Optimized ALOHA without D2D.
3. Polling with D2D.
4. Fixed ALOHA with D2D.
5. Optimized ALOHA with D2D.

The returned `JaxTraceResult` contains:

- `clusterized_devices_rate`: scalar clustering percentage.
- `error_norms`: `float[checkpoints, 6]`.
- `successful_uploads`: `float[checkpoints, 6]`.
- `successful_clusterhead_uploads`: `float[checkpoints, 3]`.
- `mean_battery`: `float[checkpoints, 6]`.
- `mean_clusterhead_battery`: `float[checkpoints, 3]`.
- `mean_energy_used`: `float[checkpoints, 6]`.
- `energy_efficiency`: `float[checkpoints, 6]`.
- `mean_clusterhead_energy_used`: `float[checkpoints, 3]`.
- `checkpoints`: `int32[checkpoints]`.

Cluster quality scalars such as CH row count, singleton count,
non-singleton D2D cluster count, clustered-device count, mean cluster size, and
mean non-singleton cluster size are added by `experiments.run_gpu_sweep` to the
CSV/metadata layer.

## Seed Policy

The sweep runner uses:

```text
round_seed = seed_start + round_index
```

Each round uses its seed for device generation, clustering choices, synthetic FL
data, local compute events, channel selection, and member-to-CH success events.
Exact per-seed equality with the older CPU notebook is not expected. The
scientific requirement is statistical consistency under a documented seed policy.

## Colab GPU Smoke Test

In a Colab GPU runtime:

```bash
pip install -r requirements-colab-gpu.txt
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

If that command prints `CudaDevice`, keep the existing JAX installation. If it
prints only `CpuDevice`, install the JAX CUDA extra that matches the active
Colab image, restart the runtime, and verify again:

```bash
pip install -U "jax[cuda12]"
# or, on CUDA 13 Colab images:
pip install -U "jax[cuda13]"
```

Then run a small sweep:

```bash
python -m experiments.run_gpu_sweep \
  --devices 1000 \
  --rounds 10 \
  --iterations 20 \
  --checkpoints 5 10 20 \
  --clustering-mode geometric \
  --clustering-strategy dense \
  --run-name gpu_smoke
```

The equivalent root command is:

```bash
python main.py \
  --devices 1000 \
  --rounds 10 \
  --iterations 20 \
  --checkpoints 5 10 20 \
  --clustering-mode geometric \
  --clustering-strategy dense \
  --run-name gpu_smoke
```

That command writes:

- `Runs/gpu_smoke/results.csv`;
- `Runs/gpu_smoke/results.metadata.json`;
- `Runs/gpu_smoke/results_error_norm.png` and `.pdf`;
- `Runs/gpu_smoke/results_figure_15_error_norm.png` and `.pdf`;
- `Runs/gpu_smoke/results_uploads.png` and `.pdf`;
- `Runs/gpu_smoke/results_clusterhead_uploads.png` and `.pdf`;
- `Runs/gpu_smoke/results_cluster_rate.png` and `.pdf`;
- `Runs/gpu_smoke/results_cluster_quality.png` and `.pdf`.
- `Runs/gpu_smoke/results_battery.png` and `.pdf` when battery columns exist;
- `Runs/gpu_smoke/results_clusterhead_battery.png` and `.pdf` when D2D CH
  battery columns exist.
- `Runs/gpu_smoke/results_energy_used.png` and `.pdf` when energy-use columns
  exist;
- `Runs/gpu_smoke/results_energy_efficiency.png` and `.pdf` when
  energy-efficiency columns exist;
- `Runs/gpu_smoke/results_clusterhead_energy_used.png` and `.pdf` when D2D CH
  energy-use columns exist.

The thesis-style figure plots every checkpoint saved in the CSV. Omit
`--checkpoints` to save and plot the full curve for every iteration
`t = 1..max_t`; use `--checkpoints 1 100 200` only when you intentionally want
a lighter start/middle/end figure.

If `Runs/gpu_smoke/` already exists, the runner writes to the next available
suffix, such as `Runs/gpu_smoke-02/`. Without `--run-name`, the folder name is a
local timestamp: `Runs/YYYY-MM-DD-HH-MM-SS/`.

Use `--no-plots` when measuring raw simulation speed and plotting time should
not be included in the end-to-end command.

## Energy-Aware D2D CH Rotation

The thesis-compatible default keeps the post-clustering CH fixed:

```bash
--d2d-ch-rotation-mode static
```

Enhanced energy experiments can enable periodic CH re-election:

```bash
--energy-drain-mode dynamic \
--d2d-ch-rotation-mode energy_aware \
--d2d-ch-rotation-interval 10 \
--d2d-energy-efficiency-level balanced
```

The re-election is performed separately for polling+D2D, fixed+D2D, and
optimized+D2D because each curve has its own battery trajectory.  A candidate
CH must be an existing cluster member and must still reach every member within
`R_D2D`, so one-hop coverage and `Cmax` remain unchanged.  The score is:

```text
score =
  channel_weight * normalized_bs_channel_quality
  + battery_weight * current_battery
  + stability_weight * is_current_ch
```

Profiles:

- `performance`: channel `0.85`, battery `0.10`, stability `0.05`;
- `balanced`: channel `0.65`, battery `0.25`, stability `0.10`;
- `eco`: channel `0.45`, battery `0.45`, stability `0.10`.

This is not centralized CH scheduling.  The BS can broadcast profile weights
and interval, while each cluster performs a local control exchange to verify
which candidates still cover all members.  The elected CH then uses the same
ALOHA access logic as before.

Scale gradually:

1. Increase `rounds` until the GPU stays busy.
2. Increase `K` while watching memory.
3. Compare `geometric`, `utility`, and `no_d2d`; use `dense` first for thesis
   reproduction and `grid` when scaling beyond dense memory/runtime limits.
4. Save both the CSV and the `.metadata.json` file with every result.

## Statistical Validation

The JAX path is a simulation backend, not a byte-exact oracle. Validate research
results by comparing distributions and curves:

- mean error norm over `t`;
- 95 percent confidence interval half-widths;
- total successful upload counts;
- successful cluster-head upload counts;
- clustering-rate distribution;
- mean battery and mean D2D cluster-head battery when
  `--energy-drain-mode dynamic` is enabled;
- mean normalized energy used, uploads per normalized battery unit, and D2D
  cluster-head energy used when energy metrics are present;
- freshness metrics for enhanced optimized-D2D policies.

Small cases should also run structural checks:

- every device appears in exactly one active row;
- every non-singleton member is within `R_D2D` of its CH;
- no active row exceeds `Cmax`;
- singleton accounting matches the padded arrays;
- overflow diagnostics are recorded when a grid cell is split.

## Metadata To Preserve

Every thesis-scale result should preserve:

- backend name and version;
- `jax.devices()` output;
- Python version;
- seed policy;
- `K`, `M`, `L`, `Cmax`, `R_BS`, `R_D2D`;
- clustering mode and strategy;
- local repair pass count;
- local CH-rotation repair pass count;
- local CH-to-CH merge repair pass count;
- initial dense cluster size;
- channel-aware direct/CH decoding mode and battery exponents;
- energy drain mode and normalized per-attempt energy costs;
- number of rounds and checkpoints;
- batch size or effective `vmap` size;
- elapsed time including compile time;
- output CSV path and metadata JSON path.
- generated figure paths and formats.
