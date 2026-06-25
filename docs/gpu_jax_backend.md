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

The enhanced physical-link option is `rayleigh_outage` for member-to-CH D2D
links, CH-to-BS links, and direct device-to-BS links:

```text
avg_snr_i = reference_snr / max(distance_i, 1)^pathloss_exponent
q_i       = exp(-snr_threshold / max(avg_snr_i, eps))
```

This is still not a full SINR simulator: ALOHA collisions remain the explicit
interference abstraction, and Rayleigh outage is the collision-free packet
decoding probability.  For member-to-CH D2D links, select
`--d2d-member-link-success-mode rayleigh_outage`; the distance term is each
member's distance to the currently elected CH, producing a
`float[max_clusters, Cmax]` probability matrix.  For CH-to-BS and direct BS
links, use `--d2d-ch-bs-success-mode rayleigh_outage` and
`--device-bs-success-mode rayleigh_outage`.  Use
`--cluster-head-channel-score-mode rayleigh_outage` when quality CH election
should rank candidate CHs by the same outage metric rather than normalized
inverse pathloss.  The BS can estimate or configure the reference SNR and
threshold as control parameters; clusters still elect only from members that
preserve one-hop D2D coverage.

The preferred enhanced energy model is `--energy-model first_order_radio`.
The previous `constant` model is retained for reproducibility and uses the
three fixed normalized costs.  The first-order radio model separates:

```text
direct device: E_tx_bs(update_size, d_device_bs)
D2D member:    E_tx_d2d(update_size, d_member_ch)
CH receive:    active_non_ch_members * E_rx(update_size)
CH aggregate:  active_updates * E_agg(update_size)
CH uplink:     E_tx_bs(aggregate_size, d_ch_bs)
```

With `--battery-feasibility-mode required_energy`, each role attempts only if
the current scenario-specific battery can pay the required energy.  This makes
battery a physical availability constraint rather than a loose multiplicative
decode-probability factor.  The older battery exponents remain available for
legacy channel-quality ablations.

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

The AoI-aware utility mode is selected with
`--optimized-d2d-access-mode aoi_aware_utility`. It is intentionally narrower
than `adaptive_diversity`: it keeps the base utility score and load allocator,
but adds a stale-tail pressure term:

```text
base_utility_h =
  norm_h^norm_exp *
  active_cluster_size_h^size_exp *
  freshness_h^freshness_exp

normalized_aoi_h = AoI_h / max_j(AoI_j)
tail_h = clip(
  (normalized_aoi_h - threshold_fraction) / (1 - threshold_fraction),
  0,
  1
)
aoi_bonus_h = 1 + aoi_weight * tail_h^aoi_exp
aoi_aware_utility_h = base_utility_h * aoi_bonus_h
```

The motivation is the observed physical-energy tradeoff: optimized D2D can
reduce error and energy strongly while still having worse mean AoI than fixed
D2D.  This mode tests whether a bounded bonus for stale clusters can improve
mean and 95th-percentile AoI without throwing away the error/energy gains.
It does not require the BS to schedule a CH centrally.  The CH can maintain its
own AoI from ACK/no-ACK feedback, while the BS broadcasts scalar normalizers and
the load-controller parameters.

The conservative AoI-floor utility mode is selected with
`--optimized-d2d-access-mode aoi_floor_utility`. It was added after the
multiplicative AoI bonus showed a real mean-AoI improvement but a worse
error/energy tradeoff. The policy keeps the base utility allocator as the main
decision, then raises only stale clusters to a bounded minimum probability:

```text
base_probability_h = load_control(base_utility_h)
stale_floor_h =
  fixed_d2d_access_probability *
  aoi_weight *
  tail_h^aoi_exp

p_h = max(base_probability_h, stale_floor_h)
p_h = min(p_h, pcomp)
```

In this mode, `aoi_weight` is not a utility multiplier. It is the largest
stale-floor probability as a fraction of fixed-D2D access probability. For
example, `aoi_weight=0.25` gives the oldest stale clusters at least one quarter
of the fixed-D2D access probability, unless `pcomp` is smaller. This keeps the
policy distributed and deployable: CHs still perform local ALOHA trials, and
the BS only needs scalar load/AoI normalizers plus ACK feedback.

The stale-tail AoI quota mode is selected with
`--optimized-d2d-access-mode aoi_tail_utility`. It was added after the
AoI-aware/floor access modes and AoI-triggered CH rotation failed to clear the
p75/p90/p95 AoI tail. The policy computes one probability from base utility and
one probability from stale-tail AoI pressure, then mixes them:

```text
base_probability_h = load_control(base_utility_h)
tail_probability_h = load_control(tail_h^aoi_exp, floor = 0)
p_h = (1 - quota) * base_probability_h + quota * tail_probability_h
```

where `quota = clip(aoi_weight, 0, 1)`. If active clusters have no
differentiated AoI tail, `p_h` is exactly the base utility probability. This
keeps early rounds and non-stale deployments from paying unnecessary fairness
overhead. The deployment model remains the same scalar-control ALOHA model:
ACK age is local to the CH, and the BS can broadcast normalizers, threshold,
and quota.

The quality-gated stale-tail mode is selected with
`--optimized-d2d-access-mode aoi_quality_tail_utility`. It keeps the same base
probability and AoI-tail quota, but changes the tail utility from only
`tail_h^aoi_exp` to:

```text
tail_h^aoi_exp *
q_ch_bs_h^aoi_channel_exp *
battery_ch_h^aoi_battery_exp
```

where `q_ch_bs_h` is the current collision-free CH-to-BS success probability
and `battery_ch_h` is the current normalized battery of the elected CH. This is
intended for the physical-energy/Rayleigh path: stale clusters are favored only
when their CH is also likely to decode at the BS and has enough energy to make
the attempt plausible. It remains scalar-control ALOHA; no CH is centrally
scheduled by the BS.

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
- `aoi_aware_utility` keeps base utility and adds bounded stale-tail AoI
  pressure to reduce freshness tail risk.
- `aoi_floor_utility` keeps base utility probabilities and only enforces a
  bounded minimum probability for stale-tail clusters; it is the lower-risk
  AoI ablation when the multiplicative bonus harms error or energy.
- `aoi_tail_utility` keeps most probability on base utility but reserves an
  explicit quota for stale-tail clusters; it is the current test aimed at p75,
  p90, p95, and stale-fraction AoI rather than only mean AoI.
- `aoi_quality_tail_utility` applies that same stale-tail quota only after
  weighting the tail by CH-BS success probability and current CH battery, so it
  is the next test when plain AoI quota lowers freshness metrics but damages
  convergence or energy efficiency.
- `member_fair_utility` reserves part of the optimized-D2D load budget for
  multi-member active aggregates containing stale or zero-participation D2D
  members. It is the current ablation for testing whether optimized D2D improves
  aggregate quality while starving some member devices. Ideal-link `K=1000`,
  `rounds=100` runs support `w=0.15 / threshold=0.70` as the cleaner Pareto
  point (`member_aoi=46.451`, `member_zero=0.163`,
  `final_error=4.018e-07`) and `w=0.25 / threshold=0.70` as the stronger
  fairness ablation (`member_aoi=45.010`, `member_zero=0.134`,
  `final_error=8.758e-07`). Physical energy/Rayleigh runs now have a refreshed
  `utility` baseline with member columns. In that path, `w=0.05 /
  threshold=0.70` is the only current member-fair Pareto candidate
  (`member_aoi=60.541` versus `62.043`, `member_zero=0.393` versus `0.431`,
  `final_error=3.066e-07` versus `1.603e-07`, `energy_efficiency=817.898`
  versus `877.028`). Larger weights reduce freshness metrics more but are not
  physical defaults because convergence and energy efficiency degrade quickly.
- `member_refresh_utility` is the next access-side test after the member-stale
  failure attribution showed `CH no attempt` dominates stale samples. It keeps
  utility load control but gives refresh-eligible clusters a local minimum
  access probability via `--optimized-d2d-member-refresh-floor-fraction`.
- `member_quota_utility` makes that test more explicit. It treats
  `--optimized-d2d-aoi-weight` as a reserved stale-member quota, reduces the
  base utility contender target by that fraction, and adds a separate
  refresh-eligible overlay. This gives the experiment a cleaner answer to
  whether member freshness improves when part of the optimized-D2D access budget
  is reserved for stale or never-delivered active members.
- `member_capped_quota_utility` keeps the same pure ALOHA base/overlay quota
  structure, but caps only the extra refresh overlay before it is added to the
  base probability. The cap is set with
  `--optimized-d2d-member-quota-cap-fraction` as a fraction of the fixed D2D
  ALOHA access probability. This tests whether stale-member quota pressure is
  too concentrated in a few CHs without introducing scheduled slots, SIC, MPR,
  NOMA, TDMA/OFDMA, or a centralized scheduler.
- `member_collision_aware_quota` keeps the member quota but dampens it when an
  EWMA of optimized-D2D CH collisions exceeds
  `--optimized-d2d-member-collision-target-fraction`. The damping strength is
  controlled by `--optimized-d2d-member-collision-gain`, with a lower bound set
  by `--optimized-d2d-member-collision-min-quota-scale`. This is the first
  collision-control candidate after the negative deficit result.
- `member_collision_aware_queue_quota` keeps the quota and deficit tie-breaker,
  but changes the virtual-queue update. No-attempt misses add full stale-member
  pressure to the queue, CH-BS misses add a small amount, and collision-caused
  misses do not add debt. It reuses
  `--optimized-d2d-member-deficit-decay` and
  `--optimized-d2d-member-deficit-weight`. The first K=3000 experiment was
  negative: the queue reduced no-attempt attribution but increased collisions,
  reduced useful uploads, and degraded member freshness.
- `semi_scheduled_member_refresh` moves beyond global ALOHA probability
  shaping. It reserves
  `ceil(M * --optimized-d2d-member-schedule-fraction)` D2D channels for the
  highest active member-pressure clusters, treats those CH attempts as
  collision-free scheduled opportunities, and leaves the remaining channels to
  utility-controlled ALOHA. Scheduled CHs still require enough battery and a
  successful CH-to-BS link draw. Use
  `--optimized-d2d-member-schedule-deficit-weight` only as a tie-breaker among
  repeatedly missed refresh opportunities. This mode assumes a small BS/CH
  control decision for reserved slots and is therefore a coordinated
  semi-scheduled ablation, not a pure distributed ALOHA mode. The optional
  `--optimized-d2d-member-schedule-control-cost` parameter charges a normalized
  per-scheduled-CH coordination overhead to optimized+D2D CH energy/battery;
  leave it at `0.0` to reproduce the original no-overhead semi-scheduled runs.
- `member_deficit_utility` keeps the quota split but ranks the refresh overlay
  with a persistent missed-refresh deficit. This is intended for cases where
  many member AoIs saturate at the same value, making instantaneous stale-tail
  pressure almost tied across many clusters. The deficit is damped by
  `--optimized-d2d-member-deficit-weight`; the undamped v2 experiment showed
  that overly strong deficit ranking can reduce `CH no attempt` but create too
  many ALOHA collisions.

The detailed physical/Rayleigh member-level comparison is recorded in
`docs/member_level_d2d_freshness_experiments.md`. Current guidance is to use
`member_quota_utility` as the strongest pure ALOHA/probability-shaping
member-freshness candidate, keep `member_deficit_utility` only as a negative
collision-dominated ablation, and use `semi_scheduled_member_refresh` only as a
coordinated upper-bound/future-work reference.
The best tested `K=1000` balanced semi-scheduled point is schedule fraction
`0.20` with deficit tie-breaker weight `0.0`; the first `K=3000` robustness
check favored `0.30` over `0.20` and beat the same-`K`
`member_quota_utility` baseline on convergence, member freshness, and final
energy efficiency. The pure-ALOHA K=3000 matrix with larger
`member_quota_utility` weights and less aggressive
`member_collision_aware_quota` damping has now been run: stronger quota weights
did not help, while `member_collision_quota_k3000_w015_t002_g2_min050` is best
kept as a convergence/energy ablation rather than a new member-freshness
winner. The first ALOHA-only structural load-cap candidate,
`member_capped_quota_utility`, has also been run. Wide caps did not bind;
`w=0.15, cap=0.10` is the best capped convergence/energy point (`t<=1e-12` at
round `87`, `+9.48%` final energy efficiency), but it worsens
zero-participation. `cap=0.25` is the most freshness-balanced capped point, but
its gains are below `0.15%` and it does not improve convergence. Treat capped
quota as another ALOHA ablation. The collision-aware virtual queue was then
implemented as `member_collision_aware_queue_quota`, but its first K=3000 run
is a negative ablation because it moved pressure into collisions and failed to
converge. Move next to re-clustering or cluster splitting before adding more
pure probability-shaping variants.

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
- `mean_aoi`: `float[checkpoints, 6]`.
- `peak_aoi`: `float[checkpoints, 6]`.
- `p75_aoi`: `float[checkpoints, 6]`.
- `p90_aoi`: `float[checkpoints, 6]`.
- `p95_aoi`: `float[checkpoints, 6]`.
- `stale_fraction_50`: `float[checkpoints, 6]`, fraction with AoI greater
  than `1 + 50%` of the elapsed iteration count.
- `stale_fraction_75`: `float[checkpoints, 6]`, fraction with AoI greater
  than `1 + 75%` of the elapsed iteration count.
- `stale_fraction_100`: `float[checkpoints, 6]`, fraction with AoI greater
  than 100 rounds.
- `checkpoints`: `int32[checkpoints]`.
- `d2d_member_mean_aoi`, `d2d_member_peak_aoi`, `d2d_member_p75_aoi`,
  `d2d_member_p90_aoi`, `d2d_member_p95_aoi`: `float[checkpoints, 3]`,
  member-level AoI over devices in non-singleton D2D clusters.
- `d2d_member_stale_compute_failure_fraction`,
  `d2d_member_stale_link_failure_fraction`,
  `d2d_member_stale_member_energy_failure_fraction`,
  `d2d_member_stale_ch_no_attempt_fraction`,
  `d2d_member_stale_collision_fraction`,
  `d2d_member_stale_ch_bs_failure_fraction`, and
  `d2d_member_stale_other_failure_fraction`: `float[checkpoints, 3]`,
  severe-stale member failure attribution fractions for D2D scenarios.
- `d2d_member_stale_fraction_50`, `d2d_member_stale_fraction_75`,
  `d2d_member_stale_fraction_100`: `float[checkpoints, 3]`, member-level
  stale-tail fractions using the same thresholds as the cluster/device AoI
  metrics.
- `d2d_member_participation_p05`,
  `d2d_member_zero_participation_fraction`: `float[checkpoints, 3]`,
  lower-tail and zero-count delivered-update participation diagnostics for
  devices in non-singleton D2D clusters.

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
- `Runs/gpu_smoke/results_aoi.png` and `.pdf` when AoI columns exist;
- `Runs/gpu_smoke/results_peak_aoi.png` and `.pdf` when peak-AoI columns exist.
- `Runs/gpu_smoke/results_p75_aoi.png`, `results_p90_aoi.png`,
  `results_p95_aoi.png`, and stale-fraction plots when those columns exist.
- `Runs/gpu_smoke/results_member_failure_breakdown.png` when optimized-D2D
  member-stale failure attribution columns exist.
- `Runs/gpu_smoke/results_member_aoi.png`, `results_member_p95_aoi.png`,
  `results_member_stale_fraction_75.png`, and
  `results_member_zero_participation_fraction.png` when D2D member-level
  columns exist.

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

Enhanced energy experiments can enable periodic or AoI-triggered CH
re-election:

```bash
--energy-drain-mode dynamic \
--d2d-ch-rotation-mode energy_aware \
--d2d-ch-rotation-interval 10 \
--d2d-ch-rotation-trigger-mode interval \
--d2d-energy-efficiency-level balanced
```

Trigger modes:

- `interval`: previous periodic behavior;
- `aoi`: rotate only clusters whose AoI is in the stale tail;
- `interval_or_aoi`: periodic rotation plus stale-tail rotation.

The AoI trigger uses `--d2d-ch-rotation-aoi-threshold-fraction` as a fraction of
the current maximum active-cluster AoI within each D2D scenario.  It is designed
to test whether stale clusters are better helped by replacing a weak CH than by
only increasing that cluster's ALOHA access probability.

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

This is not centralized CH scheduling.  The BS can broadcast profile weights,
interval, and AoI threshold, while each cluster performs a local control
exchange to verify which candidates still cover all members.  The elected CH
then uses the same ALOHA access logic as before.

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
- energy model, first-order radio coefficients, battery-feasibility mode, and
  optional CH-rotation control overhead;
- direct/CH Rayleigh reference SNR and threshold when outage decoding is used;
- quality CH channel-score mode;
- number of rounds and checkpoints;
- batch size or effective `vmap` size;
- elapsed time including compile time;
- output CSV path and metadata JSON path.
- generated figure paths and formats.
