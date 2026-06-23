# Optimized D2D Strategy Report

This document records the optimized-D2D strategies tested so far, what they
changed in the simulation, what was observed in representative runs, and what a
plausible real deployment would need in order to support each strategy.

The most important modeling boundary is this:

```text
The BS must not estimate network density only from successful FL update
uploads.  ALOHA collisions would undercount active devices and could make a
dense network look sparse.
```

Density-aware decisions should instead use a short control/cluster-formation
stage, where CHs report cluster summaries and the BS estimates the
clusterized-device fraction before the FL update rounds.

## Metric Definitions

The tables below use the following metrics:

- `cluster_rate`: mean percentage of devices that are not singletons.
- `t_to_1e-12`: first checkpoint where the optimized-D2D mean error norm is at
  or below `1e-12`.
- `logsum`: sum over checkpoints of `log10(optimized_aloha_d2d_error_norm)`.
  More negative is better because the error curve is lower over time.
- `CH ratio`: final optimized-D2D CH uploads divided by final fixed-D2D CH
  uploads. Values near `1.0` mean optimized D2D uses roughly the same
  successful CH upload budget as fixed D2D.
- `device gain`: final optimized-D2D device uploads divided by final fixed-D2D
  device uploads.

The representative runs below use `precision=float64`, `iterations=200`, and
the current dense geometric clustering strategy unless stated otherwise.

## Main Findings

| Run | K | Strategy | Allocator | Cluster rate | t_to_1e-12 | Logsum | CH ratio | Device gain |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `k1000_utility_proportional_clip` | 1000 | utility | proportional clip | 88.759 | 146 | -1630.830 | 0.8773 | 1.6037 |
| `k1000_utility_selective_wf_025` | 1000 | utility | selective water-filling 0.25 | 88.759 | 138 | -1702.258 | 0.9600 | 1.6607 |
| `k1000_utility_density_aware_dense000` | 1000 | utility | density-aware conditional | 88.759 | 139 | -1678.585 | 0.9319 | 1.6392 |
| `k3000_utility_proportional_clip` | 3000 | utility | proportional clip | 99.319 | 67 | -2456.813 | 0.9893 | 1.4464 |
| `k3000_utility_selective_wf_025` | 3000 | utility | selective water-filling 0.25 | 99.319 | 81 | -2302.814 | 0.9599 | 1.3982 |
| `k3000_utility_density_aware_dense000` | 3000 | utility | density-aware conditional | 99.319 | 67 | -2456.813 | 0.9893 | 1.4464 |
| `k1000_adaptive_diversity` | 1000 | adaptive diversity | conditional | 88.759 | 163 | -1497.749 | 0.9671 | 1.5396 |

Observed conclusions:

- For `K=1000`, partial redistribution helped. The best representative run so
  far was `selective_water_filling` with redistribution fraction `0.25`.
- For `K=3000`, extra redistribution hurt. The dense network already had very
  high cluster coverage, so proportional clipping avoided unnecessary CH
  contention and reached `1e-12` faster.
- The density-aware conditional allocator recovered both regimes: it behaves
  like the conditional partial allocator in the `K=1000` case and falls back to
  proportional clipping in the dense `K=3000` case when
  `--optimized-d2d-dense-trigger-ratio 0.0`.
- The fixed-D2D curve is not expected to improve when optimized-D2D parameters
  change. Fixed D2D uses its own fixed access probability and does not read the
  optimized-D2D utility, allocator, trigger, novelty, or density-aware
  parameters.

## Clustering Strategies

### Thesis-Compatible One-Hop D2D

Every D2D cluster must satisfy one-hop CH coverage: every member must be within
`R_D2D` of its CH. Cluster size is capped by `Cmax`.

Real-world plausibility:

- Devices can discover local neighbors through short D2D beacons.
- A candidate CH can admit only devices that it directly reaches.
- The BS does not need to know every pairwise distance during local formation.

### Dense Pair-First Geometric Clustering

The current JAX dense geometric strategy forms local pairs first, then applies
local repairs:

- singleton join repair;
- two-device CH rotation repair;
- CH-to-CH merge repair when the target CH can cover all source members and the
  merged cluster still fits `Cmax`.

This increased cluster coverage in the tested deployments:

- about `88.759%` clustered devices for `K=1000`;
- about `99.319%` clustered devices for `K=3000`.

Real-world plausibility:

- Pairing and repair can be implemented with local request/accept D2D control
  messages.
- CH rotation is local because the candidate CH only needs to verify direct
  reachability to the devices it would cover.
- Merge repair is local if the target CH checks that it can directly cover the
  source members before accepting the merge.

### Quality CH Election

Quality CH election is an optional post-clustering refinement:

```text
score_i =
  degree_weight  * normalized_D2D_degree_i
  + channel_weight * normalized_BS_channel_quality_i
  + battery_weight * normalized_battery_i
```

The cluster membership does not change.  Inside each existing cluster, the
algorithm moves the highest-scoring valid member to the CH position only if that
member can still directly cover every member in the cluster.  If the
highest-scoring member would break one-hop coverage, the algorithm keeps looking
among valid candidates; if no better valid candidate exists, the original CH
stays in place.

Real-world plausibility:

- D2D degree comes from the same local discovery beacons used for clustering.
- BS channel quality can be estimated from downlink reference signals or past
  uplink control measurements.
- Battery is local to each device and can be exchanged as a compact class, not
  necessarily as an exact value.
- The decision can be negotiated inside the cluster after formation.  The BS
  does not need to solve a global assignment or directly appoint every CH.

Modeling interpretation:

- In thesis-compatible collision-only runs, quality CH election is mostly a
  structural ablation because the aggregate member set is unchanged.
- When `--d2d-ch-bs-success-mode channel_quality` is enabled, the elected CH's
  channel/battery affects whether a collision-free D2D aggregate reaches the
  BS.  In that setting, quality CH election becomes a physical uplink strategy,
  not only a clustering reshuffle.

### Channel-Aware CH-to-BS Success

The channel-aware CH-to-BS mode completes the quality-election idea by adding a
physical-link success stage after ALOHA contention:

```text
attempt_h = Bernoulli(p_h)
channel_h = random channel among M
collision_free_h = no other attempted CH chose channel_h

if collision_free_h:
  decoded_h = Bernoulli(q_h)
else:
  decoded_h = false
```

The decoding probability is:

```text
channel_quality_i =
  (1 / distance_to_bs_i^pathloss_exponent)
  / max_j(1 / distance_to_bs_j^pathloss_exponent)

q_i =
  min_success
  + (1 - min_success) *
    channel_quality_i *
    battery_i^battery_exponent
```

The default remains `--d2d-ch-bs-success-mode none`, so thesis-compatible runs
do not change.  The enhanced mode is enabled with
`--d2d-ch-bs-success-mode channel_quality`.

For fair D2D vs non-D2D figures, enable the direct counterpart too:

```text
--device-bs-success-mode channel_quality
```

That applies the same distance/pathloss and optional battery decoding model to
polling, fixed ALOHA, and optimized ALOHA without D2D.  The difference is that
the transmitter is an individual device instead of an elected CH carrying an
aggregate.

Real-world plausibility:

- A CH still makes a local ALOHA decision; the BS does not schedule a specific
  CH.
- Weak CHs still consume ALOHA contention opportunities when they attempt; they
  can collide with stronger CHs even if their own packet is later undecodable.
- The BS channel estimate can come from reference-signal measurements, and
  battery can be exchanged as a coarse local class during cluster formation.
- This makes quality CH election meaningful because the elected CH's identity
  now affects the probability that a collision-free aggregate reaches the BS.

Interpretation:

- This mode is more realistic but is no longer a direct reproduction of the
  original thesis curve.  It should be reported as an enhanced wireless-link
  ablation.
- Because failed CH-BS decoding reduces all D2D curves, comparisons should use
  paired runs: same seeds and parameters, with and without
  `--cluster-head-selection-mode quality`.

### Utility Clustering And No-D2D

`utility` clustering ranks candidates using local degree, battery, BS channel
quality, freshness, and CH usage. `no_d2d` produces singleton clusters for a
baseline.

Real-world plausibility:

- Degree is locally measurable from D2D discovery.
- Battery is local to the device.
- BS channel quality can be estimated from reference signals.
- Freshness and CH usage can be tracked by the BS or locally by the CH.

These are available as ablations, but the main reported experiments have used
the geometric dense strategy to stay close to the thesis D2D-SRC idea.

## Access Strategies

The simulation has six scenario curves:

1. Polling without D2D.
2. Fixed ALOHA without D2D.
3. Optimized ALOHA without D2D.
4. Polling with D2D.
5. Fixed ALOHA with D2D.
6. Optimized ALOHA with D2D.

The strategy work in this project mostly changes only scenario 6.

### Fixed ALOHA With D2D

Fixed D2D uses:

```text
p_fixed_d2d = min(M / number_of_clusterheads, 1, pcomp)
```

Each CH attempts to upload with the same probability. This is a baseline, not
an optimized policy.

Real-world plausibility:

- The BS only needs an estimate of active CH count.
- The BS broadcasts one scalar access probability.
- Each CH performs an independent ALOHA draw.

What it achieved:

- It is strong when D2D clustering is dense because CHs aggregate many devices.
- It is not affected by optimized-D2D parameters. If fixed D2D changes between
  two runs, that is due to different seeds, rounds, clustering, or statistical
  variation, not the optimized-D2D allocator.

### Thesis-Style Norm Optimized D2D

The thesis-style optimized D2D policy uses the aggregate update norm and an
adaptive scalar controller:

```text
p_h = clip(e * log(norm_h) - psi, 0, pcomp)
```

Real-world plausibility:

- `norm_h` is local to the CH after it aggregates member updates.
- `psi` is a scalar control variable that the BS can broadcast or that can be
  updated from channel-load feedback.

Limitations observed:

- When D2D aggregation drives update norms very small, this policy can
  underuse the channel late in training.
- A fixed access floor can reduce starvation, but too much floor makes the
  optimized mode look like fixed D2D.

### Utility Optimized D2D

The utility policy computes:

```text
utility_h =
  norm_h^norm_exp *
  active_cluster_size_h^size_exp *
  freshness_h^freshness_exp
```

The tuned values that became the strongest baseline were:

```text
floor = 0.02
norm_exp = 3.5
size_exp = 1.5
freshness_exp = 0.25
load_target_factor = 1.1
```

Real-world plausibility:

- `norm_h` is local after aggregation.
- `active_cluster_size_h` is known to the CH.
- `freshness_h` is the number of rounds since that CH last uploaded.
- The BS may broadcast normalizers and target load scalars, but it does not
  select specific CHs.

What it achieved:

- It made optimized D2D significantly stronger than fixed D2D in the tested
  error curves.
- It created a meaningful tradeoff: selective access is good in dense regimes,
  while some redistributed load can help less dense regimes.

## Load Allocators

The allocator converts a CH utility into an ALOHA access probability.

### Proportional Clip

```text
p_h = floor_h + remaining_load * utility_h / sum(utility)
p_h = min(p_h, pcomp)
```

If a high-utility CH hits `pcomp`, the excess probability mass is discarded.

Real-world plausibility:

- It is simple and mostly local.
- The BS needs only coarse normalizers or a scalar target.
- It does not need to redistribute leftover load.

What it achieved:

- It was best for `K=3000`, where the network was already highly clusterized.
- It reached `t_to_1e-12 = 67` with `logsum = -2456.813`.

Interpretation:

- Discarding excess load can be beneficial in dense networks because it avoids
  extra collisions from low-marginal-value CH transmissions.

### Full Water-Filling

```text
p_h = floor_h + min(pcomp - floor_h, lambda * utility_h)
```

`lambda` is chosen so the sum of probabilities approaches the target load.

Real-world plausibility:

- The BS needs enough summary information to estimate a water level, or it
  needs an iterative scalar-control approximation.
- CHs still decide locally; the BS broadcasts the scalar water level or
  equivalent normalizer.
- It is not centralized CH scheduling because no specific CH is forced to
  transmit.

What it achieved:

- It increased CH usage, but that did not always improve convergence.
- In dense `K=3000` runs, extra redistributed load was harmful.

### Selective Water-Filling

Selective water-filling redistributes only a fraction of the clipped load:

```text
target_conditional =
  proportional_clip_load
  + redistribution_fraction * max(target_load - proportional_clip_load, 0)
```

Real-world plausibility:

- Same as water-filling, but more conservative.
- The BS broadcasts the redistribution fraction and water-level scalar.

What it achieved:

- `redistribution_fraction = 0.25` was the best representative `K=1000` run:
  `t_to_1e-12 = 138`, `logsum = -1702.258`.
- It was worse for `K=3000`: `t_to_1e-12 = 81`, compared with `67` for
  proportional clipping.

### Throughput-Conditional Selective Water-Filling

This allocator activates selective water-filling only when optimized-D2D CH
throughput is below the fixed-D2D reference:

```text
throughput_ratio =
  optimized_success_ewma / fixed_success_target

if throughput_ratio < trigger_ratio:
  use selective water-filling
else:
  return exact proportional_clip probabilities
```

The exact fallback matters. Returning water-filling with a similar total load
still reshuffles CH probabilities and caused regressions in dense `K=3000`
runs.

Real-world plausibility:

- `optimized_success_ewma` comes from ACK-observable successful CH uploads.
- `fixed_success_target` can be computed from `M`, the fixed access
  probability, and active CH count.
- The BS broadcasts one effective trigger and the optional water-level scalar.

What it achieved:

- It improved over always-water-filling by avoiding redistribution when useful
  throughput was already high.
- A single fixed trigger did not solve both regimes: `0.95` was too aggressive
  for dense `K=3000`, while `0.90` was less useful for `K=1000`.

### Density-Aware Conditional Selective Water-Filling

This is the current best adaptive allocator:

```text
clusterized_fraction = clusterized_devices_rate / 100

if clusterized_fraction >= density_trigger_threshold:
  effective_trigger = dense_trigger_ratio
else:
  effective_trigger = redistribution_trigger_ratio

if throughput_ratio < effective_trigger:
  use selective water-filling
else:
  return exact proportional_clip probabilities
```

The best current cross-regime setting is:

```text
redistribution_trigger_ratio = 0.95
density_trigger_threshold = 0.95
dense_trigger_ratio = 0.0
redistribution_fraction = 0.25
```

Real-world plausibility:

- The BS must estimate `clusterized_fraction` from cluster-formation/control
  signaling, not from successful FL updates.
- CHs still compute local access probabilities and make local ALOHA decisions.
- The BS only broadcasts scalar control values: effective trigger, target load,
  redistribution fraction, and any water-level/normalizer.

What it achieved:

- For `K=1000`, `clusterized_fraction < 0.95`, so it kept the non-dense
  conditional behavior: `t_to_1e-12 = 139`.
- For `K=3000`, `clusterized_fraction > 0.95`, so it fell back to
  proportional clipping and recovered the best dense-regime result:
  `t_to_1e-12 = 67`.

Interpretation:

- This does not beat proportional clipping in dense networks; it learns when
  proportional clipping is the right choice.
- It is useful because it avoids choosing one static allocator for all network
  densities.

## Other Algorithmic Ablations

### Max-Weight Threshold

```text
p_h = floor + (pcomp - floor) * sigmoid(gain * (utility_h - threshold))
```

Real-world plausibility:

- The BS broadcasts a scalar threshold.
- CHs compute utility locally.
- The threshold can be adjusted from load or throughput feedback.

Observed status:

- Kept as an ablation.
- It did not become the best current strategy in the representative results.

### Hybrid Utility With Directional Novelty

```text
hybrid_utility_h = utility_h * novelty_h^eta
```

`novelty_h` discounts CH aggregate directions aligned with recent successful
optimized-D2D uploads.

Real-world plausibility:

- The BS broadcasts a recent reference update direction with the FL model.
- Each CH compares its local aggregate direction with that reference.
- This has higher control overhead than pure utility because the reference is
  an `L`-dimensional vector, not only a scalar.

Observed status:

- Kept as an ablation.
- It did not outperform the tuned utility allocator in the representative
  runs.

### Adaptive Diversity

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

Real-world plausibility:

- The phase uses `t / max_t`, which is known from the planned experiment or
  training schedule.
- It does not use true model error, which would be unavailable in deployment.
- It needs the same reference-direction mechanism as hybrid mode.

Observed status:

- It was scientifically motivated, but the representative `K=1000` run was
  worse than tuned utility: `t_to_1e-12 = 163` and `logsum = -1497.749`.
- It remains useful as an ablation showing that novelty/freshness scheduling is
  not automatically beneficial in this specific simulation.

## Plausible Real Deployment Arrangement

A realistic implementation should separate control signaling from FL update
uploads.

### 1. BS Beacon And D2D Discovery

The BS broadcasts:

- experiment/training round parameters;
- `M`, `pcomp` assumptions or allowed access classes;
- D2D radius/power class;
- control-slot timing.

Devices exchange short D2D beacons and estimate local neighbor reachability.

### 2. Local Cluster Formation

Devices form clusters through local D2D messages:

- candidate CH announcement;
- join request;
- accept/reject based on one-hop coverage and `Cmax`;
- optional local repair/rotation/merge messages.

This stage should not require the BS to solve a global clustering assignment.

### 3. Cluster Summary Reporting

CHs report compact summaries to the BS:

- CH identity or temporary ID;
- cluster size;
- singleton/covered count if available;
- optional battery/channel/freshness class;
- optional utility summary for water-level estimation.

Singletons can be handled with robust control-plane access, backoff, or
periodic retry.

The BS estimates:

```text
number_of_clusterheads
clusterized_devices_fraction
expected_fixed_d2d_ch_successes
```

This density estimate must come from the control/cluster-formation stage. It
should not be inferred only from successful FL data uploads because collisions
would bias the estimate downward.

### 4. BS Broadcasts Scalar Control Parameters

Depending on the selected strategy, the BS broadcasts:

- fixed D2D access probability;
- utility normalizers;
- load target, usually near `M` or `M * load_target_factor`;
- `redistribution_fraction`;
- effective trigger ratio;
- water level or equivalent normalizer;
- optional max-weight threshold;
- optional novelty reference direction;
- optional adaptive-diversity phase scalar.

The BS does not need to say "CH 7 transmits now." It provides parameters; each
CH computes its own probability.

### 5. CH Local Probability And ALOHA Trial

Each CH computes:

```text
local utility -> local probability -> random ALOHA trial
```

If it transmits and avoids collision, the BS receives the aggregate update.

### 6. ACKs And EWMA Feedback

The BS sends or records ACKs for successful CH uploads. It updates:

```text
optimized_success_ewma
throughput_ratio
freshness counters
recent successful reference direction
```

These feedback signals drive conditional water-filling, max-weight threshold
updates, hybrid novelty, and adaptive-diversity state.

## Strategy Comparison By Required Information

| Strategy | CH local information | BS/global information | Extra overhead | Main risk |
| --- | --- | --- | --- | --- |
| Fixed D2D | CH identity and random draw | active CH count, `M` | very low | no utility awareness |
| Norm optimized | aggregate norm | scalar `psi`/load feedback | low | can underuse channels late |
| Utility + proportional clip | norm, cluster size, freshness | normalizers, load target | low to moderate | may waste clipped load |
| Full water-filling | utility | water level from utility distribution | moderate | can create collisions in dense regimes |
| Selective water-filling | utility | water level plus redistribution fraction | moderate | fraction may not fit every density |
| Conditional water-filling | utility | ACK EWMA, fixed-D2D reference, trigger | moderate | trigger can be density-dependent |
| Density-aware conditional | utility | clusterized fraction from control plane, ACK EWMA | moderate | bad density estimates can choose wrong mode |
| Quality CH election | local D2D degree, BS channel estimate, battery, one-hop coverage | optional score weights | low to moderate | limited gain unless CH-BS channel quality affects success/cost |
| Channel-aware CH-BS success | elected CH channel quality and battery | optional pathloss/min-success parameters | low to moderate | makes enhanced runs less thesis-comparable |
| Dynamic energy drain | local battery estimate, attempted direct/D2D/CH transmissions | energy-cost coefficients, optional battery classes | moderate | useful for energy/fairness ablations, but not calibrated yet |
| Energy-aware CH rotation | current battery, BS channel estimate, current CH identity, one-hop coverage | rotation interval and profile weights | moderate | rotation overhead may outweigh error/energy gains |
| Max-weight | utility | scalar threshold | low to moderate | threshold tuning can be unstable |
| Hybrid | aggregate direction | recent reference direction | higher | reference vector overhead |
| Adaptive diversity | norm, size, freshness, direction | reference direction, phase scalar | higher | phase schedule may not fit the task |

## Current Recommendation

For thesis-compatible reproduction, keep:

```text
--optimized-d2d-access-mode norm
```

For the current best enhanced strategy across the tested `K=1000` and `K=3000`
regimes, use:

```text
--optimized-d2d-access-mode utility
--optimized-d2d-access-floor-fraction 0.02
--optimized-d2d-norm-exponent 3.5
--optimized-d2d-cluster-size-exponent 1.5
--optimized-d2d-freshness-exponent 0.25
--optimized-d2d-load-target-factor 1.1
--optimized-d2d-load-allocation-mode conditional_selective_water_filling
--optimized-d2d-redistribution-fraction 0.25
--optimized-d2d-redistribution-trigger-ratio 0.95
--optimized-d2d-density-trigger-threshold 0.95
--optimized-d2d-dense-trigger-ratio 0.0
--optimized-d2d-throughput-ewma-decay 0.90
```

If optimizing only for `K=1000`, `selective_water_filling` with fraction `0.25`
was slightly better in the representative run. If optimizing across densities,
the density-aware conditional allocator is more defensible because it adapts to
the observed clusterization regime.

## Open Validation Work

- Test robustness when `clusterized_devices_fraction` is noisy or delayed.
- Test imperfect D2D member compute/link probabilities below `1.0`.
- Test whether density thresholds generalize to `K=500`, `K=5000`, and
  `K=10000`.
- Test quality CH election with `--d2d-ch-bs-success-mode channel_quality` to
  measure whether better elected CHs improve the decoded D2D aggregate stream.
- Calibrate the dynamic energy-cost coefficients and compare `performance`,
  `balanced`, and `eco` periodic CH re-election against its control overhead.
- Compare control overhead of scalar-only utility policies against
  reference-vector policies such as hybrid/adaptive diversity.
- Separate "better final numerical floor" from "better convergence before
  numerical saturation" by emphasizing target times and log-error AUC.
