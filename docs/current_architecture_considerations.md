# Current Architecture Considerations

This document explains the current experiment architecture and the modeling
signals used by the enhanced D2D strategies.  It is written as a design record:
what the code currently assumes, why each signal exists, how it could be
implemented in a real wireless deployment, and where the simulation is still
limited.

The most important separation is:

```text
thesis-compatible mode
  -> reproduce the original HFL + multichannel ALOHA idea as closely as useful

enhanced mode
  -> test realistic improvements such as utility access, quality CH election,
     freshness, load control, and channel-aware CH-to-BS decoding
```

Do not compare enhanced runs directly against thesis-compatible runs unless the
model differences are explicitly reported.

## High-Level Pipeline

The main `.py` workflow is:

```text
main.py
  -> experiments/run_gpu_sweep.py
  -> Clustering/jax_clustering_algorithm.py
  -> Models/jax_models_arrangement.py
  -> Runs/<run_name>/
```

`experiments/run_gpu_sweep.py` is responsible for experiment orchestration:

- creates the JAX device deployment;
- creates D2D clusters;
- runs the HFL/ALOHA trajectory;
- aggregates Monte Carlo rounds;
- writes CSV, metadata JSON, and figures.

`Clustering/jax_clustering_algorithm.py` is responsible for device geometry and
D2D cluster formation.

`Models/jax_models_arrangement.py` is responsible for FL updates, ALOHA access,
D2D aggregation, CH-to-BS transmission success, and trace metrics.

## Device Model

Each generated device has:

```text
device_id
coords = (x, y), meters
distance_to_bs, meters
battery, percentage in 1..100
```

The default placement follows the thesis:

```text
r ~ U(1, R_BS)
theta ~ U(0, 2*pi)
```

`distance_to_bs` is used as a proxy for BS channel quality.  The current channel
quality approximation is:

```text
raw_channel_i = 1 / max(distance_to_bs_i, 1)^pathloss_exponent
channel_quality_i = raw_channel_i / max(raw_channel)
```

This is intentionally simple.  It captures pathloss but does not model fast
fading, shadowing, SINR, coding rate, or time-varying channel state.

`battery` is currently static during a run.  It affects CH scoring and can
affect CH-to-BS decoding probability through a battery exponent, but the
simulator does not yet drain energy over time.

## D2D Clustering

The main clustering strategy is dense geometric one-hop D2D:

```text
dense pair-first clustering
+ singleton join repair
+ pair CH-rotation repair
+ CH-to-CH merge repair
```

The invariant is:

```text
every cluster member must be within R_D2D of the cluster head
cluster size must be <= Cmax
```

This is deliberately one-hop.  It avoids assuming a multi-hop D2D routing
protocol that the thesis did not model.

### Real Deployment Interpretation

The clustering stage can be implemented as local control signaling:

```text
1. BS broadcasts D2D discovery parameters.
2. Devices send short D2D beacons.
3. Devices estimate reachable neighbors inside R_D2D.
4. Candidate CHs announce local intent.
5. Members join only if the candidate CH directly reaches them.
6. Local repairs/merges happen only when one-hop coverage and Cmax still hold.
```

The BS does not need to solve a global clustering optimization.  It may receive
compact cluster summaries later, but cluster formation itself is local.

## Quality CH Election

The default CH selection mode is:

```text
--cluster-head-selection-mode first
```

This preserves the original first member in the cluster row as CH.

The enhanced mode is:

```text
--cluster-head-selection-mode quality
```

It keeps cluster membership fixed and only changes which member is placed in
column 0 as CH.  A candidate can become CH only if it still directly covers
every current cluster member.

The score is:

```text
score_i =
  degree_weight  * normalized_D2D_degree_i
  + channel_weight * normalized_BS_channel_quality_i
  + battery_weight * normalized_battery_i
```

Current tested weights:

```text
0.4 / 0.4 / 0.2  -> balanced first quality run
0.2 / 0.7 / 0.1  -> more channel-heavy
0.1 / 0.8 / 0.1  -> current best tested channel-aware CH election
```

### What Each Term Means

`normalized_D2D_degree_i`

- Number of D2D neighbors within `R_D2D`, normalized by the maximum degree in
  the deployment.
- A higher value means the device is more locally central.
- It helps avoid selecting a CH that has a good BS channel but is poorly placed
  for local D2D coverage.

`normalized_BS_channel_quality_i`

- Inverse pathloss proxy from device to BS.
- A higher value means a collision-free CH-to-BS upload is more likely to be
  decoded when channel-aware success is enabled.
- This became the dominant useful signal once
  `--d2d-ch-bs-success-mode channel_quality` was implemented.

`normalized_battery_i`

- Battery percentage divided by 100.
- It discourages low-battery devices from becoming CHs when weights give it
  influence.
- The current simulator does not yet model battery drain, so this is a static
  CH suitability signal rather than a full energy model.

### Why Quality CH Election Needed Channel-Aware CH-BS Success

If cluster membership does not change, the aggregate gradient sent by the
cluster is still the sum of the same members.  Therefore CH election has little
expected effect unless the elected CH identity affects something physical.

The optional channel-aware CH-to-BS mode makes CH identity matter:

```text
CH attempts ALOHA
-> CH may collide
-> if collision-free, BS decodes with probability q_h
```

Then a CH with better BS channel/battery can improve the probability that the
aggregate reaches the BS.

## Member-To-CH Availability

D2D member participation has two optional realism knobs:

```text
--d2d-member-compute-probability
--d2d-member-link-success-probability
```

Defaults:

```text
d2d_member_compute_probability = 1.0
d2d_member_link_success_probability = 1.0
```

Meaning:

- compute probability models whether a member produced its local update;
- link success probability models whether that member delivered its update to
  the CH over D2D.

When either value is below `1.0`, a successful CH-to-BS upload carries:

```text
CH own update
+ updates from members that computed successfully
+ updates from members whose D2D link succeeded
```

The current implementation does not yet make member-to-CH link success depend
on D2D distance, D2D channel quality, or interference.  It is a global
probability knob.

## CH-To-BS Channel-Aware Success

The optional enhanced mode is:

```text
--d2d-ch-bs-success-mode channel_quality
```

Default:

```text
--d2d-ch-bs-success-mode none
```

The default preserves thesis-compatible collision-only D2D uploads.  In
`channel_quality` mode, a D2D CH upload succeeds only if:

```text
1. the CH attempts access;
2. no other attempted CH selected the same ALOHA channel;
3. the BS decodes the collision-free CH packet.
```

The decode probability is:

```text
q_i =
  min_success
  + (1 - min_success)
    * channel_quality_i
    * battery_i^battery_exponent
```

Current tested values:

```text
min_success = 0.35
pathloss_exponent = 2.0
battery_exponent = 0.25
```

Important interpretation:

- Weak CHs still consume channel contention if they attempt.
- Failed physical decoding happens after a collision-free ALOHA event.
- This is not centralized scheduling.
- The BS can estimate the channel part from reference signals or historical
  ACK/NACK/CQI-like feedback.

Current limitation:

- Channel-aware success is currently applied to D2D CH-to-BS uploads.
- Direct non-D2D device-to-BS uploads are still idealized compared with D2D.
- A fully fair physical-layer comparison should add the equivalent
  device-to-BS channel-aware success for polling/fixed/optimized non-D2D curves.

## FL Update And Aggregation Model

The task is synthetic linear regression:

```text
X_i in R^L
y_i = X_i dot w_true
grad_i(w) = X_i * (X_i dot w - y_i)
```

The CH aggregate is currently a sum:

```text
aggregate_h = sum(member_updates_h)
```

The BS update follows the thesis-style unscaled step by default:

```text
w <- w - u1 * gradient
```

`--normalize-by-k` enables a conservative ablation:

```text
w <- w - u1 * gradient / K
```

Current limitation:

- The simulator does not yet model non-IID data similarity, labels, local
  epochs, compression, quantization, or privacy noise.
- CH quality is about communication and scheduling, not data distribution.

## Optimized D2D Access Policy

The thesis-compatible optimized D2D access mode is:

```text
--optimized-d2d-access-mode norm
```

The current best enhanced mode is:

```text
--optimized-d2d-access-mode utility
```

The utility score is:

```text
utility_h =
  norm_h^norm_exp
  * active_cluster_size_h^size_exp
  * freshness_h^freshness_exp
```

Current strong tested values:

```text
norm_exp = 3.5
size_exp = 1.5
freshness_exp = 0.25
load_target_factor = 1.1
```

### What Each Term Means

`norm_h`

- Norm of the aggregate update at CH `h`.
- Larger norms indicate larger immediate contribution to model movement.
- This is close to the original optimized ALOHA idea.

`active_cluster_size_h`

- Number of active member updates included in the current aggregate.
- It rewards CHs carrying more device information.
- This matters because one CH upload can represent multiple devices.

`freshness_h`

- Age since the last successful optimized-D2D upload by that CH.
- It increases while the CH is not decoded by the BS.
- It resets after a successful optimized-D2D CH upload.
- It prevents the policy from repeatedly favoring only the easiest or largest
  clusters.

Freshness is realistic because the BS observes ACKs/successes.  It does not
need true training error.

## Load Allocation

The access policy produces utilities.  The allocator converts utilities into
ALOHA access probabilities.

### Proportional Clip

```text
p_h = floor_h + remaining_load * utility_h / sum(utility)
p_h = min(p_h, pcomp)
```

It is simple, but clipped probability mass is lost.

### Water-Filling

```text
p_h = floor_h + min(pcomp - floor_h, lambda * utility_h)
```

The scalar `lambda` is chosen so the expected load approaches the target when
capacity exists.  The BS can broadcast a scalar water level or equivalent
normalizer.  CHs still decide locally.

### Selective Water-Filling

Redistributes only a fraction of clipped load.  This is more conservative than
full water-filling.

### Conditional Selective Water-Filling

Current recommended allocator:

```text
--optimized-d2d-load-allocation-mode conditional_selective_water_filling
```

It redistributes clipped probability only when observed optimized-D2D CH
throughput is below the expected fixed-D2D CH throughput.

The condition uses:

```text
optimized_success_ewma
fixed_success_target
redistribution_trigger_ratio
```

This is realistic because ACKs reveal successful CH uploads.

### Density-Aware Trigger

The allocator also considers the clusterized-device fraction:

```text
if clusterized_fraction >= density_trigger_threshold:
  effective_trigger = dense_trigger_ratio
else:
  effective_trigger = redistribution_trigger_ratio
```

Reason:

- Less dense D2D may benefit from extra redistribution.
- Very dense D2D can suffer from extra collision pressure.

The BS should estimate clusterized fraction from the control/cluster-formation
stage, not only from successful FL uploads.  Successful uploads alone can
undercount devices when collisions are frequent.

## Other Enhanced Access Modes

`max_weight`

- Uses a threshold/sigmoid gate over utility.
- More selective than smooth utility.
- Kept as an ablation, not the current best strategy.

`hybrid`

- Multiplies utility by directional novelty.
- Needs a recent reference update direction from the BS.
- Higher overhead because the BS broadcasts an `L`-dimensional vector.

`adaptive_diversity`

- Starts with early utility, then gradually shifts toward novelty/freshness.
- Uses `t / max_t` rather than true error, so it is deployable.
- Scientifically motivated, but not the current best in tested runs.

## Control-Plane Arrangement In A Real System

A plausible real deployment would separate control signaling from FL update
payloads.

### 1. BS Broadcast

The BS broadcasts:

```text
round_id
M
pcomp or access class assumptions
R_D2D / power class
Cmax
CH score weights
utility/access policy parameters
allocator parameters
optional water level or normalizer
optional density/throughput trigger
```

### 2. D2D Discovery

Devices exchange compact beacons:

```text
temporary device id
battery class
BS channel-quality class
D2D discovery signal
```

Devices estimate local D2D neighbors.

### 3. Local Cluster Formation

Candidate CHs and members exchange local request/accept messages.  A cluster is
accepted only if the elected CH can cover all members in one hop and `Cmax` is
not exceeded.

### 4. Cluster Summary Reporting

CHs report compact summaries:

```text
CH id or temporary id
cluster size
clusterized/singleton count contribution
optional channel/battery class
optional utility summary
```

The BS estimates:

```text
number of active CHs
clusterized device fraction
expected fixed-D2D CH successes
```

### 5. CH Local Access Decision

Each CH computes:

```text
aggregate update
norm
active cluster size
freshness
utility
ALOHA probability
```

Then it makes an independent random ALOHA attempt.

### 6. ACK Feedback

The BS observes successful CH uploads and updates:

```text
freshness counters
optimized success EWMA
throughput ratio
optional reference direction
```

These feedback signals are sufficient for utility, conditional water-filling,
hybrid, and adaptive-diversity modes.  They do not require the BS to know the
true optimization error.

## Current Best Enhanced Configuration

The current best tested enhanced architecture for `K=3000` with channel-aware
CH-to-BS success is:

```text
dense geometric clustering
quality CH election
CH weights = 0.1 / 0.8 / 0.1
d2d_ch_bs_success_mode = channel_quality
utility optimized D2D access
conditional selective water-filling
density-aware trigger
```

Representative command shape:

```bash
python main.py --run-name <name> \
  --devices 3000 \
  --rounds 100 \
  --precision float64 \
  --cluster-head-selection-mode quality \
  --cluster-head-degree-weight 0.1 \
  --cluster-head-channel-weight 0.8 \
  --cluster-head-battery-weight 0.1 \
  --d2d-ch-bs-success-mode channel_quality \
  --d2d-ch-bs-min-success-probability 0.35 \
  --d2d-ch-bs-pathloss-exponent 2.0 \
  --d2d-ch-bs-battery-exponent 0.25 \
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

## Scientific Comparison Rules

Use paired runs when testing one architectural change:

```text
same K
same rounds
same seed range
same precision
same checkpoints
same physical-link mode
only one strategy component changed
```

Recommended metrics:

```text
t_to_1e-6
t_to_1e-9
t_to_1e-12
log-error AUC / logsum
final error norm
CH upload ratio
device upload gain
clusterized device rate
```

Important comparison warning:

- Runs with `--d2d-ch-bs-success-mode channel_quality` are not directly
  comparable to old thesis-compatible collision-only runs.
- Direct non-D2D uploads are still more idealized unless a future
  device-to-BS channel-aware model is enabled.

## Current Limitations

The current architecture is useful for research iteration, but these limits
should be stated clearly:

- Battery is static; there is no energy drain model yet.
- CH-to-BS channel quality uses distance/pathloss only.
- Direct device-to-BS links are not yet channel-aware in the same way as D2D
  CH-to-BS links.
- Member-to-CH D2D link success is a global probability, not a per-link channel
  model.
- No fading, shadowing, interference beyond ALOHA collisions, modulation, or
  coding model is included.
- No mobility is included.
- No non-IID data or data-similarity-aware clustering is included.
- The BS does not model control-plane overhead explicitly.
- Freshness is based on successful CH uploads, not semantic information age of
  labels or data distribution.

## Best Next Architecture Improvements

The most productive next implementation steps are:

1. Add channel-aware direct device-to-BS success for non-D2D curves, so D2D and
   non-D2D physical-layer comparisons are fair.
2. Add battery drain for CH duty, D2D member transmission, and direct BS
   transmission.
3. Make member-to-CH D2D link success depend on D2D distance or D2D channel
   quality.
4. Add noisy or delayed cluster-summary reporting to test whether the
   density-aware allocator remains robust.
5. Add a stronger CH rotation policy that balances BS channel gain against
   long-term CH duty/fairness.
