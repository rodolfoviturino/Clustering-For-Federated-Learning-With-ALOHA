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
     freshness, load control, and channel-aware CH-to-BS/device-to-BS decoding
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

`battery` is static by default, which preserves comparability with the original
thesis-style curves. Enhanced runs can enable:

```text
--energy-drain-mode dynamic
```

With dynamic drain enabled, the simulator stores one battery vector for each of
the six curves. Direct device-to-BS attempts, D2D member transmissions, and
CH-to-BS aggregate attempts consume configurable normalized energy. Battery can
then affect later decoding probability through the existing battery exponents.
This is still a coarse energy abstraction, not a calibrated radio power model.

## D2D Clustering

The main clustering strategy is dense geometric one-hop D2D:

```text
dense pair-first clustering
+ singleton join repair
+ pair CH-rotation repair
+ CH-to-CH merge repair
+ optional local max-size split
```

The invariant is:

```text
every cluster member must be within R_D2D of the cluster head
cluster size must be <= Cmax
```

This is deliberately one-hop.  It avoids assuming a multi-hop D2D routing
protocol that the thesis did not model.

`--cluster-split-mode max_size` is the first structural follow-up after the
pure probability-shaping member-freshness ablations. It re-clusters large D2D
rows into valid one-hop subclusters before the FL simulation starts. This is
not MAC scheduling: CHs still contend through multichannel ALOHA, but the
member-to-CH grouping changes.
`--cluster-split-mode safe_max_size` is the selective follow-up. It keeps the
same pre-FL structural interpretation but only considers a budgeted fraction of
oversized rows and rejects any local split that would emit a subcluster smaller
than `--cluster-split-min-subcluster-size`. The first size-only safe split
point is structurally controlled but not beneficial, so future structural
selection should use member freshness/participation pressure rather than size
alone.
`--cluster-split-mode pressure_safe_max_size` is that next deployable proxy: it
ranks oversized rows by member-to-CH distance pressure and CH-to-BS channel
pressure before applying the same safe split guard.

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
- CH election in this section is the initial post-clustering choice.  Dynamic
  intra-run CH re-election is a separate optional policy documented below.  The
  separation is intentional: a run can test communication-quality CH selection
  without also changing the CH over time.

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

D2D member participation has three optional realism knobs:

```text
--d2d-member-compute-probability
--d2d-member-link-success-probability
--d2d-member-link-success-mode
```

Defaults:

```text
d2d_member_compute_probability = 1.0
d2d_member_link_success_probability = 1.0
d2d_member_link_success_mode = constant
```

Meaning:

- compute probability models whether a member produced its local update;
- `constant` link success mode models D2D delivery with one scalar probability
  shared by all member-to-CH links;
- `rayleigh_outage` link success mode computes a per-member probability from
  the distance between that member and the current CH:

  ```text
  avg_snr_i,h = reference_snr / max(d_i,h, 1)^pathloss_exponent
  q_i,h       = exp(-snr_threshold / avg_snr_i,h)
  ```

When either value is below `1.0`, a successful CH-to-BS upload carries:

```text
CH own update
+ updates from members that computed successfully
+ updates from non-CH members whose D2D link succeeded
```

The CH itself is always considered active in its own cluster. Other members
must pass compute availability, optional battery feasibility, and the
member-to-CH link draw. With dynamic CH rotation, `rayleigh_outage` recomputes
the member-to-CH distance to the elected CH for each D2D scenario, so rotation
can improve or degrade aggregate completeness even when cluster membership is
fixed.

This is still a collision-free physical-decoding abstraction. Intra-cluster
D2D interference, packet retransmissions, coding, and detailed MAC timing are
not yet modeled.

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

The same physical decoding model can now be applied to direct non-D2D uploads
with:

```text
--device-bs-success-mode channel_quality
```

When both flags are enabled, the comparison is physically symmetric:

```text
--d2d-ch-bs-success-mode channel_quality
--device-bs-success-mode channel_quality
```

The D2D curves still differ because the transmitter is the elected CH and the
payload may aggregate multiple active members.  The non-D2D curves use the same
ALOHA contention model, but the transmitter is each individual device.

Current limitation:

- The channel-aware success model is still a distance/pathloss proxy.
- It does not include explicit SINR, fading, shadowing, coding rate, or
  inter-cell interference.

## Dynamic Energy Drain

Dynamic drain is an optional enhanced ablation, not a thesis-compatible default.
It is enabled only with:

```text
--energy-drain-mode dynamic
```

The model tracks normalized battery in `[0, 1]` independently for each curve:

```text
0 polling
1 fixed ALOHA
2 optimized ALOHA
3 polling with D2D
4 fixed ALOHA with D2D
5 optimized ALOHA with D2D
```

This separation matters. The six curves are counterfactual policies evaluated
on the same generated deployment; one policy should not drain the battery state
of another policy.

Energy costs are charged on attempts:

```text
direct device-to-BS attempt -> energy_direct_bs_cost
active non-CH member sends local update to CH -> energy_d2d_member_cost
CH sends aggregate to BS -> energy_ch_bs_cost
```

An attempt pays energy even when it later collides or fails decoding. That is
the realistic ordering: the radio consumes energy before the transmitter learns
whether the packet was useful. The runner records:

```text
<scenario>_battery_mean
<scenario>_battery_ci95
<d2d_scenario>_clusterhead_battery_mean
<d2d_scenario>_clusterhead_battery_ci95
<scenario>_energy_used_mean
<scenario>_energy_used_ci95
<scenario>_energy_efficiency_mean
<scenario>_energy_efficiency_ci95
<d2d_scenario>_clusterhead_energy_used_mean
<d2d_scenario>_clusterhead_energy_used_ci95
```

`energy_efficiency` is cumulative successful uploads divided by normalized
battery consumed by that scenario.  It is not a physical bit-per-joule metric;
it is a simulation-level measure for comparing policies under the same energy
cost coefficients.

The plotter writes `results_battery.*`, `results_clusterhead_battery.*`,
`results_energy_used.*`, `results_energy_efficiency.*`, and
`results_clusterhead_energy_used.*` when those columns exist.

### Real Deployment Interpretation

A practical implementation would not need centralized per-device scheduling for
this layer. Each device can maintain its own battery estimate locally. The BS
can broadcast energy-cost coefficients or policy parameters, and CHs can report
coarse battery classes during cluster formation or periodic control windows.
ACKs from successful CH uploads are already needed by the throughput-aware load
controller; energy drain itself is local bookkeeping.

## Dynamic D2D CH Rotation

Dynamic CH rotation is optional and off by default:

```text
--d2d-ch-rotation-mode static
```

The enhanced mode is:

```text
--d2d-ch-rotation-mode energy_aware
```

It requires:

```text
--energy-drain-mode dynamic
```

The reason is practical: if battery is not evolving, an energy-aware rotation
policy has no current-energy state to react to.  When enabled, the simulator
keeps one CH vector per D2D curve:

```text
polling with D2D CHs
fixed ALOHA with D2D CHs
optimized ALOHA with D2D CHs
```

The trigger is controlled separately from the score:

```text
--d2d-ch-rotation-trigger-mode interval
--d2d-ch-rotation-trigger-mode aoi
--d2d-ch-rotation-trigger-mode interval_or_aoi
```

`interval` is the original periodic behavior. `aoi` rotates only clusters whose
AoI is in the stale tail for that D2D scenario. `interval_or_aoi` performs a
periodic rotation and also reacts to stale clusters between periodic rotations.
The stale threshold is:

```text
--d2d-ch-rotation-aoi-threshold-fraction
```

It is interpreted as a fraction of the current maximum active-cluster AoI in
that scenario.  The initial AoI value does not trigger rotation, avoiding a
cold-start mass re-election.

When the trigger activates, each D2D curve re-elects a CH inside each selected
existing cluster.  Membership does not change.  The candidate must:

```text
be a member of the cluster
and cover every cluster member within R_D2D
and keep the cluster size unchanged
```

The score is:

```text
score_i =
  channel_weight * normalized_bs_channel_quality_i
  + battery_weight * current_battery_i
  + stability_weight * is_current_ch_i
```

Profiles:

```text
performance -> channel 0.85, battery 0.10, stability 0.05
balanced    -> channel 0.65, battery 0.25, stability 0.10
eco         -> channel 0.45, battery 0.45, stability 0.10
```

The stability term avoids unnecessary CH flips when two candidates are nearly
equivalent.  The selected CH is always locally active in its cluster.  Other
members still depend on `--d2d-member-compute-probability` and
`--d2d-member-link-success-probability`.  CH-to-BS probability, ALOHA draw,
aggregate upload, and CH energy drain all use the current elected CH, not the
original column-0 CH.

### Real Deployment Interpretation

This remains a distributed/plausible policy.  The cluster can run a local
control mini-round every rotation interval or when a cluster's ACK age crosses
the AoI threshold.  Members advertise coarse battery and BS-channel class, or
the current CH polls them.  A candidate only needs neighbor measurements or
received control packets to prove one-hop coverage of the current member set.
The BS can broadcast the profile weights, interval, and AoI threshold; it does
not centrally schedule a specific CH for every ALOHA slot.

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

`aoi_aware_utility`

- Keeps the current utility/load-controller architecture.
- Adds a bounded bonus for clusters in the stale AoI tail.
- Uses ACK/no-ACK age, not true optimization error, so it remains deployable.
- Intended to test whether optimized D2D can keep its error/energy advantage
  while reducing mean and p95 AoI versus the current utility policy.

`aoi_floor_utility`

- Keeps the same base utility probability as `utility`.
- Raises only stale-tail clusters to a bounded minimum probability.
- Interprets `aoi_weight` as a fraction of fixed-D2D access probability, not as
  a multiplicative utility boost.
- Is intended as the lower-risk AoI strategy after the stronger multiplicative
  AoI-aware policy reduced mean AoI but hurt error/energy in the first test.

`aoi_tail_utility`

- Keeps the same base utility probability as `utility`.
- Computes a second probability from stale-tail AoI pressure only.
- Interprets `aoi_weight` as a reserved load-budget quota clipped to `[0, 1]`.
- Returns exact base utility when active clusters have no differentiated
  stale-tail AoI.
- Is intended to attack p75/p90/p95 AoI directly after the AoI bonus/floor and
  AoI-triggered CH rotation tests did not clear the stale tail.

`aoi_quality_tail_utility`

- Keeps the same base utility probability and reserved quota structure as
  `aoi_tail_utility`.
- Changes the reserved-tail utility from AoI pressure alone to AoI pressure
  multiplied by CH-to-BS success probability and current CH battery.
- Uses the selected physical channel model, so with `rayleigh_outage` the
  channel term is directly tied to outage probability rather than an arbitrary
  distance score.
- Is intended to test whether AoI tail relief can be made less wasteful:
  stale clusters still get attention, but stale clusters with poor CH channel
  or depleted CH energy do not receive the same reserved probability.
- Remains a distributed ALOHA policy. The BS can broadcast exponents and
  normalizers; each CH computes its own score from local AoI, channel estimate,
  and battery before drawing its local ALOHA attempt.

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
CH-to-BS success uses strongly channel-heavy CH election:

```text
dense geometric clustering
quality CH election
CH weights = 0.0 / 1.0 / 0.0
d2d_ch_bs_success_mode = channel_quality
device_bs_success_mode = channel_quality for fair non-D2D comparison
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
- If D2D uses `--d2d-ch-bs-success-mode channel_quality`, use
  `--device-bs-success-mode channel_quality` when the goal is to compare D2D
  and non-D2D curves under the same physical-link abstraction.
- The same rule applies to the enhanced Rayleigh path: if D2D uses
  `--d2d-ch-bs-success-mode rayleigh_outage`, use
  `--device-bs-success-mode rayleigh_outage` for fair D2D versus non-D2D
  comparisons. Use `--cluster-head-channel-score-mode rayleigh_outage` when
  the CH-election channel weight should map to the same outage probability.
- Results using `--energy-model first_order_radio` and
  `--battery-feasibility-mode required_energy` are not directly comparable to
  older constant-cost dynamic-energy runs unless the metadata is reported and
  the energy objective is explicitly part of the comparison.

## Current Limitations

The current architecture is useful for research iteration, but these limits
should be stated clearly:

- Battery is static by default for thesis-compatible reproduction. Dynamic
  energy drain, first-order radio energy, battery feasibility, optional
  CH-rotation control overhead, and energy-aware intra-run CH rotation are
  implemented as explicit enhanced ablations.
- The first-order radio model is normalized, not calibrated in joules for a
  specific radio chipset. There is no recharge, sleep-state, idle listening, or
  MAC-control timing model.
- CH-to-BS and direct device-to-BS links can use either inverse-pathloss
  `channel_quality` or Rayleigh outage probability. There is still no shadowing,
  explicit SINR with co-channel interference powers, modulation, coding, FER,
  or BER model.
- Member-to-CH D2D link success can use a scalar probability or per-link
  Rayleigh outage based on distance to the current CH. It still does not model
  D2D interference powers, retransmissions, modulation, coding, FER, or BER.
- ALOHA same-channel collisions remain the interference abstraction. Rayleigh
  outage models collision-free physical decoding only.
- No mobility is included.
- No non-IID data or data-similarity-aware clustering is included.
- The BS does not model full control-plane overhead explicitly. The only
  control overhead currently exposed is the optional CH-rotation control cost.
- AoI is now tracked explicitly as mean, peak, percentile, and stale-tail
  metrics. It drives scheduling only in opt-in policies such as
  `aoi_aware_utility`, `aoi_floor_utility`, `aoi_tail_utility`, or
  `aoi_quality_tail_utility`; it remains information-age over successful
  uploads, not semantic freshness of labels or data distribution.

## Best Next Architecture Improvements

The most productive next implementation steps are summarized in
`docs/current_status_and_future_work.md`.  In priority order, they are:

1. Calibrate the normalized first-order radio coefficients against a real IoT
   radio or a literature parameter table, then report the calibration source.
2. Validate and calibrate the new per-link D2D Rayleigh outage mode against
   realistic D2D ranges/SNR thresholds, then report sensitivity.
3. Extend Rayleigh outage toward an SINR/PER abstraction if co-channel
   interference power, modulation, and coding need to be represented.
4. Rerun the physical `utility` baseline with current p75/p90/p95 and
   stale-fraction AoI columns before treating any AoI-enhanced run as a fair
   improvement.
5. Move stale-tail work beyond access probability if the refreshed baseline
   confirms the same tail problem: per-link D2D outage validation,
   AoI/channel-aware CH rotation, or re-clustering are more likely to address
   structural stale clusters than more access-score tuning.
6. Run structured sensitivity sweeps for utility exponents, allocator
   thresholds, CH-rotation weights, energy coefficients, and Rayleigh SNR
   thresholds.

For confirming the current channel-heavy CH election choice, use the focused
runner instead of manually comparing long one-off commands:

```bash
python -m experiments.run_ch_quality_weight_sweep --run-name k3000_ch_quality_finalists --devices 3000 --rounds 200
```

The default `finalists` profile compares:

```text
w001000_channel_only   -> degree/channel/battery = 0.0 / 1.0 / 0.0
w009010_channel_battery -> degree/channel/battery = 0.0 / 0.9 / 0.1
```

It uses the current enhanced defaults: channel-aware CH-to-BS decoding,
channel-aware direct device-to-BS decoding for fair figures, utility
optimized-D2D access, density-aware conditional selective water-filling, and
`float64` precision.  The runner writes `ch_quality_weight_summary.csv`,
`ch_quality_weight_top.md`, `ch_quality_weight_best_fair_error_norm.*`, and
`ch_quality_weight_optimized_d2d_error_norm.*`.
