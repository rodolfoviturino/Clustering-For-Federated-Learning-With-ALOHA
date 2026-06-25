# Current Status And Future Work

This document records what the current simulator already implements, what each
piece means in the research architecture, and which modeling upgrades should be
prioritized next.  It is intentionally written as a technical status record, not
as a conversational response.

## Scope

The project now has two experiment layers:

```text
thesis-compatible layer
  -> preserves the original HFL + multichannel ALOHA + D2D one-hop clustering
     experiment structure as the reference baseline

enhanced research layer
  -> adds physically motivated ablations for faster and more realistic
     experiments: utility access, channel-aware CH election, dynamic energy,
     CH rotation, and extra diagnostic metrics
```

Enhanced runs must be reported with their full metadata.  They are not
byte-exact reproductions of the original thesis figures, and they should not be
mixed with thesis-compatible runs without explicitly naming the extra modeling
assumptions.

## Implemented Components

### GPU/CPU JAX Experiment Backend

The main experiment path is:

```text
main.py
  -> experiments/run_gpu_sweep.py
  -> Clustering/jax_clustering_algorithm.py
  -> Models/jax_models_arrangement.py
  -> Runs/<run_name>/
```

What is implemented:

- JAX device generation and simulation on CPU/GPU.
- Full `.py` experiment runner independent of notebooks.
- Automatic run folders under `Runs/`.
- CSV, metadata JSON, PNG, and PDF outputs.
- Full-curve checkpoint output by default, not only start/end points.
- Colab-compatible execution when JAX sees a CUDA GPU.

Why it matters:

- The notebook is no longer a required execution path.
- The same command can run locally on Windows CPU for smoke tests and on Colab
  GPU for larger sweeps.
- Run metadata preserves parameters needed for scientific comparison.

### D2D Clustering

The current clustering path is dense geometric one-hop D2D clustering:

```text
dense pair-first clustering
+ singleton join repair
+ pair CH-rotation repair
+ CH-to-CH merge repair
+ optional local max-size split
```

Invariants:

```text
every member is within R_D2D of the elected CH
cluster size <= Cmax
singleton fallback is preserved
polling+D2D compute/energy feasibility is evaluated on the scheduled CH
```

Real-world interpretation:

- Devices can discover D2D neighbors with local beacons.
- Candidate CHs can accept members only if one-hop coverage is valid.
- The BS does not need to solve a global assignment.
- The BS may receive compact cluster summaries after local formation.

Current result:

- Dense clustering increased coverage substantially compared with earlier
  geometric variants.
- For `K=3000`, clusterized-device rate has been around `99.319%` in the
  tested enhanced runs.

Current physical-link status:

- Member-to-CH link success can be the thesis-compatible global scalar or the
  enhanced per-link Rayleigh outage probability based on distance to the
  current CH.
- It still does not model local D2D SINR, explicit interference powers,
  retransmissions, coding, or MAC timing.

### Channel-Aware CH Election

The enhanced CH election mode keeps cluster membership fixed and changes only
which member acts as CH.

Current score:

```text
score_i =
  degree_weight  * normalized_D2D_degree_i
  + channel_weight * normalized_BS_channel_quality_i
  + battery_weight * normalized_battery_i
```

What is implemented:

- `--cluster-head-selection-mode quality`
- configurable degree/channel/battery weights
- one-hop coverage validation for any elected CH

Current empirical conclusion:

- Once CH-to-BS physical success is enabled, channel-heavy CH election becomes
  the strongest tested direction.
- The best tested CH election for `K=3000` has been channel-first/channel-only
  in the enhanced setup.

Remaining limitation:

- `normalized_BS_channel_quality_i` is currently an inverse-pathloss proxy.
- It is not yet a direct outage probability, SINR, FER/PER, or BER metric.

Future upgrade:

```text
replace normalized_BS_channel_quality_i with q_i

q_i = P(successful decoding | channel, power, interference, threshold)
```

Possible first model:

```text
SINR_i = P_tx_i * g_i / (N0 + interference)
q_i    = P(SINR_i >= gamma_threshold)
```

Then the CH score becomes easier to defend:

```text
score_i =
  channel_weight * q_i
  + battery_weight * residual_energy_i
  + stability_weight * is_current_ch_i
```

### Channel-Aware Decode Probability

Current decode probability:

```text
q_i =
  min_success
  + (1 - min_success)
    * channel_quality_i
    * battery_i^battery_exponent
```

What is implemented:

- optional CH-to-BS channel-aware success;
- optional direct device-to-BS channel-aware success;
- same physical abstraction can be applied to D2D and non-D2D curves for fair
  comparisons.

Current limitation:

- The battery multiplier is a heuristic.  It loosely represents battery-limited
  transmit capability, but it is not the cleanest physical model.
- It is better to treat battery as an energy-availability constraint:

```text
if residual_energy_i < E_required_for_attempt:
  node cannot transmit this update
else:
  success probability depends on channel/SINR/PER, not directly on battery
```

Recommended future change:

- Keep `battery_exponent=0` or de-emphasize it in main experiments.
- Implement energy-gating based on required transmit/receive/aggregation energy.
- Let channel success depend on pathloss/fading/SINR/outage.

### Dynamic Energy Drain

Dynamic energy is implemented but intentionally coarse.

Current mode:

```text
--energy-drain-mode dynamic
```

Current normalized costs:

```text
direct device-to-BS attempt -> energy_direct_bs_cost
active D2D member to CH     -> energy_d2d_member_cost
CH aggregate to BS          -> energy_ch_bs_cost
```

What is implemented:

- one independent battery vector per counterfactual curve;
- energy charged on attempts, not only successful packets;
- CH energy and mean CH battery metrics;
- energy-used and energy-efficiency metrics;
- plots for battery, CH battery, energy used, energy efficiency, and CH energy.

Why separate battery vectors are used:

The six curves are counterfactual policies on the same generated deployment.
The polling curve should not drain the battery state used by the optimized-D2D
curve.  Each policy receives its own battery trajectory.

Current result:

- Energy-aware CH rotation substantially preserves CH battery.
- In the replicated `K=3000` tests, `energy_performance` preserved roughly
  `+0.108` normalized CH battery versus static CHs while keeping final error
  and total energy close to static.

Current limitation:

- Costs are normalized constants, not a calibrated radio-power model.
- CH receive energy is not yet separated from CH transmit energy.
- Aggregation/computation energy is not explicitly modeled.
- Control-plane overhead for discovery, CH rotation, and reporting is not yet
  charged.

Recommended future physical model:

```text
E_tx(l, d) = l * E_elec + l * epsilon_amp * d^alpha
E_rx(l)    = l * E_elec
E_agg(l)   = l * E_DA
```

Then:

```text
member cost = E_tx(update_bits, d_member_to_ch)

CH cost =
  sum_members E_rx(update_bits)
  + E_agg(active_member_count * update_bits)
  + E_tx(aggregate_bits, d_ch_to_bs)

direct device cost =
  E_tx(update_bits, d_device_to_bs)
```

This structure matches the common first-order radio-energy abstraction used in
wireless sensor network clustering work such as LEACH-style analyses.

### Dynamic D2D CH Rotation

Current mode:

```text
--d2d-ch-rotation-mode energy_aware
```

Requirements:

```text
--energy-drain-mode dynamic
```

Current profile scores:

```text
score_i =
  channel_weight * normalized_bs_channel_quality_i
  + battery_weight * current_battery_i
  + stability_weight * is_current_ch_i
```

Implemented profiles:

```text
performance -> channel 0.85, battery 0.10, stability 0.05
balanced    -> channel 0.65, battery 0.25, stability 0.10
eco         -> channel 0.45, battery 0.45, stability 0.10
```

What is implemented:

- periodic CH re-election inside each fixed cluster;
- optional AoI-triggered CH re-election for clusters in the stale tail;
- re-election per D2D scenario, not globally shared between scenarios;
- one-hop coverage validation for the new CH;
- dynamic CH identity used for ALOHA, CH-to-BS success, aggregation, and energy
  drain.

Trigger modes:

```text
interval         -> previous periodic behavior
aoi              -> rotate only stale-tail clusters
interval_or_aoi  -> periodic rotation plus extra stale-tail rotation
```

The AoI trigger normalizes cluster AoI within each D2D scenario and rotates
clusters whose AoI is above `--d2d-ch-rotation-aoi-threshold-fraction` of the
current maximum active-cluster AoI.  It does not rotate on the initial AoI
value, so it avoids a cold-start mass rotation.

Current empirical conclusion:

- `K=1000`: `eco` and `balanced` are attractive energy tradeoffs; `eco`
  preserves the most CH battery.
- `K=3000`: `performance` is the most defensible profile.  It is channel-first,
  preserves much more CH battery than static CHs, and keeps error/energy close
  to static.  More battery-heavy profiles preserve more CH battery but degrade
  optimized-D2D error.

Current limitation:

- CH-rotation control overhead is optional and defaults to zero.
- The interval and profile weights are still policy parameters that need
  sensitivity analysis.
- The AoI trigger still uses the same channel/battery/stability CH score; it
  does not yet have a separate score specifically optimized for stale clusters.

### Optimized D2D Utility Access

Current enhanced utility:

```text
utility_h =
  norm_h^norm_exp
  * active_cluster_size_h^size_exp
  * freshness_h^freshness_exp
```

What is implemented:

- `utility`
- `max_weight`
- `hybrid`
- `adaptive_diversity`
- `aoi_aware_utility`
- `aoi_floor_utility`
- `aoi_tail_utility`
- `aoi_quality_tail_utility`
- `proportional_clip`
- `water_filling`
- `selective_water_filling`
- `conditional_selective_water_filling`
- density-aware conditional trigger

Current recommended allocator:

```text
conditional_selective_water_filling
```

Why:

- `proportional_clip` is stable in dense networks but can waste clipped load.
- full `water_filling` can over-redistribute and increase collision pressure.
- conditional selective water-filling uses ACK-observable throughput to decide
  whether redistribution is needed.
- in dense regimes, the trigger can become conservative through the
  clusterized-device fraction.

Current limitation:

- `norm_exp`, `size_exp`, `freshness_exp`, `load_target_factor`, and
  redistribution parameters are policy hyperparameters.
- They are not universal constants and should be reported as tuned/swept
  experiment parameters.

Required validation:

- sensitivity sweeps;
- confidence intervals;
- paired seeds;
- separate evaluation by `K`, density, channel model, and energy model.

### Freshness

Current implementation:

```text
freshness_h = rounds since CH h last uploaded successfully
```

What is implemented:

- freshness affects optimized-D2D utility;
- freshness resets after successful optimized-D2D CH upload;
- freshness does not use true model error, so it is deployable.

Current limitation:

- It is a useful scheduling signal, but it is not yet reported as a formal
  Age-of-Information metric in older pre-AoI result folders.

Current AoI metric:

```text
AoI_h(t + 1) =
  1,              if cluster h's update is received by the BS at t
  AoI_h(t) + 1,   otherwise
```

Report:

```text
mean AoI
peak AoI
p75 AoI
p90 AoI
p95 AoI
stale fraction above 1 + 50 percent of elapsed t
stale fraction above 1 + 75 percent of elapsed t
stale fraction above 100 rounds
member-level mean/p75/p90/p95/peak AoI for non-singleton D2D clusters
member zero-participation fraction for non-singleton D2D clusters
AoI distribution over clusters
AoI versus error norm
AoI versus energy efficiency
```

The `aoi_aware_utility` policy now uses AoI as an explicit objective by adding
a bounded stale-tail bonus to the existing utility score.  This makes the
freshness weight scientifically easier to analyze because the run can be judged
on error, energy, mean AoI, p75/p90/p95 AoI, stale-tail fractions, and peak AoI
together.  The initial multiplicative AoI-aware test improved mean AoI but did
not improve the full error/energy tradeoff, so the code also includes
`aoi_floor_utility`, a conservative variant that preserves base utility access
and only gives very stale clusters a bounded minimum probability. The next
continuations are `aoi_tail_utility`, which reserves a bounded share of the
load budget for the differentiated stale AoI tail, and
`aoi_quality_tail_utility`, which spends that reserved tail budget only after
also considering CH-to-BS success probability and current CH battery.

## Current Main Limitations

The current enhanced simulator is useful for research iteration, but the
following points should be explicitly disclosed:

- the first-order radio model is normalized and not yet calibrated in joules
  for a specific IoT transceiver;
- CH receive, aggregation, and BS-transmit costs are separated in the enhanced
  first-order model, but idle listening, sleep states, retransmissions, and full
  MAC control overhead are still absent;
- optional CH rotation control overhead exists, but defaults to zero and is not
  a full control-plane traffic model;
- battery can now be used as an energy-feasibility constraint, but the old
  battery exponent remains as a legacy heuristic for backward-compatible
  channel-quality runs;
- channel quality can use inverse pathloss or Rayleigh outage probability, but
  there is no explicit SINR/FER/PER model with interference powers, modulation,
  coding, or shadowing;
- member-to-CH link success can now be modeled either as the legacy scalar
  probability or as per-link Rayleigh outage based on distance to the current
  CH; it still omits D2D interference powers, retransmissions, coding, and
  MAC timing;
- utility exponents and allocator thresholds are hyperparameters;
- AoI is now measured explicitly and can drive `aoi_aware_utility`,
  `aoi_floor_utility`, `aoi_tail_utility`, `aoi_quality_tail_utility`, or
  `member_fair_utility`, but the weight, exponent, stale-tail threshold,
  quota, quality exponents, and member-fairness pressure are policy
  hyperparameters that need sensitivity analysis;
- D2D member-level AoI and zero-participation metrics are now diagnostics over
  devices in non-singleton D2D clusters only; they do not alter scheduling
  unless `member_fair_utility` or a future policy explicitly consumes them;
- data are synthetic linear-regression data, not non-IID task data.

## Current Empirical Findings

The most important current results are:

- Dense geometric clustering is now strong enough that `K=3000` reaches about
  `99.319%` clustered devices in the representative enhanced runs.
- Tuned `utility` optimized-D2D remains the best general access policy tested
  so far. The strongest hyperparameter neighborhood uses:

  ```text
  floor = 0.02
  norm_exp = 3.5
  size_exp = 1.5
  freshness_exp = 0.25
  load_target_factor = 1.1
  ```

- Allocator behavior depends on density. Partial redistribution helps some
  `K=1000` runs, but dense `K=3000` runs often prefer proportional clipping.
  The density-aware conditional allocator is currently the most defensible
  cross-density compromise.
- Under physical CH-to-BS decoding, channel-heavy CH election is the strongest
  tested CH election direction. In `k3000_ch_quality_fair_r200`, channel-only
  CH election produced logsum `-1088.656`, t200 D2D error `3.981e-12`, and a
  D2D/direct t200 log10 gain of about `6.978`.
- Energy-aware CH rotation is useful mainly as an energy/battery tradeoff. In
  `k3000_energy_rotation_profiles_r100`, the `performance` profile preserved
  about `+0.108` normalized CH battery versus static and kept energy close to
  static while slightly improving t200 error.
- AoI-aware access policies improved mean AoI but did not improve the full
  Pareto tradeoff. In the `K=1000` physical-energy comparison:

  ```text
  utility:
    t_to_1e-12 = 176
    t200 error = 2.251e-14
    energy_efficiency = 802.729
    mean AoI = 134.828

  aoi_aware_utility:
    t_to_1e-12 = 182
    t200 error = 5.953e-14
    energy_efficiency = 765.165
    mean AoI = 130.751

  aoi_floor_utility:
    t_to_1e-12 = 193
    t200 error = 2.784e-13
    energy_efficiency = 741.292
    mean AoI = 128.172
    p75/p90/p95 AoI = 201/201/201
  ```

  This means the first AoI access policies should be kept as ablations, not
  promoted as the main optimized-D2D strategy.

- AoI-triggered CH rotation was then tested as a CH-side correction. In
  `k1000_energy_rotation_aoi_profiles_r100`, it did not materially improve
  AoI: final mean AoI stayed around `161`, p90 AoI stayed at `201`, and stale75
  stayed around `72%`. The strongest result in that sweep was still the
  energy/battery tradeoff from `energy_eco` and `energy_balanced`, not the AoI
  trigger itself.

- The implementation now supports the next access-side test through
  `--optimized-d2d-access-mode aoi_tail_utility`. Unlike the older AoI bonus or
  floor, this mode reserves a real quota of the load budget for stale-tail
  clusters while returning exact base utility when no differentiated stale tail
  exists. It is specifically meant to test whether p75/p90/p95 AoI can be
  reduced without discarding the tuned utility policy.

- The next implemented refinement is
  `--optimized-d2d-access-mode aoi_quality_tail_utility`. It keeps the same
  stale-tail quota idea, but weights the reserved tail budget by CH-to-BS
  success probability and current CH battery. This is a more physical
  continuation: if AoI quota sends probability to a stale cluster whose CH has
  poor Rayleigh-outage success or depleted battery, the slot is unlikely to
  reduce either error or AoI. The policy is still deployable because the CH uses
  local battery and channel/ACK-derived success estimates, while the BS only
  broadcasts scalar exponents and normalizers.

- The first `aoi_quality_tail_utility` runs showed a marginal but informative
  tradeoff, not a new main strategy. With `w=0.10` and `threshold=0.90`, quality
  gating improved the pure AoI-tail run on error, uploads, energy, mean AoI,
  and `stale75`, but still did not reach `1e-12` by `t=200`. With the lighter
  `w=0.05` and `threshold=0.85`, it preserved `t_to_1e-12=190` and slightly
  reduced mean AoI and `stale75`, but it slightly worsened log-error AUC and
  `t_to_1e-9`. In both cases p75/p90/p95 AoI stayed at the horizon value
  `201`, so severe stale-tail AoI remains unresolved by access-probability
  reweighting alone.

- The first `member_fair_utility` smoke run directly attacked the new
  member-level diagnosis. In `p0_member_fair_v1-02` (`K=1000`, `rounds=20`,
  `t=100`, ideal D2D member links), compared with the default optimized-D2D
  `norm` run `p0_member_aoi_curve_v3`:

  ```text
  final optimized-D2D error:
    norm                1.295e-07
    member_fair_utility 5.192e-07

  member mean AoI:
    norm                63.409
    member_fair_utility 44.706

  member stale75:
    norm                0.509
    member_fair_utility 0.205

  member zero-participation fraction:
    norm                0.451
    member_fair_utility 0.132
  ```

  This is the strongest current evidence that the optimized-D2D issue is not
  merely aggregate AoI: CH-level utility can converge well while starving
  non-CH members. The caveat is that `member_p95_aoi` still reached the horizon
  value in the smoke run, so the next step is a small sensitivity sweep over
  `optimized_d2d_aoi_weight`, threshold, and exponent rather than claiming the
  tail is solved.

- The `rounds=100` sensitivity runs confirm the same ideal-link trend. Against
  `p0_member_aoi_curve_v3` (`norm`, `final_error=1.295e-07`,
  `member_aoi=63.409`, `member_stale75=0.509`, `member_zero=0.451`):

  ```text
  member_fair_w015_thr070_r100:
    final_error  = 4.018e-07
    member_aoi   = 46.451
    member_stale75 = 0.241
    member_zero  = 0.163

  member_fair_w025_thr070_r100:
    final_error  = 8.758e-07
    member_aoi   = 45.010
    member_stale75 = 0.211
    member_zero  = 0.134
  ```

  The current Pareto reading is therefore conservative: `w=0.15` is the better
  default candidate for papers because it keeps the final error below `5e-7`
  while removing most zero-participation starvation; `w=0.25` is a stronger
  fairness ablation with a larger convergence cost.

- The refreshed physical energy/Rayleigh comparison now includes member-level
  columns for the `utility` baseline. It confirms a smooth freshness/cost
  tradeoff at low weights and a sharp convergence penalty at higher weights:

  ```text
  utility_physical_member_metrics_r100:
    final_error       = 1.603e-07
    cluster_aoi       = 80.655
    member_aoi        = 62.043
    member_stale75    = 0.483
    member_zero       = 0.431
    energy_efficiency = 877.028

  member_fair_w005_thr070_physical_r100:
    final_error       = 3.066e-07
    cluster_aoi       = 78.927
    member_aoi        = 60.541
    member_stale75    = 0.453
    member_zero       = 0.393
    energy_efficiency = 817.898

  member_fair_w010_thr070_physical_r100:
    final_error       = 7.179e-07
    cluster_aoi       = 77.462
    member_aoi        = 59.300
    member_stale75    = 0.430
    member_zero       = 0.359
    energy_efficiency = 776.244

  member_fair_w015_thr070_physical_r100:
    final_error       = 8.319e-06
    cluster_aoi       = 71.728
    member_aoi        = 54.328
    member_stale75    = 0.344
    member_zero       = 0.266
    energy_efficiency = 556.080

  member_fair_w025_thr070_physical_r100:
    final_error       = 1.710e-05
    cluster_aoi       = 70.934
    member_aoi        = 53.786
    member_stale75    = 0.327
    member_zero       = 0.244
    energy_efficiency = 528.478
  ```

  This is a real freshness/convergence tradeoff, not a plotting bug. For the
  physical/Rayleigh path, `w=0.05` is the only member-fair candidate that looks
  paper-defensible without more tuning: it gives moderate member freshness gains
  with about a 2x final-error increase and about 7% lower energy efficiency. The
  `w=0.10` point is still useful as an ablation; `w=0.15` and `w=0.25` are too
  aggressive for a main physical result. Cluster p75/p90/p95 and member
  p90/p95 still hit the horizon value `101`, so severe stale-tail AoI is not
  solved by access reweighting alone.

- The next structural model upgrade is now available as
  `--d2d-member-link-success-mode rayleigh_outage`. It makes aggregate
  completeness depend on the distance from each active member to the current
  CH, rather than only on one global D2D-success probability. This should be
  validated against the current physical `utility` baseline before adding
  another AoI access formula.

## Recommended Future Work Order

### Step 1: Diagnose And Attack Member Stale Causes

The code now includes member-stale failure attribution columns and member-aware
CH-rotation triggers. The refreshed physical `utility` baseline with member
metrics is available, and the current access-only member-fair Pareto candidate
is `w=0.05 / threshold=0.70`. The immediate research step is to run that access
candidate beside a structural CH-rotation candidate, then compare:

```text
mean AoI
p75/p90/p95 AoI
stale_fraction_50/75/100
member_mean/member_p95 AoI
member_zero_participation_fraction
member_stale_*_failure_fraction
error norm
energy efficiency
CH uploads
```

Expected benefit:

- confirms whether the moderate freshness gain at `w=0.05` survives higher
  Monte Carlo precision;
- checks whether the roughly 2x final-error cost is stable or just seed noise;
- identifies whether stale members are dominated by member compute, member link,
  member energy, CH no-attempt, ALOHA collision, or CH-to-BS physical decoding;
- documents that the severe p90/p95 AoI tail already exists in the tuned
  physical utility baseline and is not solved by light member-fair access
  reweighting;
- tests whether `--d2d-ch-rotation-trigger-mode interval_or_member_aoi` can
  reduce the stale tail by changing the CH rather than only changing access
  probability;
- provides a clean reference before implementing deeper structural changes such
  as re-clustering or data-aware D2D discovery.

Latest member-rotation result (`K=1000`, `rounds=100`, physical
energy/Rayleigh, `performance` CH rotation profile):

```text
utility_physical_member_diag_r100:
  final_error       = 1.603e-07
  member_aoi        = 62.043
  member_stale75    = 0.483
  member_zero       = 0.431
  energy_efficiency = 877.028
  stale cause       = 99.2% CH no-attempt

member_rotation_utility_physical_r100:
  final_error       = 1.148e-07
  member_aoi        = 62.229
  member_stale75    = 0.484
  member_zero       = 0.432
  energy_efficiency = 871.632
  stale cause       = 99.3% CH no-attempt

member_fair_w005_rotation_physical_r100:
  final_error       = 2.849e-07
  member_aoi        = 60.711
  member_stale75    = 0.455
  member_zero       = 0.394
  energy_efficiency = 809.473
  stale cause       = 98.8% CH no-attempt
```

Interpretation: member-aware CH rotation improves the physical utility error
curve, probably by moving to stronger CH-to-BS candidates, but it does not
materially improve member freshness. The failure attribution shows that stale
members are almost never blocked by member compute, member-to-CH Rayleigh links,
member energy, or CH-to-BS decoding. The dominant cause is that the CH carrying
those stale members does not attempt in that round. Therefore the next
implementation should target stale-member access opportunity directly: a
virtual-queue/member-refresh access mode, a bounded scheduled refresh overlay,
or local re-clustering that reduces the number of stale members competing for
the same sparse CH access opportunities.

Implemented next access test: `--optimized-d2d-access-mode
member_refresh_utility`. It uses the same base utility and active
stale/zero-member pressure as `member_fair_utility`, but adds a local refresh
floor through `--optimized-d2d-member-refresh-floor-fraction`. This directly
tests the `CH no attempt` diagnosis by raising the minimum attempt probability
only for refresh-eligible clusters. It is still an ALOHA probability policy, not
a centralized scheduler; the floor should be swept conservatively because it can
increase collisions when many clusters are stale at once.

Implemented explicit quota follow-up: `--optimized-d2d-access-mode
member_quota_utility`. This mode uses `--optimized-d2d-aoi-weight` as a reserved
stale-member refresh quota instead of only blending two full-load allocations:
the base utility allocator receives `1 - weight` of the CH contender target, and
refresh-eligible clusters receive a separate overlay with `weight` of the
target. It is designed to test the most recent diagnosis directly: if
`CH no attempt` remains near the previous `~0.986-0.992` level, then the next
research step should move beyond local ALOHA probability shaping toward virtual
queues, explicit refresh scheduling, or re-clustering.

Implemented stateful deficit follow-up: `--optimized-d2d-access-mode
member_deficit_utility`. It keeps the quota split but ranks the refresh overlay
with a persistent missed-refresh deficit that decays by
`--optimized-d2d-member-deficit-decay` and resets on successful optimized-D2D CH
delivery. This targets the observed saturation of member AoI percentiles at the
simulation horizon: when many clusters look equally stale, the deficit records
which clusters have repeatedly missed refresh opportunities.

The first undamped deficit test moved the diagnosed failure mode but was too
aggressive: `CH no attempt` fell from roughly `0.984` to `0.931-0.937`, while
ALOHA collision share rose to roughly `0.059-0.064`, final error degraded to
about `3e-3`, and energy efficiency fell near `240`. The implementation now
exposes `--optimized-d2d-member-deficit-weight` so the deficit can act as a
small tie-breaker instead of dominating the refresh quota.

The damped `member_deficit_utility` sweep did not recover the Pareto front.
With `w=0.15`, `decay=0.95`, and deficit weights `0.05`, `0.10`, and `0.25`,
the final optimized-D2D error stayed near `3.1e-3`, member stale75 stayed near
`0.463`, and energy efficiency stayed near `241-246`. This is much worse than
`member_quota_utility` with `w=0.15`, which reached final error `6.11e-7`,
member stale75 `0.413`, and energy efficiency `762.7`. The useful conclusion is
that persistent deficit probability shaping can move failures from `CH no
attempt` to collisions, but it is not a good access policy in the current dense
physical/Rayleigh regime. Keep `member_quota_utility` as the current
member-freshness candidate.

The full member-level D2D freshness experiment record is in
`docs/member_level_d2d_freshness_experiments.md`, including the final table,
negative-deficit interpretation, and recommended next research directions.
The paper-facing K=3000 comparison is now centralized in
`experiments.run_research_matrix`: run
`python -m experiments.run_research_matrix --matrix k3000_core --compare-only`
to regenerate `Runs/comparison_k3000_core/run_comparison_summary.*` and
`paper_claim_summary.md` without rerunning JAX.

Implemented collision-control follow-up: `--optimized-d2d-access-mode
member_collision_aware_quota`. It keeps the useful `member_quota_utility`
structure but tracks an EWMA of optimized-D2D CH collision fraction. When that
EWMA exceeds `--optimized-d2d-member-collision-target-fraction`, the reserved
member-refresh quota is reduced according to
`--optimized-d2d-member-collision-gain` and bounded below by
`--optimized-d2d-member-collision-min-quota-scale`. This is the next candidate
tested against `member_quota_w015_floor000_physical_r100`. The result is mixed:
it avoids the `~0.06` collision regime observed in `member_deficit_utility` and
improves convergence/energy relative to `member_quota_w015`, but it worsens the
member-freshness objective. Member stale75 rose from `0.413` to about
`0.461-0.462`, and zero-participation fraction rose from `0.337` to about
`0.401-0.403`. Keep it as a conservative ablation, not as the main
member-freshness policy. The next ALOHA-only access-control step should be
stronger quota settings and less aggressive collision damping, rather than
another global damping scalar.

Implemented semi-scheduled member refresh follow-up:
`--optimized-d2d-access-mode semi_scheduled_member_refresh`. It reserves
`ceil(M * --optimized-d2d-member-schedule-fraction)` D2D channels for the
highest active member-pressure clusters and runs utility ALOHA on the remaining
channels. Reserved attempts are collision-free but still require CH energy and
CH-to-BS decoding success. This was the next candidate tested because it attacks
the dominant `CH no attempt` stale-member attribution without increasing the
number of random ALOHA contenders. Compare it directly against
`member_quota_w015_floor000_physical_r100` and reject it if it improves member
freshness only by damaging convergence/energy.

The first semi-scheduled physical/Rayleigh sweep is strongly positive but now
classified as a coordinated upper-bound ablation. With
`--optimized-d2d-member-schedule-fraction 0.20` and no deficit tie-breaker,
final optimized-D2D error improved from `6.11e-7` for `member_quota_w015` to
`6.21e-9`; member AoI improved from `58.319` to `47.842`; member stale75
improved from `0.413` to `0.253`; zero-participation fraction improved from
`0.337` to `0.138`; and energy efficiency improved from `762.7` to `1040.6`.
The `0.10` schedule also improved all headline metrics, while adding
`--optimized-d2d-member-schedule-deficit-weight 0.25` at `0.10` gave only a
minor extra benefit. Treat `semi_scheduled_member_refresh` as evidence that
coordination can beat pure ALOHA, but not as the main research direction for
the current ALOHA-focused paper. The control-cost parameter charges a
normalized per-scheduled-CH coordination overhead to optimized+D2D
CH energy/battery accounting; it defaults to `0.0` so the no-overhead results
stay reproducible.

The first control-cost sweep used the best `s=0.20`, deficit-free point and
tested control costs `0.0001`, `0.0005`, and `0.0010`. The effect was almost
entirely on energy accounting, not on learning/freshness: final error stayed
near `6.1e-9`, member stale75 stayed near `0.253`, and zero-participation
fraction stayed near `0.138`. Energy efficiency declined from `1040.6`
without overhead to `1028.8`, `983.5`, and `932.2`, but all three still
outperformed `member_quota_w015` (`762.7`). This makes the semi-scheduled result
robust to small normalized coordination costs. A larger break-even sweep can be
run later if the paper needs a maximum tolerable control-overhead estimate.

The follow-up schedule-fraction sweep at `control_cost=0.0010` shows that the
policy behaves in channel-count steps, not smoothly in the fraction value. With
the default `M=10`, `0.05`/`0.10` reserve one channel, `0.15`/`0.20` reserve two,
and `0.25`/`0.30` reserve three. One channel is the conservative point
(`member_stale75=0.358`, `zero=0.283`, `energy_efficiency=990.7`); two channels
are the balanced point (`member_stale75=0.253`, `zero=0.138`,
`energy_efficiency=932.2`); three channels are the strongest freshness/error
point (`final_error=3.36e-9`, `member_stale75=0.183`, `zero=0.026`,
`energy_efficiency=870.3`). All three remain above `member_quota_w015` on energy
efficiency while strongly improving member freshness. The next robustness check
should therefore rerun the two-channel and three-channel points at larger
`K`/round counts.

That first larger-`K` check has now been run for `K=3000`, `rounds=100`,
`iterations=100`, and `control_cost=0.0010`. Both semi-scheduled points
converged strongly: the two-channel point reached final error `2.37e-16` and
`t<=1e-12` at round `66`; the three-channel point reached `2.09e-16` and
`t<=1e-12` at round `59`. Within `K=3000`, the three-channel point is the
better candidate so far (`member_aoi=63.863`, `member_stale75=0.437`,
`member_zero=0.312`, `energy_efficiency=825.7`) than the two-channel point
(`member_aoi=66.287`, `member_stale75=0.482`, `member_zero=0.387`,
`energy_efficiency=809.4`). The fair same-`K=3000` member-quota baseline is now
available: `member_quota_k3000_w015_floor000_physical_r100` reached final error
`4.22e-13`, `t<=1e-12` only at round `100`, `member_aoi=74.230`,
`member_stale75=0.618`, `member_zero=0.554`, and `energy_efficiency=653.8`.
Thus the three-channel semi-scheduled point improves the member-quota baseline
by `41` rounds to `1e-12`, `14.0%` member AoI, `29.2%` stale75, `43.6%` zero
participation, and `26.3%` final energy efficiency. This is an upper-bound
reference, not the next implementation direction. The pure-ALOHA K=3000 matrix
has also been run. Stronger quota weights (`0.20` and `0.25`) did not improve
member freshness and hurt convergence/energy. A small quota floor (`0.02`) was
nearly neutral. The best less aggressive collision-aware point was
`member_collision_quota_k3000_w015_t002_g2_min050_physical_r100`: it reached
`1e-12` 12 rounds earlier than `member_quota_k3000_w015_floor000`, improved
final energy efficiency by `5.34%`, and slightly lowered member stale75, but it
also slightly worsened zero-participation. Treat it as a convergence/energy
ablation, not as a replacement for the pure member-freshness baseline. The next
ALOHA-only implementation path is now structural. The first step,
`member_capped_quota_utility`, has been implemented as a pure ALOHA load-cap
extension of `member_quota_utility`: it caps only the extra member-refresh
overlay using `--optimized-d2d-member-quota-cap-fraction`, then adds the capped
overlay to the base utility probability. It does not reserve slots or add SIC,
MPR, NOMA, TDMA/OFDMA, or scheduling. The capped-quota matrix has now been run:
wide caps (`0.50`, `0.75`, `1.00`) did not bind; the best convergence/energy
point is `member_capped_quota_k3000_w015_cap010_physical_r100`, which reaches
`1e-12` at round `87` and improves final energy efficiency by `9.48%`, but
worsens zero-participation by `1.22%`; the most freshness-balanced point,
`cap025`, improves member AoI/stale75/zero by less than `0.15%` and does not
improve convergence. Treat capped quota as an ablation, not as a new
member-freshness winner. The collision-aware queue follow-up has also been run:
`member_queue_quota_k3000_w015_qw005_decay095_physical_r100` reduced
stale-member `CH no attempt` attribution, but raised collision attribution from
about `0.014` to `0.057`, reduced useful uploads, failed to reach `1e-12`, and
worsened member stale75/zero participation. Treat
`member_collision_aware_queue_quota` as another negative pure-ALOHA ablation.
The next ALOHA-compatible path is structural and the first implementation is
now available as `--cluster-split-mode max_size`: after local repair/merge,
clusters larger than `--cluster-split-max-size` are re-clustered internally
into valid one-hop subclusters before optional quality CH election. This
changes local D2D membership while preserving pure multichannel ALOHA access.
The first test, `member_split_k3000_s5_w015_physical_r100`, is negative:
clusterized-device rate fell from `99.319%` to `94.719%`, singleton count rose
from `20.44` to `158.43`, final useful optimized+D2D uploads fell from
`2269.79` to `554.25`, collision attribution rose from `0.014` to `0.055`,
and member stale75/zero worsened from `0.618`/`0.554` to `0.891`/`0.835`.
The gentler `member_split_k3000_s8_w015_physical_r100` run is less damaging
but still negative: uploads fall to `1797.06`, final error reaches only
`1.25e-7`, energy efficiency falls to `506.26`, and member stale75/zero worsen
to `0.670`/`0.604`. Naive global max-size splitting is therefore the wrong
structural direction. The selective follow-up is now implemented as
`--cluster-split-mode safe_max_size`. It keeps the original `max_size`
implementation unchanged for reproducibility, but adds two guards:
`--cluster-split-budget-fraction` limits how many oversized source rows can be
split, and `--cluster-split-min-subcluster-size` rejects local split proposals
that would create tiny tails. This remains a pure ALOHA-compatible structural
ablation, not slot reservation or scheduling. The first K=3000 physical point,
`member_safe_split_k3000_s8_min2_b005_w015_physical_r100`, is structurally
safe but not beneficial: total CH rows rise only from `627.45` to `631.47` and
singletons stay at `20.44`, but final error worsens from `4.22e-13` to
`1.06e-11`, useful optimized+D2D uploads fall by `2.17%`, energy efficiency
falls by `1.58%`, and member stale75/zero worsen from `0.618`/`0.554` to
`0.620`/`0.556`. Size-only selective splitting is therefore not enough; any
future re-clustering should be freshness-aware or member-participation-aware,
not just capped by cluster size. The next opt-in implementation is now
available as `--cluster-split-mode pressure_safe_max_size`. Because true
member stale/zero pressure exists only after FL rounds begin, this mode uses a
deployable pre-FL proxy: member-to-CH distance pressure and CH-to-BS channel
pressure. It keeps the same tiny-tail rejection and split budget as
`safe_max_size`, so it remains a structural ALOHA ablation rather than online
scheduling. The first K=3000 point,
`member_pressure_split_k3000_s8_min2_b005_mw100_ch050_w015_physical_r100`, is
better than size-only safe splitting for convergence and reaches `1e-12` at
round `95`, but it is still not a member-freshness improvement. Compared with
the quota baseline, member AoI/stale75/zero worsen from
`74.230`/`0.618`/`0.554` to `74.369`/`0.619`/`0.556`, useful uploads fall by
`0.93%`, and energy efficiency falls by `0.30%`. Treat pressure-guided safe
split as a structural convergence ablation, not as the next main contribution.

### Step 2: Calibrate The First-Order Energy Model

The code now implements opt-in first-order radio accounting:

```text
E_tx_bs(size, d)
E_tx_d2d(size, d)
E_rx(size)
E_agg(size)
```

The next scientific step is calibration, not another formula rewrite. Choose
literature or datasheet values for an IoT radio and map them into the normalized
battery scale used by the simulation. Report the source and the normalization.

Relevant CLI/API parameters now available:

```text
--energy-model {constant, first_order_radio}
--energy-electronics-cost
--energy-bs-amplifier-cost
--energy-d2d-amplifier-cost
--energy-bs-pathloss-exponent
--energy-d2d-pathloss-exponent
--energy-aggregation-cost
--energy-update-size
--energy-aggregate-size
--energy-rotation-control-cost
```

Expected benefit:

- makes energy-efficiency metrics physically interpretable;
- allows energy/error/AoI tradeoffs to be reported with defensible units or
  normalized units derived from a source.

### Step 3: Keep Battery As Feasibility In New Experiments

The preferred enhanced path is now:

```text
--battery-feasibility-mode required_energy
--d2d-ch-bs-battery-exponent 0.0
--device-bs-battery-exponent 0.0
```

Use the legacy battery exponent only for backward-compatible ablations. Battery
should primarily decide whether a role can pay the required energy:

```text
attempt_allowed_i = residual_energy_i >= E_required_i
```

Expected benefit:

- clearer physical interpretation;
- avoids a loose battery-to-decoding relationship;
- allows low-battery devices to skip or fail transmission because energy is
  insufficient, not because decoding magically degrades.

### Step 4: Extend Rayleigh Outage Toward SINR/PER If Needed

The code now supports Rayleigh outage for collision-free direct and CH-to-BS
decoding:

```text
avg_snr_i = reference_snr / max(distance_i, 1)^pathloss_exponent
q_i       = exp(-snr_threshold / avg_snr_i)
```

The next step, if the paper needs more physical detail, is:

```text
pathloss + fading + noise + co-channel interference powers
success if SINR >= threshold
optional FER/PER mapping
```

Expected benefit:

- `channel_weight` in CH selection becomes connected to a recognized
  communication-performance metric;
- D2D and non-D2D comparisons become easier to defend.

### Step 5: Treat AoI As A Policy Objective, But Move Beyond Access Only

AoI is now tracked for all six scenarios:

```text
non-D2D: per-device AoI
D2D:     per-cluster AoI for delivered aggregates
D2D:     per-member AoI for devices in non-singleton D2D clusters
```

The first AoI policy experiments are informative ablations rather than a
finished solution. Compare every future AoI strategy with:

```text
mean AoI
p75 AoI
p90 AoI
p95 AoI
stale_fraction_75
peak AoI
member_p95 AoI
member_zero_participation_fraction
error norm
energy efficiency
CH upload ratio
```

Expected benefit:

- freshness becomes measurable, not only a scheduling weight;
- enables analysis of `freshness_exp`, AoI quota, exponent, and threshold
  against mean/percentile/stale-tail AoI;
- provides another scientific objective beyond final error norm;
- redirects future implementation toward the structural cause of stale tails:
  CH feasibility, per-link D2D reliability, and cluster/CH reassignment.

### Step 6: Sensitivity Analysis For Policy Hyperparameters

Run structured sweeps for:

```text
norm_exp
size_exp
freshness_exp
load_target_factor
redistribution_fraction
redistribution_trigger_ratio
CH rotation weights
CH rotation interval
```

Report:

```text
log-error AUC
t_to_1e-6 / t_to_1e-9 / t_to_1e-12
CH upload ratio
device upload gain
energy used
energy efficiency
CH battery
mean/p95/peak AoI
member p95 AoI
member zero-participation fraction
```

Expected benefit:

- separates real algorithmic gains from parameter luck;
- provides defensible ranges rather than one-off tuned constants.

### Step 7: Add Data-Aware FL Realism As A Separate Research Phase

The current simulator is still a wireless HFL/ALOHA simulator with synthetic
linear-regression updates. The next research phase should be implemented as a
separate capability, not mixed into the P0 correctness and metric patch:

```text
non-IID data generation
FedAvg weighted by local sample count
FedProx or SCAFFOLD-style drift control
data-aware or update-similarity-aware clustering
GNN/RL graph discovery baselines
distributed control-plane message simulation
OTA-FL or hybrid D2D + OTA aggregation baselines
```

Expected benefit:

- separates communication-layer gains from FL statistical-heterogeneity gains;
- gives a defensible path toward innovation beyond engineering acceleration;
- avoids overclaiming the current synthetic model as a full non-IID FL study.

## Suggested References

- LEACH / first-order radio model direction: W. Heinzelman, A. Chandrakasan,
  and H. Balakrishnan, "Energy-Efficient Communication Protocols for Wireless
  Microsensor Networks," HICSS 2000.
- Guided FL participant selection motivation: [Oort](https://arxiv.org/abs/2010.06081).
- Freshness/AoI motivation: [WiFresh](https://arxiv.org/abs/2012.14337) and
  [Age of Information: An Introduction and Survey](https://arxiv.org/abs/2007.08564).
- Data-aware clustered FL motivation:
  [FedAC](https://arxiv.org/abs/2403.16460) and
  [FedDAG](https://arxiv.org/abs/2602.23504).
- D2D graph discovery motivation:
  [Multi-Agent Reinforcement Learning for Graph Discovery in D2D-Enabled
  Federated Learning](https://arxiv.org/abs/2503.23218).
