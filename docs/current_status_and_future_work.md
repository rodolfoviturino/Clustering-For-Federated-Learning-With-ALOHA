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

### Step 1: Replicate The Physical Member-Fair Pareto Point

The immediate validation step is not a new formula. The refreshed physical
`utility` baseline with member metrics is now available, and the current
member-fair Pareto candidate is `w=0.05 / threshold=0.70`. Before promoting it
in a paper, repeat or extend that point with more seeds/iterations and keep the
same physical-link options used by the baseline. Compare:

```text
mean AoI
p75/p90/p95 AoI
stale_fraction_50/75/100
member_mean/member_p95 AoI
member_zero_participation_fraction
error norm
energy efficiency
CH uploads
```

Expected benefit:

- confirms whether the moderate freshness gain at `w=0.05` survives higher
  Monte Carlo precision;
- checks whether the roughly 2x final-error cost is stable or just seed noise;
- documents that the severe p90/p95 AoI tail already exists in the tuned
  physical utility baseline and is not solved by light member-fair access
  reweighting;
- provides a clean reference before implementing deeper structural changes such
  as AoI/channel-aware CH rotation, re-clustering, or data-aware D2D discovery.

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
