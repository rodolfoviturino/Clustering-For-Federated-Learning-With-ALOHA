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

Remaining limitation:

- Member-to-CH link success is still global when enabled; it does not yet depend
  on D2D distance, D2D fading, or local D2D SINR.

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
- re-election per D2D scenario, not globally shared between scenarios;
- one-hop coverage validation for the new CH;
- dynamic CH identity used for ALOHA, CH-to-BS success, aggregation, and energy
  drain.

Current empirical conclusion:

- `K=1000`: `eco` and `balanced` are attractive energy tradeoffs; `eco`
  preserves the most CH battery.
- `K=3000`: `performance` is the most defensible profile.  It is channel-first,
  preserves much more CH battery than static CHs, and keeps error/energy close
  to static.  More battery-heavy profiles preserve more CH battery but degrade
  optimized-D2D error.

Current limitation:

- CH-rotation control overhead is not charged.
- The interval and profile weights are still policy parameters that need
  sensitivity analysis.

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
  Age-of-Information metric.

Recommended future AoI metric:

```text
AoI_h(t + 1) =
  1,              if cluster h's update is received by the BS at t
  AoI_h(t) + 1,   otherwise
```

Report:

```text
mean AoI
peak AoI
AoI distribution over clusters
AoI versus error norm
AoI versus energy efficiency
```

This makes the freshness weight scientifically easier to analyze.

## Current Main Limitations

The current enhanced simulator is useful for research iteration, but the
following points should be explicitly disclosed:

- energy drain is normalized and not yet a calibrated radio model;
- CH receive/listening cost is not yet separated from CH transmission cost;
- CH rotation control overhead is not charged;
- battery currently can enter decode probability through a heuristic exponent;
- channel quality is inverse-pathloss based, not an outage/SINR/FER/PER model;
- member-to-CH link success is not distance/SINR dependent;
- fading, shadowing, explicit interference, coding, and modulation are absent;
- utility exponents and allocator thresholds are hyperparameters;
- freshness is not yet formal AoI;
- data are synthetic linear-regression data, not non-IID task data.

## Recommended Future Work Order

### Step 1: Replace Normalized Energy Costs With A Physical Energy Model

Implement per-attempt energy using packet size and distance:

```text
E_tx(l, d)
E_rx(l)
E_agg(l)
```

Add CLI/API parameters:

```text
--energy-model {normalized, first_order_radio}
--energy-e-elec
--energy-e-amp
--energy-pathloss-exponent
--energy-e-agg
--energy-update-bits
--energy-aggregate-bits
--energy-control-bits
```

Expected benefit:

- answers how each device energy is updated;
- differentiates member, CH receive, CH aggregation, CH-BS transmit, and direct
  device-BS transmit costs;
- makes energy-efficiency metrics more physically interpretable.

### Step 2: Use Battery As Feasibility, Not Decode Multiplier

Replace or demote:

```text
q_i *= battery_i^battery_exponent
```

with:

```text
attempt_allowed_i = residual_energy_i >= E_required_i
```

Expected benefit:

- clearer physical interpretation;
- avoids a loose battery-to-decoding relationship;
- allows low-battery devices to skip or fail transmission because energy is
  insufficient, not because decoding magically degrades.

### Step 3: Replace Channel Proxy With Outage/SINR-Based Success

Implement a channel model such as:

```text
pathloss + optional fading + noise + optional ALOHA interference
success if SINR >= threshold
```

Report:

```text
outage probability
success probability
optional PER/FER proxy
```

Expected benefit:

- `channel_weight` in CH selection becomes connected to a recognized
  communication-performance metric;
- D2D and non-D2D comparisons become easier to defend.

### Step 4: Add AoI Metrics

Track per-cluster AoI alongside freshness.

Expected benefit:

- freshness becomes measurable, not only a scheduling weight;
- enables analysis of `freshness_exp` against mean/peak AoI;
- provides another scientific objective beyond final error norm.

### Step 5: Sensitivity Analysis For Policy Hyperparameters

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
mean/peak AoI
```

Expected benefit:

- separates real algorithmic gains from parameter luck;
- provides defensible ranges rather than one-off tuned constants.

## Suggested References

- LEACH / first-order radio model direction: W. Heinzelman, A. Chandrakasan,
  and H. Balakrishnan, "Energy-Efficient Communication Protocols for Wireless
  Microsensor Networks," HICSS 2000.
- Guided FL participant selection motivation: [Oort](https://arxiv.org/abs/2010.06081).
- Freshness/AoI motivation: [WiFresh](https://arxiv.org/abs/2012.14337) and
  [Age of Information: An Introduction and Survey](https://arxiv.org/abs/2007.08564).

