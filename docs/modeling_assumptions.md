# Modeling Assumptions

This document records the simulation defaults after the code cleanup.

## Thesis-Compatible Defaults

- Device placement follows the thesis:
  - `r ~ U(1, R_BS)`;
  - `theta ~ U(0, 2*pi)`.
- D2D-SRC clusters are one-hop from every member to the cluster head.
- Cluster size is capped by `Cmax`.
- Dense JAX geometric clustering forms local pairs first by default. This is a
  local D2D request/accept stage, not a BS-side global assignment.
- Dense JAX clustering may run local singleton join-repair passes. A singleton
  can join only a directly reachable CH with spare capacity; this models local
  D2D control messages, not BS-side global orchestration.
- Dense JAX clustering may run local CH-rotation repair for two-device clusters.
  A member becomes CH only when it directly reaches both the old CH and the
  singleton being admitted.
- Dense JAX clustering may run local CH-to-CH merge repair after singleton and
  rotation repair. A source D2D cluster can merge into a target D2D cluster
  only when the target CH can directly cover every source member and the union
  still fits within `Cmax`.
- Enhanced runs can enable quality CH election with
  `--cluster-head-selection-mode quality`. This does not change cluster
  membership. It only moves the CH role to the highest-scoring member that can
  still directly cover the whole cluster. The score uses local D2D degree,
  normalized BS channel quality, and battery. In the collision-only default it
  is mostly a structural CH-selection ablation; its physical effect becomes
  measurable when channel-aware CH-to-BS decoding or dynamic energy drain is
  enabled.
- Enhanced runs can enable channel-aware CH-to-BS decoding with
  `--d2d-ch-bs-success-mode channel_quality`. A CH still contends through
  ALOHA and can still collide; only after a collision-free attempt does the BS
  decode the packet with probability derived from the elected CH's normalized
  inverse pathloss and optional battery factor. The default `none` preserves
  thesis-compatible collision-only D2D uploads.
- Enhanced runs can instead use `--d2d-ch-bs-success-mode rayleigh_outage`.
  This keeps the same ALOHA contention and collision model, but maps the
  collision-free physical decoding probability to:

  ```text
  avg_snr_i = reference_snr / max(distance_i, 1)^pathloss_exponent
  q_i       = exp(-snr_threshold / avg_snr_i)
  ```

  The Rayleigh option gives the CH-channel term a standard outage-probability
  interpretation. `--cluster-head-channel-score-mode rayleigh_outage` makes
  quality CH election use the same D2D CH-to-BS reference SNR and threshold
  instead of normalized inverse pathloss.
- Enhanced runs can enable the same channel-aware decoding for direct non-D2D
  device-to-BS uploads with `--device-bs-success-mode channel_quality`. This
  affects polling, fixed ALOHA, and optimized ALOHA without D2D. A direct
  device still needs to compute/access and avoid collision first; only then is
  the packet decoded according to the device's normalized inverse pathloss and
  optional battery factor. Use both device-BS and D2D CH-BS channel modes when
  the goal is a physically fair D2D vs non-D2D comparison.
- Direct non-D2D links can also use
  `--device-bs-success-mode rayleigh_outage`, with independent reference SNR
  and threshold parameters. For fair physical comparisons, pair Rayleigh D2D
  CH-to-BS decoding with Rayleigh direct device-to-BS decoding.
- Enhanced runs can enable dynamic battery drain with
  `--energy-drain-mode dynamic`. The model keeps one normalized battery vector
  per curve, charges energy on attempted direct BS, D2D member, and CH-to-BS
  transmissions, and lets later battery-aware decoding probabilities see the
  updated battery. The default `none` preserves static battery behavior.
- `--energy-model constant` is the legacy dynamic-energy model. It uses three
  fixed normalized costs:

  ```text
  energy_direct_bs_cost
  energy_d2d_member_cost
  energy_ch_bs_cost
  ```

  It is useful for reproducing previous enhanced runs, but it is not a
  calibrated radio-energy model.
- `--energy-model first_order_radio` is the preferred enhanced energy path.
  Energy remains normalized to the initial battery range `[0, 1]`, but the
  accounting separates the radio roles:

  ```text
  E_tx_bs(size, d)  = size * E_elec + size * E_amp_bs  * d^alpha_bs
  E_tx_d2d(size, d) = size * E_elec + size * E_amp_d2d * d^alpha_d2d
  E_rx(size)        = size * E_elec
  E_agg(size)       = size * E_agg
  ```

  Direct devices pay `E_tx_bs(update_size, d_device_bs)`. D2D members pay
  `E_tx_d2d(update_size, d_member_ch)`. A CH pays receive cost for active
  non-CH members, aggregation cost for active updates, and
  `E_tx_bs(aggregate_size, d_ch_bs)` for the BS aggregate. The CH-to-BS payload
  size stays fixed by `energy_aggregate_size`, while aggregation processing
  scales with the number of active updates.
- `--battery-feasibility-mode required_energy` treats battery as an attempt
  feasibility constraint. A direct device, D2D member, or CH attempts only when
  its current normalized battery can pay the required role energy. If not, it
  skips the attempt and does not drain. This is the cleaner physical
  interpretation; the older `*_battery_exponent` terms should be treated as
  legacy/heuristic unless a specific experiment is studying them.
- Enhanced runs can enable intra-run D2D CH rotation with
  `--d2d-ch-rotation-mode energy_aware`. This requires
  `--energy-drain-mode dynamic`, keeps cluster membership fixed, and re-elects a
  CH per D2D curve from valid members that still cover the full cluster in one
  hop. The trigger can be periodic (`interval`), AoI-tail based (`aoi`), or the
  union of both (`interval_or_aoi`). It can also use member-level AoI triggers:
  `member_aoi`, `interval_or_member_aoi`, `aoi_or_member_aoi`, and
  `interval_or_aoi_or_member_aoi`. The score combines normalized BS channel
  quality, current battery, and a stability bonus for keeping the current CH.
  When the trigger contains `member_aoi`, an optional member-link score
  (`--d2d-ch-rotation-member-link-weight`) favors CHs that are more likely to
  decode stale members under Rayleigh D2D links. The default `static` preserves
  thesis-compatible fixed CH identity.
- First-tier HFL aggregation at the CH is a sum of member updates.
- The thesis figure code applies the BS update as an unscaled SGD step:
  `w <- w - u1 * gradient`.
- Polling without D2D schedules direct device IDs. Polling with D2D schedules
  cluster rows and evaluates compute/energy feasibility on the scheduled CH,
  not on the direct polling user from the non-D2D curve.
- D2D member availability is ideal by default:
  - `d2d_member_compute_probability=1.0`;
  - `d2d_member_link_success_probability=1.0`.
- Enhanced runs can replace the scalar member-to-CH link probability with
  distance-aware Rayleigh outage:

  ```text
  --d2d-member-link-success-mode rayleigh_outage
  q_member,h = exp(-snr_threshold / avg_snr_member,h)
  avg_snr_member,h =
    reference_snr / max(distance_member_to_current_CH, 1)^pathloss_exponent
  ```

  This is still a collision-free packet decoding abstraction, but it removes a
  major earlier simplification: all D2D members no longer have identical link
  reliability. With dynamic CH rotation, changing the CH can change both
  member energy cost and member decoding probability.
- Optimized ALOHA uses the aggregate norm of the CH update.
- Optimized ALOHA can optionally use a fixed-access floor. This keeps the
  norm-based controller from starving the channel late in training, especially
  in D2D runs where aggregate norms can shrink quickly.
- Optimized D2D can optionally use a utility policy that combines aggregate
  norm, active aggregate size, and freshness while keeping the expected ALOHA
  load near the channel count.
- The utility load target can be tuned with
  `--optimized-d2d-load-target-factor`, but the default `1.0` preserves the
  current target of approximately `M` CH contenders.
- Enhanced utility-style modes use conditional selective water-filling by
  default. This can redistribute CH access-probability mass clipped by `pcomp`,
  but only when the EWMA of successful optimized-D2D CH uploads falls below the
  configured trigger relative to the expected fixed-D2D CH throughput. It does
  not centrally schedule CHs; each CH still performs an independent ALOHA
  decision. When the trigger is not active, the allocator returns the exact
  legacy proportional clipped probabilities.
- Optimized D2D can also use a max-weight threshold policy. The BS only needs
  to broadcast a scalar threshold/dual variable; each CH computes its own
  utility score locally from its aggregate norm, active member count, and
  freshness.
- Optimized D2D can also use a hybrid utility/novelty policy. The BS keeps a
  recent successful optimized-D2D update direction and broadcasts it with the
  model; each CH discounts aggregates that are directionally redundant with
  that reference.
- Optimized D2D can also use an adaptive-diversity policy. It keeps the same
  CH-level load controller as utility/hybrid, but changes the score over time:
  early rounds emphasize large useful aggregates, while later rounds emphasize
  novelty and freshness. The phase uses `t / max_t`, not measured model error,
  so the policy remains plausible when the true optimum is unknown.
- Optimized D2D can also use AoI-enhanced utility policies. The multiplicative
  `aoi_aware_utility` policy preserves the base utility terms, then gives an
  extra bounded bonus to clusters whose current AoI sits in the stale tail. The
  more conservative `aoi_floor_utility` policy preserves the base utility
  probability and only raises stale clusters to a bounded minimum probability.
  In both cases, the CH can maintain this age from ACK/no-ACK feedback, and the
  BS only needs to broadcast scalar normalizers.
- `member_fair_utility` is a member-level ablation. It keeps the base utility
  allocator and reserves a configurable quota for multi-member clusters whose
  currently active aggregate contains stale or zero-participation devices.
  This tests whether member starvation is caused by CH access allocation rather
  than by clustering alone. It is disabled by default and does not introduce
  non-IID/FedAvg semantics.
- `member_refresh_utility` is the access-side response to the observed
  `CH no attempt` stale-member diagnosis. It keeps the same base utility and
  refresh pressure as `member_fair_utility`, but applies
  `--optimized-d2d-member-refresh-floor-fraction` as a local minimum attempt
  probability for refresh-eligible clusters. It remains distributed ALOHA:
  clusters draw locally from a probability, and the floor can increase collision
  risk when many stale clusters become eligible at once.
- `member_quota_utility` is the explicit quota version of the same hypothesis.
  When active stale or zero-participation members exist, it splits the optimized
  D2D CH contender target into a base utility budget and a member-refresh
  overlay. `--optimized-d2d-aoi-weight` is the reserved refresh quota fraction,
  and `--optimized-d2d-member-refresh-floor-fraction` remains an optional local
  floor on the overlay. This tests whether the dominant member-stale cause is
  truly lack of CH access opportunity rather than the smoother probability
  blending used by `member_fair_utility`/`member_refresh_utility`.
- `member_capped_quota_utility` keeps the same explicit base/overlay split, but
  limits only the extra member-refresh overlay with
  `--optimized-d2d-member-quota-cap-fraction`. The cap is a fraction of the
  fixed D2D ALOHA access probability and is applied before adding the overlay to
  the base utility probability. This is a pure ALOHA load-shaping ablation:
  CHs still attempt locally with a probability, multichannel collisions remain
  the MAC abstraction, and the mode does not reserve slots or add SIC, MPR,
  NOMA, TDMA/OFDMA, or a centralized scheduler.
- `member_collision_aware_quota` keeps that explicit member-refresh quota but
  dampens it using an EWMA of optimized-D2D CH collisions. The quota is
  unchanged below `--optimized-d2d-member-collision-target-fraction` and is
  smoothly reduced by `--optimized-d2d-member-collision-gain` toward
  `--optimized-d2d-member-collision-min-quota-scale` above the target. This is
  the collision-control response to the `member_deficit_utility` result.
- `member_collision_aware_queue_quota` keeps the same local ALOHA quota
  structure but changes the state update behind the deficit term. A missed
  active stale-member opportunity caused by no CH attempt increases the queue;
  a collision-caused miss does not increase it; a CH-BS decoding miss receives
  only a small increment. The queue is then used as the same normalized
  tie-breaker controlled by `--optimized-d2d-member-deficit-decay` and
  `--optimized-d2d-member-deficit-weight`. This mode is still probability
  shaping, not scheduling.
- `semi_scheduled_member_refresh` is the first member-aware policy that is not
  pure ALOHA probability shaping. It reserves
  `ceil(M * --optimized-d2d-member-schedule-fraction)` D2D channels for the
  clusters with the highest active member AoI/zero-participation pressure, then
  runs utility ALOHA on the remaining channels. A scheduled CH attempt is
  collision-free, but it still requires CH battery feasibility and CH-to-BS
  decoding success. The optional
  `--optimized-d2d-member-schedule-deficit-weight` only breaks ties in the
  ranking; it does not create extra scheduled slots. This mode assumes a small
  BS/CH control decision for the reserved refresh slots, so it is a
  coordinated semi-scheduled ablation rather than fully distributed random
  ALOHA. `--optimized-d2d-member-schedule-control-cost` optionally charges a
  normalized per-scheduled-CH coordination overhead to the optimized+D2D CH
  battery and energy accounting. The default is `0.0` to preserve prior runs;
  positive values are intended for overhead-sensitivity ablations.
- `member_deficit_utility` adds one piece of memory to the explicit quota
  policy. The optimized-D2D scenario keeps a per-cluster missed-refresh deficit
  that increases when active stale or zero-participation members are available
  but the aggregate is not delivered, resets on successful delivery, and decays
  by `--optimized-d2d-member-deficit-decay`. The deficit is damped by
  `--optimized-d2d-member-deficit-weight` so it acts as a tie-breaker rather
  than a hard scheduler; excessive weight can simply trade no-attempt failures
  for ALOHA collisions. The deficit only ranks the local ALOHA probability
  overlay; it is not centralized scheduling.
  See `docs/member_level_d2d_freshness_experiments.md` for the current
  physical/Rayleigh conclusion: `member_quota_utility` is the useful pure
  ALOHA/probability-shaping member-freshness baseline.
  `member_capped_quota_utility` is implemented and useful as a convergence/energy
  ablation, but it is not a new member-freshness winner because its best
  convergence point worsens zero-participation and its balanced freshness point
  has only sub-`0.15%` gains. `member_deficit_utility` is a negative
  collision-dominated ablation. `member_collision_aware_queue_quota` is also a
  negative ALOHA ablation after the first K=3000 test: suppressing
  collision-caused queue increments was not enough to prevent stateful debt
  ranking from over-concentrating CH attempts and increasing collisions.
  `semi_scheduled_member_refresh` is the strongest current enhanced candidate
  when a small explicit refresh schedule is allowed. With `M=10`, the two-slot
  point (`schedule_fraction` `0.15`/`0.20`) is the balanced candidate and the
  three-slot point (`0.25`/`0.30`) is the strongest tested member-freshness
  candidate; deficit tie-breaker weight `0.0` is sufficient for both.
- AoI is tracked as an output metric for all six scenarios. Non-D2D scenarios
  track per-device AoI and reset a device to `1` after its successful upload.
  D2D scenarios track per-cluster AoI and reset a cluster to `1` after its CH
  aggregate reaches the BS. Otherwise AoI increments by one. The CSV includes
  `<scenario>_aoi_*`, `<scenario>_peak_aoi_*`, `<scenario>_p75_aoi_*`,
  `<scenario>_p90_aoi_*`, `<scenario>_p95_aoi_*`, and stale-fraction columns.
  The plotter emits the corresponding figures when those columns exist.
  The proportional stale-tail fractions use `AoI > 1 + fraction * elapsed_t`
  so a freshly updated device or cluster with AoI `1` is not stale at early
  checkpoints. The fixed long-stale diagnostic remains `AoI > 100`.
- D2D scenarios also report member-level AoI diagnostics over devices in
  non-singleton D2D clusters. A member-level AoI sample resets only when that
  device's update is active inside a CH aggregate that reaches the BS;
  otherwise it increments. These columns expose cases where cluster-level AoI
  looks fresh because the CH uploads often, while non-CH members are stale
  because compute, D2D link, or energy feasibility kept them out of delivered
  aggregates. The CSV uses the
  `<scenario>_member_*` prefix for these D2D-only diagnostics.
- D2D scenarios also report member-stale failure attribution columns. They are
  fractions over devices in non-singleton D2D clusters that remain stale at the
  checkpoint, using the `AoI > 1 + 0.75 * elapsed_t` severe-tail threshold. The
  hierarchy is diagnostic rather than causal proof: member compute failure,
  member-to-CH link failure, member energy infeasibility, CH no-attempt, ALOHA
  collision, CH-to-BS decode failure, and an `other` residual bucket. These
  columns identify whether stale-tail AoI is mostly local member availability,
  medium access, physical uplink, or an instrumentation gap.
- AoI affects scheduling only when an explicit AoI-enhanced policy is selected.

These defaults are intended to preserve the thesis figure behavior while fixing
code bugs such as angle units, unsafe cluster merging, fragile cluster-array
inputs, and polling+D2D attempts that were incorrectly coupled to direct
polling users.

## Realism Knobs

Set either of the following below `1.0` to model imperfect member-to-CH
participation with the legacy constant-link abstraction:

```python
error_calculator_trace_jax(
    ...,
    d2d_member_compute_probability=0.8,
    d2d_member_link_success_probability=0.9,
)
```

When enabled, a successful CH-to-BS transmission carries:

- the CH's own update;
- only member updates whose local computation and D2D link both succeed.

This makes D2D gains less optimistic and separates CH-to-BS success from
member-to-CH availability.

For per-link D2D outage, use:

```python
error_calculator_trace_jax(
    ...,
    d2d_member_link_success_mode="rayleigh_outage",
    d2d_member_pathloss_exponent=2.0,
    d2d_member_reference_snr=100000.0,
    d2d_member_snr_threshold=1.0,
    device_coords=device_coords,
)
```

The CH is always considered locally active for its own cluster. Other members
must compute their local update, have enough energy if battery feasibility is
enabled, and pass the member-to-current-CH decoding draw.

## Ablations Not Enabled By Default

- `uniform_area=True` in `devices_generator(...)` samples devices uniformly over
  disk area instead of following the thesis radial distribution.
- `normalize_by_k=True` or `--normalize-by-k` divides the received gradient sum
  by `K`. This is a conservative learning-scale ablation, not the default used
  for reproducing the thesis-style Figure 15 curve.
- `--precision float64` enables JAX double precision. It is slower, especially
  on consumer GPUs, but is needed when reproducing optimized ALOHA error norms
  below the `float32` numerical floor.
- CH access utilities can be normalized by cluster size or combined with
  freshness, energy, or channel quality. The current default intentionally keeps
  the thesis aggregate-norm policy.
- `--optimized-d2d-access-floor-fraction 1.0` is an enhanced optimized-ALOHA
  ablation. It preserves norm-based priority while ensuring optimized D2D uses
  at least the fixed-D2D access probability.
- `--optimized-d2d-access-mode utility` is a stronger optimized-D2D ablation.
  It is not thesis-exact; it tests whether utility-aware probability allocation
  can outperform fixed D2D without increasing total expected channel load.
- `--optimized-d2d-load-allocation-mode proportional_clip` reproduces the older
  enhanced-mode allocator. The default `conditional_selective_water_filling`
  uses a scalar water level only when observed optimized-D2D CH throughput is
  below the fixed-D2D reference throughput, so clipped high-utility probability
  can be assigned to other CHs that are still below `pcomp` without forcing
  extra load in already high-throughput regimes. If the trigger is off, no
  water-filling reshuffle is applied.
- `--optimized-d2d-load-allocation-mode selective_water_filling` redistributes
  only a fraction of the clipped probability mass. This tests the middle ground
  between selective high-utility CH access and full water-filling load usage.
- `--optimized-d2d-access-mode max_weight` is a more selective enhanced
  optimized-D2D ablation. It maps the same utility score through an adaptive
  sigmoid threshold, concentrating access on high-utility CHs while using the
  observed contention count to move the threshold up or down.
- `--optimized-d2d-access-mode hybrid` keeps smooth utility load control but
  adds directional novelty. It tests whether the optimized D2D controller can
  improve by selecting less redundant CH aggregate directions while preserving
  roughly the same expected CH contention load.
- `--optimized-d2d-access-mode adaptive_diversity` is a stronger algorithmic
  ablation. It uses:

  ```text
  phase(t) = sigmoid(gain * ((t / max_t) - switch_fraction))

  adaptive_utility_h =
    (1 - phase(t)) *
      norm_h^early_norm *
      active_cluster_size_h^size_exp *
      freshness_h^early_freshness
    + phase(t) *
      norm_h^late_norm *
      active_cluster_size_h^size_exp *
      freshness_h^late_freshness *
      novelty_h^novelty_exp
  ```

  The CH can compute its local score from aggregate norm, active aggregate size,
  freshness, and aggregate direction. The BS can broadcast scalar normalizers,
  the recent optimized-D2D reference direction, and the temporal phase. This is
  not a BS-side global assignment and does not use the true error curve to
  switch behavior.
- `--optimized-d2d-access-mode aoi_aware_utility` is a targeted freshness
  ablation. It starts from the same base utility as `utility`:

  ```text
  base_utility_h =
    norm_h^norm_exp *
    active_cluster_size_h^size_exp *
    freshness_h^freshness_exp
  ```

  It then computes a stale-tail AoI pressure:

  ```text
  normalized_aoi_h = AoI_h / max_j(AoI_j)
  tail_h = clip(
    (normalized_aoi_h - threshold_fraction) / (1 - threshold_fraction),
    0,
    1
  )
  aoi_bonus_h = 1 + aoi_weight * tail_h^aoi_exp
  aoi_aware_utility_h = base_utility_h * aoi_bonus_h
  ```

  This is not meant to maximize AoI priority blindly.  The threshold makes the
  bonus act mostly on the stale tail, so the policy can test whether optimized
  D2D can keep its error/energy gains while reducing mean and p95 AoI.
- `--optimized-d2d-access-mode aoi_floor_utility` is the conservative AoI
  ablation. It computes the base utility probability first and then applies:

  ```text
  p_h = max(
    base_probability_h,
    fixed_d2d_access_probability *
      aoi_weight *
      tail_h^aoi_exp
  )
  ```

  Here `aoi_weight` is a fraction of the fixed-D2D access probability, not a
  multiplicative utility boost. This keeps stale clusters from being completely
  ignored while preserving most of the original utility ranking.
- `--optimized-d2d-access-mode aoi_tail_utility` is the stronger stale-tail
  quota ablation. It computes the base utility probability and a second
  probability using only AoI-tail pressure:

  ```text
  base_probability_h = load_control(base_utility_h)
  tail_probability_h = load_control(tail_h^aoi_exp, floor = 0)

  p_h =
    (1 - quota) * base_probability_h +
    quota * tail_probability_h
  ```

  In this mode `quota = clip(aoi_weight, 0, 1)`. The policy returns
  `base_probability_h` exactly when active clusters have no differentiated AoI
  tail, so early rounds with uniform AoI are not disturbed. The motivation is
  the latest physical-energy observation that mean AoI can improve while
  p75/p90/p95 AoI remain saturated; this policy reserves a real share of the
  load budget for the stale tail instead of only multiplying utility.
- `--optimized-d2d-access-mode aoi_quality_tail_utility` uses the same quota
  structure as `aoi_tail_utility`, but the tail allocator is physically
  qualified:

  ```text
  quality_tail_h =
    tail_h^aoi_exp *
    q_ch_bs_h^aoi_channel_exp *
    battery_ch_h^aoi_battery_exp

  tail_probability_h = load_control(quality_tail_h, floor = 0)
  p_h =
    (1 - quota) * base_probability_h +
    quota * tail_probability_h
  ```

  Here `q_ch_bs_h` is the current collision-free CH-to-BS success probability
  under the selected channel model, and `battery_ch_h` is current normalized CH
  battery. This is the next AoI/access hypothesis after plain stale-tail quota:
  it tests whether AoI tail reduction can be obtained without spending scarce
  CH contender probability on stale clusters whose CH is unlikely to deliver
  the aggregate. The deployment interpretation is still distributed: the CH can
  know or estimate its own BS-channel success and battery, while the BS only
  needs to broadcast scalar exponents and normalizers.
- `--optimized-d2d-access-mode member_fair_utility` reuses the same
  load-controlled utility allocator, but reserves
  `optimized_d2d_aoi_weight` of the CH access budget for active aggregates that
  can refresh stale or zero-participation members:

  ```text
  base_probability_h = load_control(base_utility_h)
  member_pressure_h =
    max(
      stale_tail(mean_active_member_aoi_h),
      zero_participation_fraction_h
    )
  member_probability_h = load_control(member_pressure_h)
  p_h = (1 - quota) * base_probability_h + quota * member_probability_h
  ```

  The pressure is evaluated only for active members inside non-singleton
  aggregates, because extra CH access cannot refresh a member that failed local
  compute, D2D link, or energy feasibility in the current round. Current
  ideal-link runs show this policy can strongly reduce member starvation. In the
  refreshed physical energy/Rayleigh comparison, the lightest tested weight
  (`w=0.05 / threshold=0.70`) is the current Pareto candidate, while larger
  weights show substantial error and energy-efficiency cost. Treat it as a
  fairness ablation, not as the physical default.
- `conditional_selective_water_filling` is the default allocator for enhanced
  load-controlled policies. It is intentionally not a centralized scheduler:
  the BS can broadcast only the target load, trigger ratio, redistribution
  fraction, and scalar water-level normalizer. Each CH still computes its own
  access probability and performs its own ALOHA trial. The condition uses an
  EWMA of successful CH uploads, which is observable through ACKs, rather than
  attempted contenders, the inflated exploratory target, or the training error.
  True optimization error is not available in a real deployment.
- The conditional allocator is density-aware. When the clusterized-device
  fraction exceeds `--optimized-d2d-density-trigger-threshold`, the effective
  trigger becomes `--optimized-d2d-dense-trigger-ratio`; otherwise it remains
  `--optimized-d2d-redistribution-trigger-ratio`. This is still a scalar
  control rule, not BS-side CH scheduling. The clusterization fraction is
  available after cluster formation, and the BS can broadcast the effective
  trigger with the normal FL control parameters. The motivation is empirical
  and physical: dense D2D coverage already creates many eligible CHs, so extra
  probability redistribution can increase collision pressure without adding
  proportional information gain.
- Dynamic energy drain is an enhanced ablation, not a calibrated power model.
  It uses normalized per-attempt costs and does not model recharge, voltage,
  thermal effects, detailed transmit power control, CH receive/listening energy,
  aggregation energy, or control-plane overhead. Intra-run CH re-election is
  available separately through `--d2d-ch-rotation-mode energy_aware`, but that
  rotation currently uses the same normalized energy abstraction.
- Channel-aware CH-to-BS success is implemented as a decoding-probability
  ablation. When dynamic energy is also enabled, repeated CH duty can reduce
  later battery-aware decoding probability; when dynamic energy is disabled,
  battery remains the generated static suitability/decoding signal.
- Channel-aware direct device-to-BS success is implemented with the same
  probability model as CH-to-BS decoding, but it is still not a full physical
  layer. The current abstraction uses distance/pathloss and optional battery,
  not SINR, coding rate, fading, or shadowing.
