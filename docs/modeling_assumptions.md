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
  normalized BS channel quality, and battery. In the current collision-oriented
  ALOHA model, this is a structural CH-selection ablation; strong deterministic
  gains require a follow-up CH-to-BS channel or energy model that makes the
  elected CH's quality affect successful uplink delivery or cost.
- Enhanced runs can enable channel-aware CH-to-BS decoding with
  `--d2d-ch-bs-success-mode channel_quality`. A CH still contends through
  ALOHA and can still collide; only after a collision-free attempt does the BS
  decode the packet with probability derived from the elected CH's normalized
  inverse pathloss and optional battery factor. The default `none` preserves
  thesis-compatible collision-only D2D uploads.
- First-tier HFL aggregation at the CH is a sum of member updates.
- The thesis figure code applies the BS update as an unscaled SGD step:
  `w <- w - u1 * gradient`.
- D2D member availability is ideal by default:
  - `d2d_member_compute_probability=1.0`;
  - `d2d_member_link_success_probability=1.0`.
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

These defaults are intended to preserve the thesis figure behavior while fixing
code bugs such as angle units, unsafe cluster merging, and fragile cluster-array
inputs.

## Realism Knobs

Set either of the following below `1.0` to model imperfect member-to-CH
participation:

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
- Energy-aware CH selection is not implemented in this cleanup.
- Channel-aware CH-to-BS success is implemented as a decoding-probability
  ablation, not a full energy-drain model. It changes whether a collision-free
  D2D aggregate reaches the BS; it does not yet reduce battery over time.
