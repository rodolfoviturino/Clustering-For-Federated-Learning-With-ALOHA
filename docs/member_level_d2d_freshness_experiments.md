# Member-Level D2D Freshness Experiments

This note records the current research conclusion for member-level freshness in
the enhanced JAX simulator. It separates the useful result from the negative
ablations so the same line of experimentation is not repeated accidentally.

## Motivation

Cluster-level D2D AoI resets when the elected cluster head (CH) delivers an
aggregate to the BS. That can hide starvation inside a cluster: a CH can upload
often while some non-CH members almost never enter a delivered aggregate.

The simulator now therefore tracks D2D member-level AoI and participation for
devices in non-singleton clusters. A member sample resets only when that
device's update is active inside a delivered CH aggregate.

The physical/Rayleigh runs showed that stale member samples are usually not
blocked by member compute, member-to-CH link, member energy, or CH-to-BS decode.
The dominant attribution is that the CH carrying those stale members does not
attempt in that round.

## Policies Tested

`member_fair_utility`

Smoothly reserves part of the optimized-D2D load budget for clusters whose
currently active aggregate contains stale or zero-participation members. This
was the first member-level access ablation.

`member_refresh_utility`

Adds a local minimum access probability for refresh-eligible clusters. This
directly tests whether stale members are access-opportunity limited.

`member_quota_utility`

Splits the optimized-D2D CH contender target into a base utility budget and an
explicit member-refresh overlay. `--optimized-d2d-aoi-weight` is the reserved
refresh quota. This is the current best member-freshness candidate.

`member_collision_aware_quota`

Keeps the quota structure but reduces the reserved member-refresh overlay when
an EWMA of optimized-D2D CH collisions exceeds a configured target. This is the
first follow-up after the negative deficit result and should be compared against
`member_quota_w015_floor000_physical_r100`.

`member_deficit_utility`

Adds persistent per-cluster missed-refresh debt to the quota overlay. The goal
was to break ties when many clusters have saturated member AoI. This policy is
implemented for ablation, but the current physical/Rayleigh results reject it
as a main candidate because it turns no-attempt failures into collisions.

## Physical/Rayleigh Setup

The comparison below uses `K=1000`, `rounds=100`, `iterations=100`,
`precision=float64`, dynamic first-order-radio energy, required-energy battery
feasibility, Rayleigh D2D member links, Rayleigh CH-BS links, Rayleigh direct
device-BS links, and channel-quality CH selection:

```text
--energy-drain-mode dynamic
--energy-model first_order_radio
--battery-feasibility-mode required_energy
--d2d-member-link-success-mode rayleigh_outage
--d2d-ch-bs-success-mode rayleigh_outage
--device-bs-success-mode rayleigh_outage
--cluster-head-selection-mode quality
--cluster-head-channel-score-mode rayleigh_outage
--cluster-head-degree-weight 0.0
--cluster-head-channel-weight 1.0
--cluster-head-battery-weight 0.0
--optimized-d2d-load-allocation-mode conditional_selective_water_filling
--optimized-d2d-access-floor-fraction 0.02
--optimized-d2d-norm-exponent 3.5
--optimized-d2d-cluster-size-exponent 1.5
--optimized-d2d-freshness-exponent 0.25
--optimized-d2d-load-target-factor 1.1
--optimized-d2d-aoi-exponent 1.0
--optimized-d2d-aoi-threshold-fraction 0.70
```

## Final Results

Final checkpoint values for the optimized ALOHA + D2D curve:

| Run | Mode | Error | Member AoI | Member Stale75 | Member Zero | Energy Efficiency | CH No Attempt | Collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `utility_physical_member_diag_r100` | `utility` | `1.603e-07` | `62.043` | `0.483` | `0.431` | `877.0` | `0.992` | `0.0058` |
| `member_fair_w010_thr070_physical_r100` | `member_fair_utility` | `7.179e-07` | `59.300` | `0.430` | `0.359` | `776.2` | n/a | n/a |
| `member_refresh_w010_floor020_physical_r100` | `member_refresh_utility` | `6.123e-07` | `59.681` | `0.434` | `0.364` | `803.4` | `0.986` | `0.0100` |
| `member_quota_w010_floor000_physical_r100` | `member_quota_utility` | `4.323e-07` | `59.405` | `0.434` | `0.364` | `797.3` | `0.988` | `0.0088` |
| `member_quota_w015_floor000_physical_r100` | `member_quota_utility` | `6.109e-07` | `58.319` | `0.413` | `0.337` | `762.7` | `0.984` | `0.0119` |
| `member_deficit_v3_w015_dw005_decay095_physical_r100` | `member_deficit_utility` | `3.111e-03` | `63.341` | `0.462` | `0.373` | `242.2` | `0.935` | `0.0600` |
| `member_collision_quota_w015_t002_g2_min025_physical_r100` | `member_collision_aware_quota` | `2.663e-07` | `60.872` | `0.461` | `0.401` | `839.4` | `0.991` | `0.0064` |
| `member_collision_quota_w015_t002_g4_min025_physical_r100` | `member_collision_aware_quota` | `2.683e-07` | `60.901` | `0.462` | `0.403` | `839.6` | `0.990` | `0.0072` |
| `member_collision_quota_w015_t003_g4_min025_physical_r100` | `member_collision_aware_quota` | `2.608e-07` | `61.016` | `0.462` | `0.402` | `837.6` | `0.990` | `0.0072` |

`member_deficit_v3` with deficit weights `0.05`, `0.10`, and `0.25` all stayed
near the same poor regime: final error around `3.1e-3`, member stale75 around
`0.463`, and energy efficiency around `241-246`.

## Interpretation

`member_quota_utility` is the useful result. Relative to the utility physical
baseline, `member_quota_w015_floor000_physical_r100` reduces member stale75 from
`0.483` to `0.413` and zero-participation fraction from `0.431` to `0.337`.
It also improves over `member_fair_w010` on member freshness and final error,
with only a small energy-efficiency penalty relative to that member-fair point.

`member_deficit_utility` is the useful negative result. It proves that the
diagnosed failure mode can be moved: `CH no attempt` falls from about `0.984`
to about `0.935`. However, that simply creates an ALOHA collision regime:
collision attribution rises from about `0.012` to about `0.060`, CH uploads
drop, error stagnates near `3e-3`, and energy efficiency collapses.

`member_collision_aware_quota` is a mixed ablation. It successfully avoids the
deficit policy's collision regime: collision attribution stays around
`0.006-0.007`, below the `member_quota_w015` value of about `0.012`. It also
improves final error and energy efficiency relative to `member_quota_w015`.
However, the collision feedback dampens the refresh overlay enough that
member-level freshness regresses: member stale75 rises from `0.413` to about
`0.461-0.462`, and zero-participation fraction rises from `0.337` to about
`0.401-0.403`. This makes it useful as a conservative convergence/energy
ablation, but not as the main member-freshness policy.

This means the current dense physical/Rayleigh regime is not limited only by
"which stale cluster should get more probability". It is also collision limited.
Persistent probability shaping is therefore not enough once the access overlay
becomes too concentrated.

## Current Recommendation

Use `member_quota_utility` as the current member-level freshness contribution.

Recommended points:

- `w=0.10`, `floor=0`: balanced member-freshness improvement with lower error
  cost than the stronger quota point.
- `w=0.15`, `floor=0`: strongest current member-freshness candidate and the
  main point to compare against future access-control ideas.

Do not present `member_deficit_utility` as a main policy. Keep it as an
implemented negative ablation showing that stale-member opportunity pressure can
be over-concentrated and collision dominated.

Do not replace `member_quota_utility` with `member_collision_aware_quota` as the
main member-freshness result. Keep `member_collision_aware_quota` as a documented
ablation showing that feedback can protect convergence/energy, but that simple
global collision damping gives back too much member freshness.

## Next Research Direction

The next implementation should not be another global smooth probability-only
tweak. The results point to one of these directions:

1. Per-cluster or per-refresh-class collision control: keep the member quota
   for the most starved clusters, but cap the base/overlay load locally instead
   of damping the entire refresh overlay from one global EWMA.
2. Scheduled or semi-scheduled refresh: use a small deterministic refresh budget
   for the oldest member-starved clusters instead of pure ALOHA.
3. Re-clustering or cluster splitting: reduce the number of stale members
   competing behind the same sparse CH access opportunities.
4. Virtual queues with collision feedback: update the queue only when added
   access did not collide, so debt does not push too many CHs into the same
   contention interval.

The immediate paper-defensible claim is narrower: member-level D2D freshness
reveals starvation hidden by cluster-level AoI, and a quota-based refresh overlay
improves that member freshness under physical energy/Rayleigh assumptions. The
deficit experiments show the boundary where access opportunity becomes collision
limited.
