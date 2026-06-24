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
refresh quota. This is the current best pure ALOHA/probability-shaping
member-freshness candidate.

`member_collision_aware_quota`

Keeps the quota structure but reduces the reserved member-refresh overlay when
an EWMA of optimized-D2D CH collisions exceeds a configured target. This is the
first follow-up after the negative deficit result and should be compared against
`member_quota_w015_floor000_physical_r100`.

`semi_scheduled_member_refresh`

Reserves a small number of D2D channels for the highest active member-pressure
clusters and runs utility ALOHA on the remaining channels. The scheduled CH
attempts are collision-free, but they still require CH battery feasibility and
CH-to-BS link success. This is the first implemented step beyond global ALOHA
probability shaping. `--optimized-d2d-member-schedule-control-cost` can charge
a normalized per-scheduled-CH coordination overhead, so the semi-scheduled
claim can be stress-tested against nonzero control-plane cost.

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
| `member_semischedule_s010_dw000_physical_r100` | `semi_scheduled_member_refresh` | `1.744e-08` | `54.010` | `0.358` | `0.283` | `1053.4` | `0.992` | `0.0031` |
| `member_semischedule_s020_dw000_physical_r100` | `semi_scheduled_member_refresh` | `6.209e-09` | `47.842` | `0.253` | `0.138` | `1040.6` | `0.984` | `0.0055` |
| `member_semischedule_s010_dw025_physical_r100` | `semi_scheduled_member_refresh` | `1.528e-08` | `53.928` | `0.357` | `0.282` | `1053.6` | `0.992` | `0.0032` |

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

`semi_scheduled_member_refresh` is the strongest result so far, with an explicit
scope caveat. It is no longer pure distributed ALOHA probability shaping: it
assumes a small BS/CH control decision to reserve refresh slots. With that
coordination, the `s=0.20`, `deficit_weight=0` run dominates the previous
member-freshness candidates in this physical/Rayleigh setup: final error drops
from `6.109e-7` for `member_quota_w015` to `6.209e-9`, member AoI drops from
`58.319` to `47.842`, member stale75 from `0.413` to `0.253`, and
zero-participation fraction from `0.337` to `0.138`. Energy efficiency also
improves from `762.7` to `1040.6`, because reserved collision-free attempts
increase useful CH deliveries instead of spending energy on collided ALOHA
contention. The main cost is conceptual rather than numerical: it changes the
protocol class from random access to semi-scheduled access.

The first coordination-overhead sweep keeps that conclusion intact. Adding
`--optimized-d2d-member-schedule-control-cost` at `0.0001`, `0.0005`, and
`0.0010` to the `s=0.20`, `deficit_weight=0` point barely changes convergence
or member freshness: final error stays near `6.1e-9`, member stale75 stays near
`0.253`, and zero-participation fraction stays near `0.138`. Energy efficiency
falls monotonically from `1040.6` with no overhead to `1028.8`, `983.5`, and
`932.2`, respectively, but even the largest tested overhead remains above
`member_quota_w015` (`762.7`). This supports the claim that the semi-scheduled
gain is not erased by a small normalized control-plane cost.

The schedule-fraction sweep with `control_cost=0.0010` shows the expected
integer-channel steps because the simulator reserves
`ceil(M * schedule_fraction)` channels and these runs use `M=10`. Fractions
`0.05` and `0.10` both reserve one channel; `0.15` and `0.20` both reserve two;
`0.25` and `0.30` both reserve three. One reserved channel already improves over
`member_quota_w015` (`member_stale75=0.358`, `zero=0.283`,
`energy_efficiency=990.7`). Two channels recover the previous `s=0.20`
freshness regime with overhead (`member_stale75=0.253`, `zero=0.138`,
`energy_efficiency=932.2`). Three channels are the strongest tested
freshness/error point (`final_error=3.36e-9`, `member_stale75=0.183`,
`zero=0.026`, `energy_efficiency=870.3`) and still remain above
`member_quota_w015` on energy efficiency. Thus, under `M=10` and
`control_cost=0.0010`, the practical choices are one channel for a conservative
energy/freshness tradeoff, two channels for the balanced point, and three
channels for the strongest member-freshness point.

This means the current dense physical/Rayleigh regime is not limited only by
"which stale cluster should get more probability". It is also collision limited.
Persistent probability shaping is therefore not enough once the access overlay
becomes too concentrated.

## Current Recommendation

Use `member_quota_utility` as the current pure ALOHA/probability-shaping
member-level freshness contribution.

Recommended points:

- `w=0.10`, `floor=0`: balanced member-freshness improvement with lower error
  cost than the stronger quota point.
- `w=0.15`, `floor=0`: strongest current pure ALOHA/probability-shaping
  member-freshness candidate and the main point to compare against coordinated
  access-control ideas.

Do not present `member_deficit_utility` as a main policy. Keep it as an
implemented negative ablation showing that stale-member opportunity pressure can
be over-concentrated and collision dominated.

Do not replace `member_quota_utility` with `member_collision_aware_quota` as the
main member-freshness result. Keep `member_collision_aware_quota` as a documented
ablation showing that feedback can protect convergence/energy, but that simple
global collision damping gives back too much member freshness.

Use `semi_scheduled_member_refresh` as the current strongest enhanced-policy
candidate when coordinated refresh slots are allowed. With `M=10`, the
two-channel point (`schedule_fraction` `0.15` or `0.20`) is the balanced
candidate, while the three-channel point (`0.25` or `0.30`) is the strongest
tested member-freshness/error candidate. The `0.10` deficit-weight variant
changed little, so persistent deficit is not needed for the first paper claim.
For overhead-sensitive reporting, include the control-cost sweep through
`--optimized-d2d-member-schedule-control-cost 0.0010`: both the two-channel and
three-channel points still dominate the best pure ALOHA/probability-shaping
member policy on member freshness and energy efficiency.

## Next Research Direction

The next experiments should validate whether the semi-scheduled gain is robust
outside the current `K=1000`, `M=10`, `rounds=100` setting:

1. Re-run the best two-channel and three-channel points at larger `K` and/or
   more rounds to verify that the dominance is not a `K=1000`, `rounds=100`
   artifact.
2. Sweep larger nonzero `--optimized-d2d-member-schedule-control-cost` values
   above `0.0010` if a break-even overhead is needed; the tested range up to
   `0.0010` still keeps energy efficiency above `member_quota_w015`.
3. Per-cluster or per-refresh-class collision control: keep the member quota
   for the most starved clusters, but cap the base/overlay load locally instead
   of damping the entire refresh overlay from one global EWMA.
4. Re-clustering or cluster splitting: reduce the number of stale members
   competing behind the same sparse CH access opportunities.
5. Virtual queues with collision feedback: update the queue only when added
   access did not collide, so debt does not push too many CHs into the same
   contention interval.

The immediate paper-defensible claim is now two-tiered: member-level D2D
freshness reveals starvation hidden by cluster-level AoI; a quota-based refresh
overlay improves that member freshness under physical energy/Rayleigh
assumptions while staying in pure ALOHA probability shaping; and a small
semi-scheduled refresh budget improves much further when lightweight
coordination is allowed. The deficit experiments show the boundary where access
opportunity becomes collision limited.
