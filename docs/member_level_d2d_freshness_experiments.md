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

`member_capped_quota_utility`

Uses the same base utility plus member-refresh overlay as
`member_quota_utility`, but caps the overlay per cluster before adding it to
the base probability. The cap is controlled by
`--optimized-d2d-member-quota-cap-fraction` and is measured as a fraction of
the fixed D2D ALOHA access probability. This is a pure ALOHA load-cap
candidate: CHs still draw local access attempts, and multichannel ALOHA
collisions remain the MAC abstraction.

`member_collision_aware_quota`

Keeps the quota structure but reduces the reserved member-refresh overlay when
an EWMA of optimized-D2D CH collisions exceeds a configured target. This is the
first follow-up after the negative deficit result and should be compared against
`member_quota_w015_floor000_physical_r100`.

`member_collision_aware_queue_quota`

Keeps the same quota/deficit access structure, but changes the persistent
state update: no-attempt misses add full stale-member pressure, CH-BS misses
add only a small increment, and collision-caused misses add no debt. This keeps
the policy inside pure local ALOHA probability shaping. The first K=3000 run is
negative because the queue still concentrates access enough to increase
collisions and degrade convergence/freshness.

`semi_scheduled_member_refresh`

Reserves a small number of D2D channels for the highest active member-pressure
clusters and runs utility ALOHA on the remaining channels. The scheduled CH
attempts are collision-free, but they still require CH battery feasibility and
CH-to-BS link success. This is the first implemented step beyond global ALOHA
probability shaping, so it is treated as a coordinated upper-bound/future-work
ablation, not as the main ALOHA contribution.
`--optimized-d2d-member-schedule-control-cost` can charge a normalized
per-scheduled-CH coordination overhead,
so the semi-scheduled bound can be stress-tested against nonzero control-plane
cost.

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

`semi_scheduled_member_refresh` is the strongest coordinated upper-bound result
so far, with an explicit scope caveat. It is no longer pure distributed ALOHA
probability shaping: it assumes a small BS/CH control decision to reserve
refresh slots. With that coordination, the `s=0.20`, `deficit_weight=0` run
dominates the previous member-freshness candidates in this physical/Rayleigh
setup: final error drops from `6.109e-7` for `member_quota_w015` to `6.209e-9`,
member AoI drops from `58.319` to `47.842`, member stale75 from `0.413` to
`0.253`, and zero-participation fraction from `0.337` to `0.138`. Energy
efficiency also improves from `762.7` to `1040.6`, because reserved
collision-free attempts increase useful CH deliveries instead of spending
energy on collided ALOHA contention. This result is useful as an upper bound
and motivation for future coordinated protocols, but it should not be used as
the main ALOHA claim.

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

The first `K=3000`, `rounds=100`, `iterations=100` robustness check reran the
two-channel and three-channel semi-scheduled points with the same physical
Rayleigh settings and `control_cost=0.0010`. The fair same-`K` baseline is
`member_quota_k3000_w015_floor000_physical_r100`, which uses the strongest
pure ALOHA/probability-shaping member-quota point identified at `K=1000`.

| Run | Reserved Channels | Error | t <= 1e-12 | Member AoI | Member Stale75 | Member Zero | Energy Efficiency | CH No Attempt | Collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `member_quota_k3000_w015_floor000_physical_r100` | 0 | `4.216e-13` | `100` | `74.230` | `0.618` | `0.554` | `653.8` | `0.983` | `0.0144` |
| `member_semischedule_k3000_s020_cc0010_physical_r100` | 2 | `2.366e-16` | `66` | `66.287` | `0.482` | `0.387` | `809.4` | `0.984` | `0.0128` |
| `member_semischedule_k3000_s030_cc0010_physical_r100` | 3 | `2.093e-16` | `59` | `63.863` | `0.437` | `0.312` | `825.7` | `0.983` | `0.0117` |

Against the `K=3000` member-quota baseline, the two-channel semi-scheduled
point reaches `1e-12` 34 rounds earlier, lowers member AoI by `10.7%`, lowers
member stale75 by `22.0%`, lowers zero participation by `30.2%`, and improves
final energy efficiency by `23.8%`. The three-channel point is stronger:
`1e-12` is reached 41 rounds earlier, member AoI falls by `14.0%`, member
stale75 by `29.2%`, zero participation by `43.6%`, and final energy efficiency
by `26.3%`. The energy efficiency gain happens despite `11.0%` more final
energy used because the scheduled policy produces many more useful uploads:
final CH uploads rise from `300.73` to `460.48`.

Within `K=3000`, the three-channel point also dominates the two-channel point
on the tested metrics: faster convergence to `1e-12`, lower final error, lower
member AoI/stale/zero fractions, slightly lower collision attribution, and
higher energy efficiency. The remaining caveat is that absolute member
freshness is still worse than in the `K=1000` semi-scheduled runs. That is
plausible rather than surprising: with the default `M=10`, the policy reserves
only two or three scheduled refresh slots per round while the clustered
population is about three times larger. The member-failure breakdown remains
dominated by `CH no attempt`, so the dense-scale bottleneck is still refresh
opportunity budget, not member compute, member link, member energy, or CH-BS
decoding.

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
global collision damping gives back too much member freshness. The `K=3000`
follow-up with less aggressive damping found a better convergence/energy
ablation, but still did not produce a material member-freshness win.

Use `semi_scheduled_member_refresh` only as a coordinated upper-bound/future-work
reference for the current paper path. With `M=10`, the two-channel point
(`schedule_fraction` `0.15` or `0.20`) is the balanced coordinated reference,
while the three-channel point (`0.25` or `0.30`) is the strongest tested
coordinated reference. The `0.10` deficit-weight variant changed little, so
persistent deficit is not needed for this reference. Do not develop larger
scheduled budgets, NOMA, IRSA, AirFL, TDMA/OFDMA, SIC, or multi-packet
reception in the current ALOHA-focused phase.

## K=3000 Pure-ALOHA Matrix

The next pure-ALOHA matrix was run at `K=3000`, `rounds=100`,
`iterations=100`, and the same physical/Rayleigh setup. It compared stronger
member quota weights, small quota floors, and less aggressive
collision-aware damping against `member_quota_k3000_w015_floor000_physical_r100`.
The later capped-quota and collision-aware-queue runs are also included below.
The coordinated semi-scheduled point is kept only as an upper-bound row.

| Run | Mode | Error | t <= 1e-12 | Member AoI | Member Stale75 | Member Zero | Energy Efficiency |
|---|---:|---:|---:|---:|---:|---:|---:|
| `member_quota_k3000_w015_floor000_physical_r100` | `member_quota_utility` | `4.216e-13` | `100` | `74.230` | `0.618` | `0.554` | `653.8` |
| `member_quota_k3000_w020_floor000_physical_r100` | `member_quota_utility` | `1.102e-12` | n/a | `74.380` | `0.618` | `0.553` | `637.0` |
| `member_quota_k3000_w025_floor000_physical_r100` | `member_quota_utility` | `3.256e-11` | n/a | `74.723` | `0.624` | `0.555` | `604.3` |
| `member_quota_k3000_w015_floor002_physical_r100` | `member_quota_utility` | `1.153e-13` | `92` | `74.212` | `0.617` | `0.554` | `654.9` |
| `member_quota_k3000_w020_floor002_physical_r100` | `member_quota_utility` | `1.167e-12` | n/a | `74.391` | `0.618` | `0.553` | `636.5` |
| `member_collision_quota_k3000_w015_t002_g1_min050_physical_r100` | `member_collision_aware_quota` | `7.649e-14` | `93` | `73.971` | `0.615` | `0.556` | `681.4` |
| `member_collision_quota_k3000_w015_t002_g1_min075_physical_r100` | `member_collision_aware_quota` | `5.388e-14` | `92` | `74.164` | `0.619` | `0.557` | `668.4` |
| `member_collision_quota_k3000_w015_t002_g2_min050_physical_r100` | `member_collision_aware_quota` | `1.816e-14` | `88` | `73.968` | `0.614` | `0.556` | `688.7` |
| `member_collision_quota_k3000_w015_t002_g2_min075_physical_r100` | `member_collision_aware_quota` | `6.388e-14` | `92` | `74.234` | `0.620` | `0.557` | `670.5` |
| `member_capped_quota_k3000_w015_cap005_physical_r100` | `member_capped_quota_utility` | `1.730e-14` | `88` | `74.265` | `0.621` | `0.565` | `742.1` |
| `member_capped_quota_k3000_w015_cap008_physical_r100` | `member_capped_quota_utility` | `1.269e-14` | `86` | `74.159` | `0.618` | `0.563` | `722.2` |
| `member_capped_quota_k3000_w015_cap010_physical_r100` | `member_capped_quota_utility` | `1.893e-14` | `87` | `74.068` | `0.616` | `0.561` | `715.7` |
| `member_capped_quota_k3000_w015_cap012_physical_r100` | `member_capped_quota_utility` | `2.019e-14` | `88` | `74.063` | `0.617` | `0.558` | `711.5` |
| `member_capped_quota_k3000_w015_cap015_physical_r100` | `member_capped_quota_utility` | `3.978e-14` | `89` | `74.161` | `0.618` | `0.557` | `689.7` |
| `member_capped_quota_k3000_w015_cap025_physical_r100` | `member_capped_quota_utility` | `4.185e-13` | `100` | `74.191` | `0.617` | `0.553` | `653.5` |
| `member_capped_quota_k3000_w020_cap010_physical_r100` | `member_capped_quota_utility` | `1.030e-11` | n/a | `74.410` | `0.623` | `0.565` | `726.6` |
| `member_capped_quota_k3000_w020_cap012_physical_r100` | `member_capped_quota_utility` | `1.170e-13` | `93` | `74.317` | `0.621` | `0.561` | `722.7` |
| `member_queue_quota_k3000_w015_qw005_decay095_physical_r100` | `member_collision_aware_queue_quota` | `3.126e-01` | n/a | `84.230` | `0.753` | `0.689` | `394.3` |
| `member_split_k3000_s8_w015_physical_r100` | `member_quota_utility` + `cluster_split_max_size=8` | `1.253e-07` | n/a | `78.167` | `0.670` | `0.604` | `506.3` |
| `member_split_k3000_s5_w015_physical_r100` | `member_quota_utility` + `cluster_split_max_size=5` | `3.120e-02` | n/a | `93.788` | `0.891` | `0.835` | `103.9` |
| `member_semischedule_k3000_s030_cc0010_physical_r100` | `semi_scheduled_member_refresh` | `2.093e-16` | `59` | `63.863` | `0.437` | `0.312` | `825.7` |

The conclusion is conservative. Increasing the pure quota from `w=0.15` to
`w=0.20` or `w=0.25` does not improve member freshness and hurts convergence
and energy efficiency. Adding a small `floor=0.02` to `w=0.15` is nearly
neutral and slightly improves convergence, but the member-freshness change is
too small to claim. The best pure-ALOHA follow-up is
`member_collision_quota_k3000_w015_t002_g2_min050_physical_r100`: relative to
the K=3000 quota baseline, it reaches `1e-12` 12 rounds earlier, improves final
energy efficiency by `5.34%`, lowers member AoI by `0.35%`, and lowers
member stale75 by `0.51%`. However, zero-participation is slightly worse
(`0.556` versus `0.554`), so this is a convergence/energy ablation rather than
a new member-freshness winner.

The capped quota follow-up is informative but not a new member-freshness
winner. Wide caps (`0.50`, `0.75`, `1.00`) were byte-identical to
`member_quota_k3000_w015_floor000_physical_r100`, so the cap did not bind. Low
caps do bind. The best convergence point is
`member_capped_quota_k3000_w015_cap010_physical_r100`: it reaches `1e-12` 13
rounds earlier than the quota baseline, improves final energy efficiency by
`9.48%`, lowers member AoI by `0.22%`, and lowers stale75 by `0.19%`. However,
it worsens zero-participation by `1.22%`, so it is a convergence/energy
ablation, not a replacement for the member-freshness baseline. The most
freshness-balanced capped point is `cap025`, which slightly improves member AoI,
stale75, and zero-participation, but the gains are all below `0.15%` and it does
not improve the time to `1e-12`. Increasing the quota to `w=0.20` with caps
`0.10`/`0.12` worsens member freshness and, for `cap010`, fails to reach
`1e-12`.

The collision-aware queue follow-up is a clear negative ablation. The tested
point, `member_queue_quota_k3000_w015_qw005_decay095_physical_r100`, lowers
the stale-member `CH no attempt` attribution from `0.983` to `0.942`, but moves
that pressure into ALOHA collisions: collision attribution rises from `0.014`
to `0.057`. Final useful uploads fall from `2269.8` to `1517.6`, final error
stalls at `0.313`, member stale75 rises from `0.618` to `0.753`, and
zero-participation rises from `0.554` to `0.689`. Excluding collision-caused
misses from the queue update was not enough; stateful debt ranking still
over-concentrates attempts in dense K=3000 ALOHA.

The first structural cluster split test is also negative. With
`--cluster-split-mode max_size --cluster-split-max-size 5`, the number of
cluster rows rises from `627.45` to `905.66`, singletons rise from `20.44` to
`158.43`, and mean non-singleton cluster size falls from `4.91` to `3.80`.
This reduces stale `CH no attempt` attribution from `0.983` to `0.944`, but it
creates too many ALOHA contenders: collision attribution rises to `0.055`, CH
uploads fall from `300.73` to `124.28`, final error stalls at `3.12e-2`, and
member stale75/zero worsen sharply. Naive fixed max-size splitting is therefore
not a member-freshness solution. The gentler `max_size=8` run is less damaging
but still negative: cluster rows rise to `706.45`, singletons to `47.41`,
optimized+D2D uploads fall by `20.8%`, final energy efficiency falls by
`22.6%`, final error only reaches `1.25e-7`, and member stale75/zero still
worsen from `0.618`/`0.554` to `0.670`/`0.604`. This effectively rules out
global fixed-threshold splitting as the next main path.

## Next Research Direction

The current pure-ALOHA probability-shaping branch has likely reached its useful
limit: quota, collision-aware damping, capped quota, deficit, and
collision-aware queue variants either improve convergence/energy only or move
the bottleneck into collisions. The next ALOHA-compatible steps should be more
structural:

1. Keep `member_quota_k3000_w015_floor000_physical_r100` as the clean
   member-freshness baseline.
2. Keep `member_capped_quota_k3000_w015_cap010_physical_r100` as the best
   pure-ALOHA convergence/energy capped-quota ablation.
3. Keep `member_collision_quota_k3000_w015_t002_g2_min050_physical_r100` as the
   simpler collision-aware convergence/energy ablation.
4. Keep `member_collision_aware_queue_quota` as a negative pure-ALOHA ablation.
5. Move next to re-clustering or cluster splitting, because persistent
   zero-participation suggests that some members are structurally hidden behind
   overloaded or unlucky CHs.
   The first implemented structural ablation is
   `--cluster-split-mode max_size`, which locally re-clusters large D2D rows
   into valid one-hop subclusters before FL rounds while keeping ALOHA access
   unchanged. The first `max_size=5` and `max_size=8` runs are negative, so
   the next structural attempt should be selective rather than globally
   splitting every large cluster.

The immediate paper-defensible claim is now two-tiered: member-level D2D
freshness reveals starvation hidden by cluster-level AoI; a quota-based refresh
overlay improves that member freshness under physical energy/Rayleigh
assumptions while staying in pure ALOHA probability shaping. The semi-scheduled
results are useful only as an upper-bound reference showing that additional
coordination could improve further, while the deficit experiments show the
boundary where pure ALOHA access opportunity becomes collision limited. The
collision-aware queue result reinforces that the remaining problem is likely
structural cluster membership, not just a missing access-probability weight.
