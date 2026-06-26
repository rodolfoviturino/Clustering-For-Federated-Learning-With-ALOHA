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
attempt in that round. New runs also split that aggregate CH no-attempt bucket
into CH compute, CH energy, access no-draw, missing required schedule, and
residual no-attempt subcauses, so the next comparison should report both the
aggregate bucket and its subcomposition.

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
attempts are collision-free, but they still require the CH compute draw to pass
`pcomp`, CH battery feasibility, and CH-to-BS link success. If the scheduled CH
is not compute-ready, the reserved slot is wasted; this is a blind scheduler,
not a ready-aware grant. This is the first implemented step beyond global ALOHA
probability shaping, so it is treated as a coordinated upper-bound/future-work
ablation, not as the main ALOHA contribution. Semi-scheduled result rows in this
note that were generated before this `pcomp` gating fix should be treated as
pre-correction diagnostics and re-run before being used as claims.
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

The canonical comparison can be regenerated without rerunning JAX with:

```bash
python -m experiments.run_research_matrix --matrix k3000_core --compare-only
```

This writes `Runs/comparison_k3000_core/run_comparison_summary.*` and
`paper_claim_summary.md`. Use `--execute-missing` only when a listed K=3000 run
is absent and should be launched intentionally.

After regenerating the comparison, export manuscript-facing tables with:

```bash
python -m experiments.export_paper_tables \
  --comparison-csv Runs/comparison_k3000_core/run_comparison_summary.csv
```

This post-processing step writes `paper_results_table.md`,
`paper_results_table.tex`, and `paper_claim_bullets.md` next to the comparison
CSV. The exported table keeps the baseline-relative deltas visible for member
AoI, member stale75, zero-participation, and energy efficiency. When
`run_comparison_summary.csv` contains `final_*_ci95` fields, the table renders
metric cells as `mean +/- ci95`. The claim bullets preserve the current
interpretation: `member_quota_utility` is the main pure-ALOHA baseline,
collision/capped/pressure variants are ablations, and
`semi_scheduled_member_refresh` is a coordinated upper-bound.

Generate the paired paper figures with:

```bash
python -m experiments.plot_research_matrix \
  --comparison-csv Runs/comparison_k3000_core/run_comparison_summary.csv
```

The plotter writes canonical matrix figures for member stale75,
zero-participation, energy efficiency, rounds to `1e-12`, and the
member-stale75/energy-efficiency tradeoff. It uses CI95 error bars when the
comparison CSV contains them. Like the table export, this is a post-processing
step over the comparison CSV and does not rerun JAX.

The paper-facing interpretation text is collected in
`docs/paper_results_section.md`. Use it as the starting point for the manuscript
results section after regenerating the comparison, table, and figures above.
The full manuscript outline is in `docs/paper_manuscript_outline.md`, and the
reproducibility/claim audit checklist is in
`docs/reproducibility_checklist.md`.

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
| `member_safe_split_k3000_s8_min2_b005_w015_physical_r100` | `member_quota_utility` + `safe_max_size=8/min2/budget0.05` | `1.060e-11` | n/a | `74.325` | `0.620` | `0.556` | `643.4` |
| `member_pressure_split_k3000_s8_min2_b005_mw100_ch050_w015_physical_r100` | `member_quota_utility` + `pressure_safe_max_size=8/min2/budget0.05` | `2.663e-13` | `95` | `74.369` | `0.619` | `0.556` | `651.8` |
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
`1e-12`. These capped runs do not change the main claim.

## Minimal K=1000/K=5000 Robustness Check

After the canonical K=3000 package was consolidated, a minimal density
robustness matrix was run with four roles per density: member-quota baseline,
capped-quota pure-ALOHA ablation, global split negative ablation, and
semi-scheduled coordinated upper-bound. These are external-validity checks, not
new strategies.

| Density | Variant | Error | t <= 1e-12 | Member AoI | Member Stale75 | Member Zero | Energy Efficiency | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| K=1000 | `member_quota` | `6.109e-07 +/- 1.978e-07` | n/a | `58.319 +/- 0.415` | `0.413 +/- 0.005` | `0.337 +/- 0.006` | `762.7 +/- 16.6` | Baseline remains the pure-ALOHA freshness reference. |
| K=1000 | `capped_quota cap010` | `2.725e-07 +/- 9.890e-08` | n/a | `61.269 +/- 0.338` | `0.466 +/- 0.005` | `0.407 +/- 0.004` | `851.5 +/- 21.6` | Energy improves, but freshness worsens. |
| K=1000 | `split max8` | `1.559e-06 +/- 7.042e-07` | n/a | `59.046 +/- 0.412` | `0.420 +/- 0.006` | `0.341 +/- 0.006` | `699.4 +/- 18.2` | Negative structural ablation. |
| K=1000 | `semi_scheduled` | `3.360e-09 +/- 8.904e-10` | n/a | `44.194 +/- 0.471` | `0.183 +/- 0.006` | `0.026 +/- 0.003` | `870.3 +/- 9.9` | Strong coordinated upper-bound. |
| K=5000 | `member_quota` | `3.543e-12 +/- 6.942e-12` | n/a | `80.832 +/- 0.237` | `0.701 +/- 0.004` | `0.638 +/- 0.004` | `567.7 +/- 12.0` | Baseline remains the density reference. |
| K=5000 | `capped_quota cap010` | `1.434e-15 +/- 1.817e-15` | `83` | `80.609 +/- 0.252` | `0.699 +/- 0.004` | `0.637 +/- 0.004` | `613.0 +/- 13.4` | Best high-density pure-ALOHA ablation; small freshness gain, larger energy gain. |
| K=5000 | `split max8` | `9.625e-03 +/- 8.486e-03` | n/a | `91.683 +/- 1.237` | `0.860 +/- 0.019` | `0.800 +/- 0.020` | `232.8 +/- 46.9` | Strongly negative structural ablation. |
| K=5000 | `semi_scheduled` | `1.103e-16 +/- 1.236e-17` | `45` | `70.430 +/- 0.216` | `0.533 +/- 0.003` | `0.405 +/- 0.002` | `852.6 +/- 6.1` | Strong coordinated upper-bound. |

The density check strengthens the paper narrative. At K=1000, capped quota is
clearly an energy/convergence ablation because it worsens member freshness. At
K=5000, capped quota becomes more attractive: it improves energy efficiency by
`8.0%` and slightly improves member AoI, stale75, and zero participation, but
the freshness gains are very small compared with the coordinated upper-bound.
The split ablation remains negative at both densities, especially at K=5000.
The semi-scheduled upper-bound remains much stronger than pure ALOHA at both
densities, so it continues to motivate future coordinated MAC work rather than
the current pure-ALOHA claim.

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

The selective structural follow-up is implemented as
`--cluster-split-mode safe_max_size`. It preserves the `max_size` split
mechanics but adds two safety guards: only the largest oversized source rows
inside `--cluster-split-budget-fraction` are considered, and a proposed split
is rejected if any emitted subcluster is smaller than
`--cluster-split-min-subcluster-size`. This keeps the mode inside the pure
ALOHA comparison frame: no reserved slots, no SIC/MPR, no scheduling, and no
change to the multichannel collision abstraction. It should be evaluated as a
selective structural ablation against the quota baseline and the negative
global split points. The first evaluated point, `max_size=8`, minimum emitted
subcluster size `2`, and budget fraction `0.05`, is much safer than global
splitting but still not useful as a paper candidate. Relative to
`member_quota_k3000_w015_floor000_physical_r100`, it adds only `4.02` CH rows
on average and no extra singletons, but final error worsens to `1.06e-11`,
optimized+D2D uploads fall from `2269.79` to `2220.63`, energy efficiency falls
from `653.77` to `643.43`, member AoI rises from `74.230` to `74.325`, stale75
rises from `0.618` to `0.620`, and zero-participation rises from `0.554` to
`0.556`. The failure attribution shifts only slightly: collision attribution
falls from `0.01436` to `0.01355`, but CH-no-attempt rises from `0.98344` to
`0.98416`. Size-only selective splitting is therefore mostly neutral-to-negative
and should not replace quota/collision-quota results.

The next structural candidate is
`--cluster-split-mode pressure_safe_max_size`. It keeps the same ALOHA-safe
split mechanics and tiny-tail rejection, but changes candidate selection from
pure size to a static participation-risk score. The score combines
member-to-CH distance pressure and CH-to-BS channel pressure, with public
weights `--cluster-split-pressure-member-weight` and
`--cluster-split-pressure-ch-weight`. This is intentionally a proxy, not an
oracle: it does not use future member AoI, zero-participation, labels, model
error, or post-run outcomes. The first K=3000 point (`member_weight=1.0`,
`ch_weight=0.5`, `budget=0.05`) is a useful convergence ablation but not a
freshness win. It reaches `1e-12` at round `95`, improves final error versus
the quota baseline (`2.66e-13` versus `4.22e-13`), and avoids the severe
size-only safe-split stall. However, useful uploads remain `0.93%` below the
baseline, final energy efficiency is `0.30%` lower, member AoI is `0.19%`
higher, stale75 is `0.23%` higher, and zero-participation is `0.46%` higher.
Pressure guidance lowers CH-no-attempt attribution slightly (`0.98344` to
`0.98274`) but raises collision attribution (`0.01436` to `0.01513`), so the
member-starvation bottleneck is not solved by static split selection.

## CH No-Attempt Subcause Diagnosis

The `comparison_chdiag_core` runs were generated after the `pcomp` gating fix
and after adding subcauses for the aggregate CH no-attempt bucket. The result is
consistent across `K=1000` and `K=3000`: severe-stale members are still mostly
blocked by CH no-attempt, and roughly 91-93% of that no-attempt bucket is CH
compute unavailability.

| Run | K | CH No Attempt | CH Compute | Access No-Draw | Collision | CH-BS |
|---|---:|---:|---:|---:|---:|---:|
| `chdiag_k1000_member_quota_w015` | 1000 | `0.9842` | `0.9116` | `0.0726` | `0.0119` | `0.0032` |
| `chdiag_k1000_collision_quota` | 1000 | `0.9899` | `0.9050` | `0.0848` | `0.0068` | `0.0026` |
| `chdiag_k1000_capped_quota_cap010` | 1000 | `0.9915` | `0.9056` | `0.0859` | `0.0052` | `0.0026` |
| `chdiag_k1000_semischedule_s020` | 1000 | `0.9922` | `0.9043` | `0.0879` | `0.0055` | `0.0014` |
| `chdiag_k3000_member_quota_w015` | 3000 | `0.9834` | `0.9037` | `0.0797` | `0.0144` | `0.0014` |
| `chdiag_k3000_collision_quota` | 3000 | `0.9839` | `0.9047` | `0.0790` | `0.0138` | `0.0016` |
| `chdiag_k3000_capped_quota_cap010` | 3000 | `0.9851` | `0.9049` | `0.0800` | `0.0127` | `0.0015` |
| `chdiag_k3000_semischedule_s020_cc0010` | 3000 | `0.9875` | `0.9028` | `0.0847` | `0.0105` | `0.0012` |

Within the CH no-attempt bucket, CH compute accounts for `91.1%` to `92.6%`
and local access no-draw accounts for only `7.4%` to `8.9%`. CH energy and
missing required schedule are effectively zero in these runs. This means that
additional ALOHA access-probability shaping has limited headroom under
`pcomp=0.1`: most stale members are waiting on CHs that did not have a computed
aggregate ready, not CHs that were ready but denied channel access.

## Next Research Direction

The current pure-ALOHA probability-shaping branch has likely reached its useful
limit under `pcomp=0.1`: quota, collision-aware damping, capped quota, deficit,
and collision-aware queue variants either improve convergence/energy only or
move the bottleneck into collisions. The CH subcause diagnosis shows that the
dominant member-stale mechanism is CH compute unavailability, not CH energy,
not physical CH-BS failure, and not a missing access draw.

1. Keep `member_quota_k3000_w015_floor000_physical_r100` as the clean
   member-freshness baseline.
2. Keep `member_capped_quota_k3000_w015_cap010_physical_r100` as the best
   pure-ALOHA convergence/energy capped-quota ablation.
3. Keep `member_collision_quota_k3000_w015_t002_g2_min050_physical_r100` as the
   simpler collision-aware convergence/energy ablation.
4. Keep `member_collision_aware_queue_quota` as a negative pure-ALOHA ablation.
5. Run a focused `pcomp` sensitivity sweep before adding more access policies:
   `pcomp=0.05, 0.10, 0.20, 0.50` for `member_quota_utility`,
   `member_collision_aware_quota`, and the corrected semi-scheduled point.
6. Treat current blind `semi_scheduled_member_refresh` as a negative/diagnostic
   coordination ablation for freshness. If coordination is pursued, implement a
   separate ready-aware request/grant protocol that schedules only CHs with a
   ready aggregate and charges explicit control overhead.
7. Keep structural splitting as lower priority. The max-size, safe split, and
   pressure-safe split runs already show that splitting alone does not solve
   member freshness under the current access and compute assumptions.

The immediate paper-defensible claim is now two-tiered: member-level D2D
freshness reveals starvation hidden by cluster-level AoI; a quota-based refresh
overlay improves that member freshness under physical energy/Rayleigh
assumptions while staying in pure ALOHA probability shaping. The semi-scheduled
results are useful as a coordinated diagnostic only after the `pcomp` fix; they
do not replace the quota baseline for member freshness. The deficit experiments
show the boundary where pure ALOHA access opportunity becomes collision
limited, while the CH subcause diagnosis shows that the remaining dominant
constraint is CH compute readiness.
