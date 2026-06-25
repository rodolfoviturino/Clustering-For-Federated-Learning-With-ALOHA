# Paper Manuscript Outline

This outline turns the current implementation and canonical K=3000 results into
a paper structure. It is intentionally conservative: the main contribution stays
inside pure multichannel ALOHA, while coordinated access remains an upper-bound
and future-work signal.

## Working Title

Member-Level Freshness-Aware Probability Shaping for Hierarchical Federated
Learning over Multichannel ALOHA D2D Clusters

Alternative shorter title:

Freshness-Aware ALOHA for D2D-Aided Hierarchical Federated Learning

## One-Sentence Claim

In a dense physical/Rayleigh HFL setting, member-level D2D freshness exposes a
participation-tail problem that cluster-level AoI hides; pure-ALOHA probability
shaping improves convergence and energy tradeoffs but does not fully close the
member-freshness tail, while coordinated refresh provides a clear upper bound
for future MAC designs.

## Draft Abstract

Hierarchical federated learning can reduce uplink pressure by aggregating local
updates inside device-to-device clusters before forwarding them to a base
station. In multichannel ALOHA networks, however, cluster-level delivery metrics
can hide stale member devices whose updates rarely enter a delivered aggregate.
This paper extends a D2D clustering simulator for hierarchical federated
learning with dynamic radio energy, Rayleigh outage links, quality-aware cluster
head selection, and member-level freshness metrics. We evaluate pure-ALOHA
probability-shaping policies under a canonical K=3000 physical/Rayleigh setup.
The member-quota ALOHA policy remains the strongest pure-ALOHA member-freshness
baseline tested, while capped and collision-aware variants improve convergence
and energy efficiency without dominating all member-freshness metrics. Naive
cluster splitting degrades the ALOHA operating point, and a semi-scheduled
member-refresh upper bound shows the remaining gain available to coordinated
MAC designs. The results support a conservative contribution: member-level
freshness should be measured explicitly, and pure-ALOHA policies should be
claimed as probability-shaping tradeoffs rather than as complete solutions to
the D2D participation tail.

## Contribution Bullets

- A member-level D2D freshness accounting layer for HFL simulations, separating
  aggregate/CH freshness from actual per-device participation in delivered D2D
  aggregates.
- A physical/Rayleigh enhanced simulation setup with dynamic first-order-radio
  energy, required-energy feasibility, Rayleigh outage links, and quality-based
  CH selection.
- A pure-ALOHA probability-shaping comparison focused on member freshness,
  convergence, and energy efficiency under a canonical K=3000 setup.
- A set of negative structural ablations showing that naive cluster splitting
  does not solve the member-freshness tail and can damage ALOHA performance.
- A coordinated semi-scheduled upper-bound result that motivates future MAC
  designs without being claimed as part of the pure-ALOHA contribution.

## Paper Structure

### 1. Introduction

Purpose:
Introduce the HFL/D2D/ALOHA setting and the core problem: aggregate-level
success does not guarantee that individual member devices stay fresh.

Key points to cover:

- HFL reduces direct device-to-BS traffic but depends on useful first-tier
  aggregation.
- D2D clusters can improve scalability but introduce a member-selection and
  member-delivery problem.
- Multichannel ALOHA is simple and distributed, but random access can leave
  some members persistently unrefreshed.
- Cluster-level AoI is insufficient because the CH aggregate can be delivered
  even when many members rarely participate.
- The paper focuses on pure ALOHA first, rather than switching to scheduled,
  NOMA, SIC, MPR, or OTA aggregation protocols.

Suggested paragraph close:
The paper asks how far local probability shaping can go before stronger MAC
coordination becomes necessary.

### 2. Related Work

Purpose:
Position the work without expanding the implemented scope.

Suggested subsections:

- Hierarchical federated learning and communication-efficient aggregation.
- AoI and freshness-aware scheduling in federated learning.
- Random access, ALOHA, and D2D aggregation.
- Data-aware or graph-aware D2D clustering as future work, not current scope.

Important framing:
Use recent AoI-FL and selection-guided FL literature as motivation. Keep
non-IID/FedAvg/FedProx/SCAFFOLD, GNN/GCN, OTA-FL, NOMA, IRSA, TDMA/OFDMA, SIC,
and MPR as related/future work unless a separate implementation phase is added.

### 3. System Model

Purpose:
Define the simulation model tightly enough that the experiments are
reproducible.

Include:

- Device deployment around one BS.
- D2D clustering and CH/member roles.
- HFL update flow: member updates to CH, CH aggregate to BS.
- Multichannel ALOHA abstraction: probabilistic CH attempts and collisions.
- Physical/Rayleigh link success modes.
- Energy model and required-energy feasibility.
- CH quality selection based on channel score.

Keep separate:

- Thesis-compatible defaults.
- Enhanced physical/Rayleigh setup used for the paper-facing K=3000 matrix.

### 4. Member-Level Freshness Metrics

Purpose:
Explain why the paper uses member-level metrics.

Definitions:

- Cluster/aggregate AoI: freshness of a delivered aggregate/CH row.
- Member AoI: freshness of an individual device update included in a delivered
  D2D aggregate.
- Member stale75: fraction of members with AoI above 75 rounds.
- Member zero: fraction of members with no successful delivered participation.

Core argument:
Cluster AoI can reset when the CH delivers an aggregate, even if non-CH members
were not included. Member-level AoI is therefore required to measure the actual
freshness of the D2D contribution.

### 5. Pure-ALOHA Probability Shaping

Purpose:
Describe the strategy family without overclaiming novelty beyond what is
implemented.

Modes to discuss:

- `member_quota_utility`: main pure-ALOHA baseline.
- `member_collision_aware_quota`: collision-aware ablation.
- `member_capped_quota_utility`: capped quota ablation.
- `pressure_safe_max_size`: structural convergence ablation.
- `max_size` and size-only safe split: negative structural ablations.
- `semi_scheduled_member_refresh`: coordinated upper-bound, not pure ALOHA.

Pure-ALOHA definition:
CHs use local probabilistic access decisions. The system does not reserve slots,
does not use SIC, MPR, NOMA, TDMA/OFDMA, OTA aggregation, or centralized
scheduling for the main claim.

### 6. Experimental Setup

Purpose:
Report the canonical matrix and reproducibility boundary.

Canonical matrix:
Use `experiments.run_research_matrix --matrix k3000_core`.

Setup:

- `K=3000`;
- `rounds=100`;
- `iterations=100`;
- dynamic energy drain;
- `first_order_radio`;
- required-energy feasibility;
- Rayleigh outage for D2D member links, CH-BS links, and device-BS links;
- quality CH selection using Rayleigh channel score;
- optimized D2D load allocation with conditional selective water filling.

Artifacts:

- Table: `Runs/comparison_k3000_core/paper_results_table.tex`.
- Figures:
  - `research_matrix_member_stale75.pdf`;
  - `research_matrix_member_zero_participation.pdf`;
  - `research_matrix_energy_efficiency.pdf`;
  - `research_matrix_t_to_1e12.pdf`;
  - `research_matrix_stale75_energy_pareto.pdf`.

### 7. Results

Purpose:
Use `docs/paper_results_section.md` as the primary text source.

Recommended figure/table order:

1. Table with all canonical runs and CI95.
2. Member stale75 bar figure.
3. Zero-participation bar figure.
4. Energy efficiency and convergence figures.
5. Pareto scatter: member stale75 vs energy efficiency.

Main interpretation:

- `member_quota_utility` is the main pure-ALOHA member-freshness baseline.
- Collision-aware and capped quota improve convergence/energy but do not
  dominate the baseline on all member freshness metrics.
- Naive cluster splitting worsens the dense ALOHA operating point.
- Semi-scheduled refresh demonstrates the upper-bound value of coordination but
  is outside the pure-ALOHA contribution.

### 8. Limitations

Purpose:
Keep claims scientifically defensible.

State explicitly:

- Results are for the canonical K=3000 physical/Rayleigh setup.
- The learning task is synthetic and not yet a non-IID FL benchmark.
- No FedAvg/FedProx/SCAFFOLD comparison is implemented in this phase.
- No GNN/GCN clustering, data-aware clustering, OTA-FL, NOMA, IRSA, SIC, MPR,
  TDMA/OFDMA, or full scheduling is part of the main implementation.
- Semi-scheduled refresh is an upper-bound/future-work reference.

### 9. Conclusion

Purpose:
Close with a bounded contribution.

Suggested close:
The results show that member-level freshness is a necessary diagnostic for
D2D-aided HFL over ALOHA, and that local probability shaping can improve some
tradeoffs without fully solving the participation tail. The remaining gap to
the coordinated upper bound motivates future MAC designs and graph/data-aware
member discovery, but the current contribution remains a reproducible
pure-ALOHA analysis.

## Text Assets To Reuse

- Results section draft: `docs/paper_results_section.md`.
- Modeling assumptions: `docs/modeling_assumptions.md`.
- Member-level experiment history:
  `docs/member_level_d2d_freshness_experiments.md`.
- Current status and future work: `docs/current_status_and_future_work.md`.
- Architecture boundaries: `docs/project_architecture.md`.

## Do Not Claim

- Do not claim semi-scheduled refresh as the main ALOHA contribution.
- Do not claim support for non-IID learning unless implemented in a later phase.
- Do not claim optimal clustering or scheduling.
- Do not claim that capped/collision-aware quota is a new member-freshness
  winner; the current evidence supports convergence/energy ablation claims.
- Do not mix pure-ALOHA probability shaping with scheduled/NOMA/SIC/MPR
  protocol claims.
