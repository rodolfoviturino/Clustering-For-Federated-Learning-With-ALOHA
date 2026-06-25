# Draft Results Section: Canonical K=3000 ALOHA Matrix

This draft is a paper-oriented results section generated from the canonical
comparison in `Runs/comparison_k3000_core`. It should be treated as source text
for manuscript writing, not as a new simulation result. Regenerate the source
artifacts before editing numeric claims:

```bash
python -m experiments.run_research_matrix --matrix k3000_core --compare-only
python -m experiments.export_paper_tables --comparison-csv Runs/comparison_k3000_core/run_comparison_summary.csv
python -m experiments.plot_research_matrix --comparison-csv Runs/comparison_k3000_core/run_comparison_summary.csv
```

The table source is `Runs/comparison_k3000_core/paper_results_table.tex`. The
figure sources are:

- `Runs/comparison_k3000_core/research_matrix_member_stale75.pdf`;
- `Runs/comparison_k3000_core/research_matrix_member_zero_participation.pdf`;
- `Runs/comparison_k3000_core/research_matrix_energy_efficiency.pdf`;
- `Runs/comparison_k3000_core/research_matrix_t_to_1e12.pdf`;
- `Runs/comparison_k3000_core/research_matrix_stale75_energy_pareto.pdf`.

## Experimental Setup

The canonical evaluation uses the physical/Rayleigh `K=3000` configuration with
`100` FL rounds and `100` Monte Carlo iterations. The setup enables dynamic
energy drain, the `first_order_radio` energy model, required-energy feasibility,
Rayleigh outage models for device-to-BS, CH-to-BS, and D2D member links, and
quality-based CH selection using the Rayleigh channel score. All variants use
the same optimized-D2D baseline configuration unless explicitly stated by the
matrix entry.

The main comparison keeps the MAC claim inside pure multichannel ALOHA:
cluster heads make local probabilistic access decisions, and collisions remain
the interference abstraction. The semi-scheduled member-refresh point is
included only as a coordinated upper-bound and future-work reference; it is not
claimed as part of the pure-ALOHA contribution.

## Metrics

The primary learning metric is the final error norm and the first round that
reaches `1e-12`. Energy is summarized by final energy efficiency. Freshness is
reported at the D2D-member level because aggregate or cluster-level AoI can hide
members that almost never enter a delivered aggregate.

The key freshness metrics are:

- `member AoI`: final mean AoI across D2D member devices;
- `member stale75`: fraction of D2D member devices with AoI above 75 rounds;
- `member zero`: fraction of D2D member devices with zero successful
  participation in delivered D2D aggregates.

All reported cells use mean with 95% confidence interval (`mean +/- ci95`) when
the comparison CSV contains the corresponding CI95 column.

## Main Result

Table `Runs/comparison_k3000_core/paper_results_table.tex` summarizes the
canonical K=3000 comparison. The member-quota policy
`member_quota_k3000_w015_floor000_physical_r100` remains the main pure-ALOHA
baseline. It reaches final error `4.216e-13 +/- 7.177e-13`, member AoI
`74.230 +/- 0.295`, member stale75 `0.618 +/- 0.004`, zero participation
`0.554 +/- 0.004`, and energy efficiency `653.8 +/- 10.6`.

No tested pure-ALOHA ablation dominates this baseline simultaneously on member
AoI, member stale75, and zero participation. This is the central conservative
claim: the implemented probability-shaping extensions identify useful
tradeoffs, but none should replace the member-quota baseline as the main
member-freshness result under the current K=3000 physical/Rayleigh setup.

## Pure-ALOHA Ablations

The collision-aware quota ablation improves convergence and energy but does not
strictly improve the full member-freshness vector. It reaches `1e-12` in
`88` rounds instead of `100`, lowers member AoI from `74.230 +/- 0.295` to
`73.968 +/- 0.266`, and improves energy efficiency from `653.8 +/- 10.6` to
`688.7 +/- 8.4`. However, zero participation increases from
`0.554 +/- 0.004` to `0.556 +/- 0.004`. This supports a narrow claim:
collision-aware probability shaping can improve convergence and energy
efficiency, but does not solve the zero-participation tail.

The capped-quota ablation is the strongest convergence/energy point among the
tested pure-ALOHA ablations. It reaches `1e-12` in `87` rounds, `13` rounds
before the quota baseline, and has the highest pure-ALOHA energy efficiency in
the matrix: `715.7 +/- 9.9`, a `+9.5%` gain relative to baseline. Its member
stale75 is slightly lower (`0.616 +/- 0.004` versus `0.618 +/- 0.004`), but
zero participation worsens to `0.561 +/- 0.004`. Therefore, it should be
reported as a convergence/energy ablation, not as the main freshness result.

The pressure-guided safe split recovers convergence relative to naive cluster
splitting, reaching `1e-12` at round `95`, but it does not improve member
freshness or energy efficiency against the member-quota baseline. Its member
AoI is `74.369 +/- 0.299`, member stale75 is `0.619 +/- 0.005`, zero
participation is `0.556 +/- 0.004`, and energy efficiency is
`651.8 +/- 9.6`.

## Negative Structural Ablations

The structural split ablations are important because they rule out the
interpretation that simply reducing cluster size is sufficient. The safe
size-only split does not reach the `1e-12` target in the saved horizon and
slightly worsens member stale75 and zero participation. The global split with
maximum size `8` is worse on all headline freshness and efficiency metrics:
member AoI rises to `78.167 +/- 0.631`, member stale75 rises to
`0.670 +/- 0.009`, zero participation rises to `0.604 +/- 0.008`, and energy
efficiency falls to `506.3 +/- 31.6`.

The aggressive global split with maximum size `5` is a clear negative ablation.
It ends with final error `3.120e-02 +/- 1.592e-02`, member AoI
`93.788 +/- 0.813`, member stale75 `0.891 +/- 0.012`, zero participation
`0.835 +/- 0.015`, and energy efficiency `103.9 +/- 27.2`. This confirms that
more CH contenders and smaller clusters can severely damage the ALOHA operating
point.

## Coordinated Upper Bound

The semi-scheduled member-refresh variant is intentionally separated from the
pure-ALOHA claim. It adds coordinated, collision-free refresh opportunities and
therefore changes the MAC assumption. Its result is useful as an upper bound:
it reaches `1e-12` by round `59`, reduces member AoI to `63.863 +/- 0.229`,
reduces member stale75 to `0.437 +/- 0.003`, reduces zero participation to
`0.312 +/- 0.003`, and increases energy efficiency to `825.7 +/- 6.4`.

Relative to the quota baseline, this corresponds to a `29.2%` reduction in
member stale75 and a `43.6%` reduction in zero participation. The result shows
that coordinated access can substantially improve the member-freshness tail,
but it should be framed as future work or an upper-bound reference, not as the
main contribution of the ALOHA-focused study.

## Interpretation

The results support a narrow and defensible story. Within pure multichannel
ALOHA, member-quota probability shaping is the strongest member-freshness
baseline tested so far. Collision-aware and capped-quota variants improve
convergence and energy efficiency, but they do not eliminate the
zero-participation tail. Structural cluster splitting is not sufficient and can
be harmful when it increases the number of contenders in a dense ALOHA regime.
The coordinated upper bound indicates that the remaining freshness gap is
primarily a MAC-access limitation rather than only a clustering or scoring
problem.

## Threats To Validity

These results are based on a single canonical K=3000 physical/Rayleigh setup
with `100` FL rounds and `100` Monte Carlo iterations. The comparison is
designed to be reproducible and internally consistent, but the ranking may
change under different device densities, channel counts, mobility assumptions,
data heterogeneity, or non-convex learning tasks. The current model also keeps
the learning objective synthetic and does not yet include non-IID data,
FedAvg/FedProx/SCAFFOLD-style optimizer differences, GNN-based clustering, or
over-the-air aggregation.

The pure-ALOHA claim should therefore be limited to the simulated regime:
local probabilistic CH access with multichannel collision modeling, Rayleigh
outage links, dynamic first-order-radio energy accounting, and member-level D2D
freshness metrics.

## Next Work

The next paper-oriented step is not another ALOHA variant. The immediate need
is to convert this draft into the manuscript format, place the generated LaTeX
table and PDF figures, and align the text with the introduction and related
work. Future technical work can then be organized around two separate tracks:

- pure-ALOHA robustness across more densities, channel counts, and physical
  regimes, starting from the implemented `k1000_minimal` and `k5000_minimal`
  research matrices;
- non-ALOHA future work, where semi-scheduled refresh motivates coordinated
  access, graph/data-aware clustering, and eventually learning-aware
  comparisons with non-IID FL objectives.
