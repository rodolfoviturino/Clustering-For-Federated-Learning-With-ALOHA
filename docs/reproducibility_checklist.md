# Reproducibility Checklist

This checklist defines the current reproducible paper package for the
ALOHA-focused K=3000 study. It separates simulation execution, comparison,
paper artifacts, tests, and claim boundaries.

## Scope

Canonical matrix:

```bash
python -m experiments.run_research_matrix --matrix k3000_core --compare-only
```

Primary output directory:

```text
Runs/comparison_k3000_core
```

Main claim boundary:

- Main contribution: pure multichannel ALOHA probability shaping for
  member-level D2D freshness in HFL.
- Upper bound only: `semi_scheduled_member_refresh`.
- Out of scope for the current paper package: NOMA, IRSA, SIC, MPR, TDMA/OFDMA,
  OTA-FL, centralized scheduling, GNN/GCN clustering, non-IID learning,
  FedAvg/FedProx/SCAFFOLD comparisons, and data-aware clustering.

## Environment

Use the same Python environment used for the recent validation runs. On this
machine, the working environment has been:

```powershell
C:\Users\Rodol\anaconda3\envs\research\python.exe
```

Portable command form for an activated environment:

```powershell
python -m unittest discover -v
```

Explicit local command form:

```powershell
& 'C:\Users\Rodol\anaconda3\envs\research\python.exe' -m unittest discover -v
```

Expected dependencies include JAX, NumPy, pandas, matplotlib, and the packages
listed in `requirements.txt` or `requirements-colab-gpu.txt`, depending on CPU
or GPU execution.

## Canonical Runs

The matrix `k3000_core` contains these runs:

| Run | Role |
|---|---|
| `member_quota_k3000_w015_floor000_physical_r100` | main pure-ALOHA baseline |
| `member_collision_quota_k3000_w015_t002_g2_min050_physical_r100` | pure-ALOHA convergence/energy ablation |
| `member_capped_quota_k3000_w015_cap010_physical_r100` | pure-ALOHA convergence/energy ablation |
| `member_safe_split_k3000_s8_min2_b005_w015_physical_r100` | negative structural ablation |
| `member_pressure_split_k3000_s8_min2_b005_mw100_ch050_w015_physical_r100` | structural convergence ablation |
| `member_split_k3000_s8_w015_physical_r100` | negative structural ablation |
| `member_split_k3000_s5_w015_physical_r100` | negative structural ablation |
| `member_semischedule_k3000_s030_cc0010_physical_r100` | coordinated upper-bound / future work |

Each run should contain:

```text
Runs/<run_name>/results.csv
Runs/<run_name>/results.metadata.json
```

If a run is missing and you intentionally want to execute it locally or on a
GPU environment:

```powershell
python -m experiments.run_research_matrix --matrix k3000_core --execute-missing
```

Default mode is compare-only to avoid accidentally launching long K=3000
simulations.

## Regenerate Comparison Artifacts

After all run folders exist:

```powershell
python -m experiments.run_research_matrix --matrix k3000_core --compare-only
```

Expected generated files:

```text
Runs/comparison_k3000_core/run_comparison_summary.csv
Runs/comparison_k3000_core/run_comparison_summary.md
Runs/comparison_k3000_core/paper_claim_summary.md
```

The comparison CSV should include `final_*_ci95` columns when the source run
CSVs contain CI95 values.

## Regenerate Paper Tables

```powershell
python -m experiments.export_paper_tables --comparison-csv Runs/comparison_k3000_core/run_comparison_summary.csv
```

Expected generated files:

```text
Runs/comparison_k3000_core/paper_results_table.md
Runs/comparison_k3000_core/paper_results_table.tex
Runs/comparison_k3000_core/paper_claim_bullets.md
```

Checklist:

- `paper_results_table.md` uses `mean +/- ci95` when available.
- Deltas are relative to
  `member_quota_k3000_w015_floor000_physical_r100`.
- Semi-scheduled refresh is labeled as an upper-bound, not a pure-ALOHA
  contribution.
- No table text claims that capped or collision-aware quota dominates all
  member-freshness metrics.

## Regenerate Paper Figures

```powershell
python -m experiments.plot_research_matrix --comparison-csv Runs/comparison_k3000_core/run_comparison_summary.csv
```

Expected generated files:

```text
Runs/comparison_k3000_core/research_matrix_member_stale75.png
Runs/comparison_k3000_core/research_matrix_member_stale75.pdf
Runs/comparison_k3000_core/research_matrix_member_zero_participation.png
Runs/comparison_k3000_core/research_matrix_member_zero_participation.pdf
Runs/comparison_k3000_core/research_matrix_energy_efficiency.png
Runs/comparison_k3000_core/research_matrix_energy_efficiency.pdf
Runs/comparison_k3000_core/research_matrix_t_to_1e12.png
Runs/comparison_k3000_core/research_matrix_t_to_1e12.pdf
Runs/comparison_k3000_core/research_matrix_stale75_energy_pareto.png
Runs/comparison_k3000_core/research_matrix_stale75_energy_pareto.pdf
```

Checklist:

- Bar plots show CI95 error bars when available.
- The Pareto plot uses member stale75 on the x-axis and energy efficiency on
  the y-axis.
- Colors separate baseline, pure-ALOHA ablations, structural/negative
  ablations, and upper-bound.
- Figure labels remain readable after regeneration.

## Tests

Focused paper-artifact tests:

```powershell
python -m unittest tests.test_compare_runs tests.test_research_matrix tests.test_export_paper_tables tests.test_plot_research_matrix
```

Broader paper/plot/sweep validation:

```powershell
python -m unittest tests.test_compare_runs tests.test_research_matrix tests.test_export_paper_tables tests.test_plot_research_matrix tests.test_plot_gpu_sweep tests.test_enhanced_models
```

Full suite:

```powershell
python -m unittest discover -v
```

Recent expected full-suite result:

```text
Ran 130 tests ... OK
```

Matplotlib/pyparsing deprecation warnings may appear during plot tests. They
are warnings from dependencies, not project test failures.

## Paper Text Sources

Use these files when writing the paper:

```text
docs/paper_manuscript_outline.md
docs/paper_results_section.md
docs/member_level_d2d_freshness_experiments.md
docs/modeling_assumptions.md
docs/current_status_and_future_work.md
```

Recommended writing flow:

1. Regenerate comparison, tables, and figures.
2. Copy or adapt `paper_results_table.tex` into the manuscript.
3. Place the PDF figures generated by `plot_research_matrix`.
4. Use `docs/paper_results_section.md` as the first draft of the Results
   section.
5. Use `docs/paper_manuscript_outline.md` to draft the remaining sections.

## Claim Audit

Before submitting or sharing a draft, check each claim:

- Main baseline is `member_quota_utility`.
- Capped quota is a convergence/energy ablation, not a freshness winner.
- Collision-aware quota is a convergence/energy ablation with slight
  zero-participation regression.
- Pressure-guided split is a structural convergence ablation, not a freshness
  improvement.
- Global and size-only splits are negative ablations.
- Semi-scheduled refresh is a coordinated upper-bound/future-work result.
- Results are stated for the K=3000 physical/Rayleigh setup.
- Member-level freshness is distinguished from cluster/aggregate AoI.

## Minimal Robustness Extension

Minimal robustness matrices are available in `experiments.run_research_matrix`.
They should be used before adding another strategy if the paper needs evidence
outside the canonical K=3000 setting.

Available matrices:

```text
k1000_minimal
k5000_minimal
k1000_k5000_minimal
```

The combined matrix is convenient for launching missing runs:

```powershell
python -m experiments.run_research_matrix --matrix k1000_k5000_minimal --execute-missing
```

For analysis, compare/export each density separately so deltas are relative to
the baseline with the same device count:

```powershell
python -m experiments.run_research_matrix --matrix k1000_minimal --compare-only
python -m experiments.export_paper_tables --comparison-csv Runs/comparison_k1000_minimal/run_comparison_summary.csv
python -m experiments.plot_research_matrix --comparison-csv Runs/comparison_k1000_minimal/run_comparison_summary.csv

python -m experiments.run_research_matrix --matrix k5000_minimal --compare-only
python -m experiments.export_paper_tables --comparison-csv Runs/comparison_k5000_minimal/run_comparison_summary.csv
python -m experiments.plot_research_matrix --comparison-csv Runs/comparison_k5000_minimal/run_comparison_summary.csv
```

Each density uses four roles:

- member-quota pure-ALOHA baseline;
- capped-quota pure-ALOHA convergence/energy ablation;
- global split negative structural ablation;
- semi-scheduled coordinated upper-bound.

Observed outcome after the first K=1000/K=5000 run:

- K=1000: capped quota improves energy efficiency but worsens member AoI,
  stale75, and zero participation, so it remains an energy/convergence
  ablation.
- K=5000: capped quota improves energy efficiency and slightly improves member
  freshness, but the freshness gains are small relative to the semi-scheduled
  upper-bound gap.
- Global max-size split remains negative at both densities, especially at
  K=5000.
- Semi-scheduled refresh remains a strong coordinated upper-bound at both
  densities.

The goal is external validity of the current story, not a new contribution.
