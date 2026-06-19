# Project Architecture

The repository is organized around a `.py`-first workflow. Notebooks are useful
for analysis and inspection, but they are not required to generate research
outputs.

## Entry Points

`main.py`

- Human-friendly root command.
- Delegates to `experiments.run_gpu_sweep`.
- Use this when you want the default experiment pipeline from the repository
  root.

`experiments/run_gpu_sweep.py`

- Main experiment runner.
- Generates devices, clusters them, simulates HFL/ALOHA, writes CSV/metadata,
  and creates figures.
- Creates `Runs/<timestamp>/` by default after the simulation succeeds.

`experiments/plot_gpu_sweep.py`

- Regenerates figures from an existing sweep CSV.
- Does not rerun the simulation.

## Core Modules

`Clustering/jax_clustering_algorithm.py`

- JAX device generation and GPU-oriented one-hop D2D clustering.
- Produces fixed-shape padded cluster arrays.

`Models/jax_models_arrangement.py`

- JAX HFL/ALOHA simulation.
- Uses `jax.lax.scan` to run one trajectory and record all requested
  checkpoints.

`Models/models_arrangement.py`

- Compatibility facade for older imports.
- Re-exports the JAX model API.

## Notebooks

`notebooks/explore_results.ipynb`

- Optional analysis notebook.
- Should read CSV, metadata, and figures from `Runs/`.
- Should not be required to execute the simulation.

`notebooks/proposed_clustering_step_by_step.ipynb`

- Optional algorithm-inspection notebook.
- Useful for explaining or visualizing the clustering flow.

## Recommended Workflow

Run an experiment:

```bash
python main.py --run-name local_smoke --devices 100 --rounds 5 --iterations 20 --checkpoints 5 10 20
```

Regenerate figures:

```bash
python -m experiments.plot_gpu_sweep Runs/local_smoke/results.csv
```

Explore results:

```text
Open notebooks/explore_results.ipynb and load files from Runs/<run-name>/.
```

