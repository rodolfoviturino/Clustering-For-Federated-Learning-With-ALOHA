# GPU JAX Backend

This document describes the JAX-only accelerated backend used for large
experiments. The older CPU clustering code remains useful as a readable
description of the original D2D-SRC idea, but it is no longer the runtime
backend for new sweeps.

## Why JAX

JAX is used because the experiment has three performance requirements that fit
JAX well:

1. Many independent Monte Carlo rounds can be batched with `jax.vmap`.
2. The FL time loop can run as one compiled loop with `jax.lax.scan`.
3. Fixed-shape arrays can stay on GPU across device generation, clustering, and
   model simulation.

The official installation guide lists `pip install -U "jax[cuda13]"` for
NVIDIA CUDA 13, `jax[cuda12]` as the CUDA 12 alternative, and `jax[tpu]` for
Google Cloud TPU VMs. It also notes that NVIDIA GPU wheels are available for
Linux, while native Windows has CPU support but no NVIDIA GPU support:
https://docs.jax.dev/en/latest/installation.html

For that reason this repository uses two requirement files:

- `requirements.txt`: local development, Windows CPU, and small smoke tests.
- `requirements-colab-gpu.txt`: Colab/Linux GPU runs with the CUDA 13 extra.

TPU execution is not the first target. The simulation has random access,
branching, and sparse cluster membership, which are easier to tune first on GPU.
Once the GPU path is statistically validated, TPU experiments can reuse the same
fixed-shape arrays with additional profiling.

## Main Modules

`main.py`

- Root command-line entry point for the full `.py` workflow.
- Delegates to `experiments.run_gpu_sweep`.
- Use it when running experiments from the repository root.

`Clustering/jax_clustering_algorithm.py`

- Generates device arrays with `devices_generator_jax`.
- Builds padded one-hop cluster arrays with `clusterizer_jax`.
- Provides validation helpers for one-hop coverage, `Cmax`, singleton count, and
  overflow diagnostics.

`Models/jax_models_arrangement.py`

- Runs the HFL/ALOHA simulation with `error_calculator_trace_jax`.
- Uses `jax.lax.scan` to advance one trajectory from `t = 1` to `max_t`.
- Returns metrics at all requested checkpoints without rerunning earlier
  iterations.

`experiments/run_gpu_sweep.py`

- Creates one compiled batched program with `jax.jit(jax.vmap(...))`.
- Aggregates mean values and 95 percent confidence interval half-widths.
- Creates a fresh `Runs/YYYY-MM-DD-HH-MM-SS/` folder after the simulation
  completes successfully, unless `--output` is used for a direct CSV path.
- Writes run metadata, including backend, JAX version, device type, seed policy,
  dimensions, clustering mode, output paths, and elapsed time.
- Generates standard figures from the output CSV unless `--no-plots` is passed.

`experiments/plot_gpu_sweep.py`

- Converts a saved sweep CSV into figures without rerunning the simulation.
- Uses a non-interactive Matplotlib backend, so it works in terminals, CI,
  Windows shells, and Colab.
- Produces error norm, device upload, cluster-head upload, and clustering-rate
  plots.

## Device Arrays

`devices_generator_jax(K, R_BS, seed)` returns a `JaxDeviceBatch`.

Array contracts:

- `device_ids`: `int32[K]`. Stable IDs, normally `0..K-1`.
- `coords`: `float32/float64[K, 2]`. Device `(x, y)` coordinates in meters.
- `device_angle`: `float32/float64[K]`. Polar angle in radians.
- `distance_to_bs`: `float32/float64[K]`. Distance from each device to the BS in meters.
- `battery`: `int32[K]`. Battery percentage in `[1, 100]`.
- `stats_product`: `float32/float64[K]`. Product `distance_to_bs * battery`,
  kept for compatibility with the thesis-style descriptive statistics.
- `bs_radius`: scalar `float32/float64`. BS coverage radius in meters.

The default radius model is thesis-compatible:

```text
r ~ U(1, R_BS)
theta ~ U(0, 2*pi)
x = r cos(theta)
y = r sin(theta)
```

Set `uniform_area=True` only for an explicit disk-uniform ablation:

```text
r = sqrt(U(1, R_BS^2))
```

## Cluster Arrays

`clusterizer_jax(...)` returns a `JaxClusterResult`.

Array contracts:

- `cluster_members`: `int32[K, Cmax]`. Padded cluster rows. Unused entries are
  `-1`.
- `cluster_sizes`: `int32[K]`. Number of valid devices in each row.
- `cluster_heads`: `int32[K]`. The selected CH for each row, or `-1` for empty
  rows.
- `cluster_mask`: `bool[K]`. True only for active cluster rows.
- `number_of_clusters`: scalar `int32`. Count of active cluster rows.
- `clusterized_devices_rate`: scalar `float32/float64`. Percentage of devices
  in non-singleton clusters.
- `singleton_count`: scalar `int32`. Number of one-device clusters.
- `overflow_count`: scalar `int32`. Number of rows split because a grid cell had
  more than `Cmax` devices.
- `mode_code` and `strategy_code`: scalar `int32` metadata for saved runs.

Column 0 of every active row is the cluster head. For example, if one row is
`[8, 21, 32, -1, -1]`, device 8 is the CH and devices 21 and 32 are one-hop
members.

## GPU-Native Clustering

The default production strategy is `strategy="dense"` because it is closest to
the original thesis experiment at `K=1000`.

Dense strategy:

- ranks candidate CHs by the selected clustering mode;
- in `geometric` mode, ranks CHs by one-hop degree and then device ID, using
  only the D2D radius graph;
- forms initial local pairs by default (`--initial-cluster-size 2`), matching
  the D2D-SRC pair-formation stage;
- then lets CHs absorb reachable singletons through local repair, up to `Cmax`;
- runs local singleton repair passes after the greedy phase: a singleton can
  join only a CH it reaches directly and only if that CH has spare capacity;
- runs local CH-rotation repair for two-device clusters: if a singleton reaches
  the member but not the current CH, the member may become CH and admit it;
- reruns singleton join repair after CH rotations so newly promoted CHs can
  accept additional reachable singletons;
- runs local CH-to-CH merge repair after singleton/rotation repair: two
  non-singleton clusters may merge only when the target CH can cover the whole
  union and the merged cluster remains within `Cmax`;
- preserves one-hop CH coverage directly from the radius test;
- is more expensive than grid clustering, but gives a clustering-rate behavior
  much closer to the old D2D-SRC notebook.

The scalable large-`K` option is `strategy="grid"`.

The grid side length is:

```text
cell_side = R_D2D / sqrt(2)
```

Any two points inside the same square cell are at most:

```text
sqrt(cell_side^2 + cell_side^2) = R_D2D
```

This gives a direct one-hop guarantee from the CH to every member when the CH is
chosen from inside the cell. The method avoids a full `K x K` neighbor matrix,
which would be the main memory bottleneck for very large `K`.

Dense cells are split into balanced chunks, so a cell with 12 devices and
`Cmax = 10` becomes `6 + 6` rather than `10 + 2`. If a whole cell has size in
`2..Cmin-1`, it falls back to singleton rows. That conservative fallback keeps
the clustered-device rate from counting weak tiny groups as successful D2D
clusters.

Modes:

- `geometric`: ranks devices by one-hop degree, with device ID as the
  deterministic tie-breaker. This keeps the mode geometry-only while usually
  reducing singleton clusters compared with arbitrary ID ordering.
- `utility`: ranks devices by degree, battery, normalized BS channel quality,
  and ID. Degree is computed with tiled distance comparisons to avoid materializing
  a full all-pairs tensor.
- `no_d2d`: returns singleton clusters for the baseline comparison.

`neighbor_counts_tiled_jax(coords, R_D2D, tile_size)` is available for profiling
and utility scoring. It compares all devices against fixed-size candidate tiles
on the accelerator.

The dense repair pass is intentionally distributed-style.  It is equivalent to
nearby singletons broadcasting join requests and reachable CHs with spare
capacity accepting requests. It does not require the BS to solve a global
assignment problem.

The pair-rotation repair is also local: it is limited to two-device clusters
where the promoted member can directly reach both the old CH and the singleton.
No global reassignment is used.

The CH-to-CH merge repair is a cluster-quality pass, not a clustering-rate
shortcut. It does not turn singletons into clustered devices by itself. Instead,
it consolidates nearby valid D2D clusters when one existing CH can cover the
combined member set. The practical interpretation is that neighboring CHs
exchange compact cluster summaries and accept a merge only when the target CH
can serve every member in one hop. This reduces the number of CH rows competing
on the BS uplink and increases the average aggregate size seen by HFL.

Use `--initial-cluster-size Cmax` to recover the earlier greedy-fill behavior.
The default `2` prioritizes covering more devices with at least one D2D partner
before growing clusters, which is closer to the original D2D-SRC sequence.

SciPy `cKDTree` is not used in the GPU backend. It is CPU-side and remains a
useful validation/profiling reference only:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

## FL And ALOHA Trace

`error_calculator_trace_jax(...)` simulates a synthetic linear-regression FL
task.

Data model:

```text
X_i in R^L
w_true in R^L
y_i = X_i dot w_true
```

At each round, every device has a local gradient:

```text
grad_i(w) = X_i * (X_i dot w - y_i)
```

The BS update uses the thesis figure's unscaled SGD step by default:

```text
w <- w - u1 * sum(successful_gradients)
```

Use `--normalize-by-k` or `normalize_by_k=True` for the conservative normalized
ablation:

```text
w <- w - u1 * sum(successful_gradients) / K
```

Use `--precision float64` for thesis reproduction when optimized ALOHA reaches
very small error norms. `float32` is the speed default for GPU sweeps, but it
usually floors norms around single-precision accuracy.

Optimized ALOHA can optionally apply an access floor:

```text
p_opt_guarded = max(p_opt_norm_based, floor_fraction * p_fixed)
```

For D2D, `p_fixed` is the fixed D2D ALOHA access probability
`min(M / number_of_clusterheads, pcomp)`. The recommended stress test for the
merge-enhanced clustering path is `--optimized-d2d-access-floor-fraction 1.0`.
This prevents optimized D2D from starving the BS channels after aggregate norms
become very small, while still letting CHs with larger aggregate norms use
higher access probabilities. The thesis-exact behavior is preserved by the
default floor fraction `0.0`.

The enhanced optimized-D2D utility mode is selected with
`--optimized-d2d-access-mode utility`. It uses:

```text
utility_h =
  norm_h^beta * active_cluster_size_h^delta * freshness_h^gamma
```

and then allocates access probabilities so the expected number of CH contenders
stays close to the channel count `M`. The access floor reserves a fraction of
the fixed-D2D load before the remaining probability mass is distributed by
utility. For this reason, a floor of `1.0` intentionally collapses the utility
mode toward fixed D2D; useful exploratory values are usually `0.25` and `0.5`.

Six scenario columns are returned:

0. Polling without D2D.
1. Fixed ALOHA without D2D.
2. Optimized ALOHA without D2D.
3. Polling with D2D.
4. Fixed ALOHA with D2D.
5. Optimized ALOHA with D2D.

The returned `JaxTraceResult` contains:

- `clusterized_devices_rate`: scalar clustering percentage.
- cluster quality scalars: CH row count, singleton count, non-singleton D2D
  cluster count, clustered-device count, mean cluster size, and mean
  non-singleton cluster size.
- `error_norms`: `float[checkpoints, 6]`.
- `successful_uploads`: `float[checkpoints, 6]`.
- `successful_clusterhead_uploads`: `float[checkpoints, 3]`.
- `checkpoints`: `int32[checkpoints]`.

## Seed Policy

The sweep runner uses:

```text
round_seed = seed_start + round_index
```

Each round uses its seed for device generation, clustering choices, synthetic FL
data, local compute events, channel selection, and member-to-CH success events.
Exact per-seed equality with the older CPU notebook is not expected. The
scientific requirement is statistical consistency under a documented seed policy.

## Colab GPU Smoke Test

In a Colab GPU runtime:

```bash
pip install -r requirements-colab-gpu.txt
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

If the runtime reports CUDA 12 compatibility instead of CUDA 13, use:

```bash
pip install -U "jax[cuda12]==0.10.2"
```

Then run a small sweep:

```bash
python -m experiments.run_gpu_sweep \
  --devices 1000 \
  --rounds 10 \
  --iterations 20 \
  --checkpoints 5 10 20 \
  --clustering-mode geometric \
  --clustering-strategy dense \
  --run-name gpu_smoke
```

The equivalent root command is:

```bash
python main.py \
  --devices 1000 \
  --rounds 10 \
  --iterations 20 \
  --checkpoints 5 10 20 \
  --clustering-mode geometric \
  --clustering-strategy dense \
  --run-name gpu_smoke
```

That command writes:

- `Runs/gpu_smoke/results.csv`;
- `Runs/gpu_smoke/results.metadata.json`;
- `Runs/gpu_smoke/results_error_norm.png` and `.pdf`;
- `Runs/gpu_smoke/results_figure_15_error_norm.png` and `.pdf`;
- `Runs/gpu_smoke/results_uploads.png` and `.pdf`;
- `Runs/gpu_smoke/results_clusterhead_uploads.png` and `.pdf`;
- `Runs/gpu_smoke/results_cluster_rate.png` and `.pdf`;
- `Runs/gpu_smoke/results_cluster_quality.png` and `.pdf`.

The thesis-style figure plots every checkpoint saved in the CSV. Omit
`--checkpoints` to save and plot the full curve for every iteration
`t = 1..max_t`; use `--checkpoints 1 100 200` only when you intentionally want
a lighter start/middle/end figure.

If `Runs/gpu_smoke/` already exists, the runner writes to the next available
suffix, such as `Runs/gpu_smoke-02/`. Without `--run-name`, the folder name is a
local timestamp: `Runs/YYYY-MM-DD-HH-MM-SS/`.

Use `--no-plots` when measuring raw simulation speed and plotting time should
not be included in the end-to-end command.

Scale gradually:

1. Increase `rounds` until the GPU stays busy.
2. Increase `K` while watching memory.
3. Compare `geometric`, `utility`, and `no_d2d`; use `dense` first for thesis
   reproduction and `grid` when scaling beyond dense memory/runtime limits.
4. Save both the CSV and the `.metadata.json` file with every result.

## Statistical Validation

The JAX path is a simulation backend, not a byte-exact oracle. Validate research
results by comparing distributions and curves:

- mean error norm over `t`;
- 95 percent confidence interval half-widths;
- total successful upload counts;
- successful cluster-head upload counts;
- clustering-rate distribution;
- energy/freshness metrics if those modes are added back to the JAX model.

Small cases should also run structural checks:

- every device appears in exactly one active row;
- every non-singleton member is within `R_D2D` of its CH;
- no active row exceeds `Cmax`;
- singleton accounting matches the padded arrays;
- overflow diagnostics are recorded when a grid cell is split.

## Metadata To Preserve

Every thesis-scale result should preserve:

- backend name and version;
- `jax.devices()` output;
- Python version;
- seed policy;
- `K`, `M`, `L`, `Cmax`, `R_BS`, `R_D2D`;
- clustering mode and strategy;
- local repair pass count;
- local CH-rotation repair pass count;
- local CH-to-CH merge repair pass count;
- initial dense cluster size;
- number of rounds and checkpoints;
- batch size or effective `vmap` size;
- elapsed time including compile time;
- output CSV path and metadata JSON path.
- generated figure paths and formats.
