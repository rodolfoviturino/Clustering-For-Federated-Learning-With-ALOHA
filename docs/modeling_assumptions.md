# Modeling Assumptions

This document records the simulation defaults after the code cleanup.

## Thesis-Compatible Defaults

- Device placement follows the thesis:
  - `r ~ U(1, R_BS)`;
  - `theta ~ U(0, 2*pi)`.
- D2D-SRC clusters are one-hop from every member to the cluster head.
- Cluster size is capped by `Cmax`.
- First-tier HFL aggregation at the CH is a sum of member updates.
- The thesis figure code applies the BS update as an unscaled SGD step:
  `w <- w - u1 * gradient`.
- D2D member availability is ideal by default:
  - `d2d_member_compute_probability=1.0`;
  - `d2d_member_link_success_probability=1.0`.
- Optimized ALOHA uses the aggregate norm of the CH update.

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
- CH rotation and energy-aware CH selection are not implemented in this cleanup.
