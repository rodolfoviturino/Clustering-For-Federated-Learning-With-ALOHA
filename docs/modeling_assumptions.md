# Modeling Assumptions

This document records the simulation defaults after the code cleanup.

## Thesis-Compatible Defaults

- Device placement follows the thesis:
  - `r ~ U(1, R_BS)`;
  - `theta ~ U(0, 2*pi)`.
- D2D-SRC clusters are one-hop from every member to the cluster head.
- Cluster size is capped by `Cmax`.
- Dense JAX geometric clustering forms local pairs first by default. This is a
  local D2D request/accept stage, not a BS-side global assignment.
- Dense JAX clustering may run local singleton join-repair passes. A singleton
  can join only a directly reachable CH with spare capacity; this models local
  D2D control messages, not BS-side global orchestration.
- Dense JAX clustering may run local CH-rotation repair for two-device clusters.
  A member becomes CH only when it directly reaches both the old CH and the
  singleton being admitted.
- Dense JAX clustering may run local CH-to-CH merge repair after singleton and
  rotation repair. A source D2D cluster can merge into a target D2D cluster
  only when the target CH can directly cover every source member and the union
  still fits within `Cmax`.
- First-tier HFL aggregation at the CH is a sum of member updates.
- The thesis figure code applies the BS update as an unscaled SGD step:
  `w <- w - u1 * gradient`.
- D2D member availability is ideal by default:
  - `d2d_member_compute_probability=1.0`;
  - `d2d_member_link_success_probability=1.0`.
- Optimized ALOHA uses the aggregate norm of the CH update.
- Optimized ALOHA can optionally use a fixed-access floor. This keeps the
  norm-based controller from starving the channel late in training, especially
  in D2D runs where aggregate norms can shrink quickly.
- Optimized D2D can optionally use a utility policy that combines aggregate
  norm, active aggregate size, and freshness while keeping the expected ALOHA
  load near the channel count.

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
- `--optimized-d2d-access-floor-fraction 1.0` is an enhanced optimized-ALOHA
  ablation. It preserves norm-based priority while ensuring optimized D2D uses
  at least the fixed-D2D access probability.
- `--optimized-d2d-access-mode utility` is a stronger optimized-D2D ablation.
  It is not thesis-exact; it tests whether utility-aware probability allocation
  can outperform fixed D2D without increasing total expected channel load.
- Energy-aware CH selection is not implemented in this cleanup.
