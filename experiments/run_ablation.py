"""Compatibility entry point for the JAX GPU sweep runner.

This module routes ablation-style calls to ``experiments.run_gpu_sweep`` so old
notebook imports can migrate gradually without keeping two simulation entry
points.
"""

from experiments.run_gpu_sweep import main, run_gpu_sweep


def run_ablation(args):
    """Run the JAX sweep and return aggregate rows.

    ``run_gpu_sweep`` returns ``(rows, metadata)``.  The historical
    ``run_ablation`` API returned only rows, so this wrapper keeps that shape.
    """
    rows, _ = run_gpu_sweep(args)
    return rows


if __name__ == "__main__":
    main()
