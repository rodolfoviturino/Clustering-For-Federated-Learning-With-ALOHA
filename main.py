"""Root command-line entry point for the experiment pipeline.

This file exists so the repository has an obvious executable starting point:

    python main.py --run-name local_smoke --devices 100 --rounds 5

The implementation stays in ``experiments.run_gpu_sweep`` because that module is
also importable by tests, notebooks, and Colab scripts.
"""

from experiments.run_gpu_sweep import main


if __name__ == "__main__":
    main()
