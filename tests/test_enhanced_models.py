import math
import unittest

from experiments import run_gpu_sweep as gpu_sweep
from experiments.run_ablation import run_ablation
from experiments.run_gpu_sweep import namespace_from_defaults, run_gpu_sweep


JAX_AVAILABLE = gpu_sweep.jax is not None


@unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed in this interpreter")
class GpuSweepTests(unittest.TestCase):
    def _small_args(self, **overrides):
        defaults = dict(
            devices=12,
            bs_radius=80.0,
            device_radius=15.0,
            max_cluster_size=6,
            min_cluster_size=1,
            rounds=2,
            iterations=3,
            checkpoints=[1, 3],
            data_dimension=2,
            channels=2,
            pcomp=0.5,
            learning_rate=0.01,
            step_size=0.1,
            seed=202406,
            clustering_mode="geometric",
            tile_size=16,
            d2d_member_compute_probability=1.0,
            d2d_member_link_success_probability=1.0,
            output=None,
        )
        defaults.update(overrides)
        return namespace_from_defaults(**defaults)

    def test_gpu_sweep_rows_and_metadata_are_finite(self):
        rows, metadata = run_gpu_sweep(self._small_args())

        self.assertEqual(metadata["backend"], "jax")
        self.assertEqual(metadata["devices"], 12)
        self.assertEqual(metadata["rounds"], 2)
        self.assertEqual(len(rows), 2)

        for row in rows:
            self.assertIn("fixed_aloha_d2d_error_norm_mean", row)
            for value in row.values():
                if isinstance(value, str):
                    continue
                self.assertTrue(math.isfinite(float(value)))

    def test_ablation_wrapper_returns_rows_only(self):
        rows = run_ablation(self._small_args(clustering_mode="no_d2d"))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["clustering_mode"], "no_d2d")

    def test_gpu_sweep_supports_float64_precision(self):
        rows, metadata = run_gpu_sweep(
            self._small_args(
                devices=10,
                rounds=1,
                iterations=2,
                checkpoints=[1, 2],
                precision="float64",
            )
        )

        self.assertEqual(metadata["precision"], "float64")
        self.assertTrue(metadata["jax_enable_x64"])
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
