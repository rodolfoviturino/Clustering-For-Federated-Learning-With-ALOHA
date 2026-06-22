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

    def test_gpu_sweep_records_utility_load_target_factor(self):
        rows, metadata = run_gpu_sweep(
            self._small_args(
                devices=10,
                rounds=1,
                iterations=2,
                checkpoints=[1, 2],
                optimized_d2d_access_mode="utility",
                optimized_d2d_load_target_factor=1.2,
                cluster_head_selection_mode="quality",
                cluster_head_degree_weight=0.5,
                cluster_head_channel_weight=0.3,
                cluster_head_battery_weight=0.2,
                d2d_ch_bs_success_mode="channel_quality",
                d2d_ch_bs_min_success_probability=0.3,
                d2d_ch_bs_pathloss_exponent=2.5,
                d2d_ch_bs_battery_exponent=0.25,
                optimized_d2d_load_allocation_mode="proportional_clip",
                optimized_d2d_redistribution_fraction=0.0,
                optimized_d2d_redistribution_trigger_ratio=0.90,
                optimized_d2d_density_trigger_threshold=0.93,
                optimized_d2d_dense_trigger_ratio=0.85,
                optimized_d2d_throughput_ewma_decay=0.80,
            )
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(metadata["optimized_d2d_access_mode"], "utility")
        self.assertEqual(metadata["optimized_d2d_load_target_factor"], 1.2)
        self.assertEqual(metadata["cluster_head_selection_mode"], "quality")
        self.assertEqual(metadata["cluster_head_degree_weight"], 0.5)
        self.assertEqual(metadata["cluster_head_channel_weight"], 0.3)
        self.assertEqual(metadata["cluster_head_battery_weight"], 0.2)
        self.assertEqual(metadata["d2d_ch_bs_success_mode"], "channel_quality")
        self.assertEqual(metadata["d2d_ch_bs_min_success_probability"], 0.3)
        self.assertEqual(metadata["d2d_ch_bs_pathloss_exponent"], 2.5)
        self.assertEqual(metadata["d2d_ch_bs_battery_exponent"], 0.25)
        self.assertEqual(
            metadata["optimized_d2d_load_allocation_mode"],
            "proportional_clip",
        )
        self.assertEqual(metadata["optimized_d2d_redistribution_fraction"], 0.0)
        self.assertEqual(metadata["optimized_d2d_redistribution_trigger_ratio"], 0.90)
        self.assertEqual(metadata["optimized_d2d_density_trigger_threshold"], 0.93)
        self.assertEqual(metadata["optimized_d2d_dense_trigger_ratio"], 0.85)
        self.assertEqual(metadata["optimized_d2d_throughput_ewma_decay"], 0.80)

    def test_gpu_sweep_records_adaptive_diversity_parameters(self):
        rows, metadata = run_gpu_sweep(
            self._small_args(
                devices=10,
                rounds=1,
                iterations=2,
                checkpoints=[1, 2],
                optimized_d2d_access_mode="adaptive_diversity",
                optimized_d2d_access_floor_fraction=0.02,
                optimized_d2d_norm_exponent=3.5,
                optimized_d2d_cluster_size_exponent=1.5,
                optimized_d2d_freshness_exponent=0.25,
                optimized_d2d_late_norm_exponent=1.25,
                optimized_d2d_late_freshness_exponent=1.0,
                optimized_d2d_novelty_exponent=1.5,
                optimized_d2d_novelty_floor=0.25,
                optimized_d2d_reference_decay=0.90,
                optimized_d2d_load_target_factor=1.1,
                optimized_d2d_load_allocation_mode="selective_water_filling",
                optimized_d2d_redistribution_fraction=0.5,
                optimized_d2d_redistribution_trigger_ratio=0.95,
                optimized_d2d_density_trigger_threshold=0.95,
                optimized_d2d_dense_trigger_ratio=0.90,
                optimized_d2d_throughput_ewma_decay=0.90,
                optimized_d2d_adaptive_switch_fraction=0.30,
                optimized_d2d_adaptive_switch_gain=12.0,
            )
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(metadata["optimized_d2d_access_mode"], "adaptive_diversity")
        self.assertEqual(metadata["optimized_d2d_late_norm_exponent"], 1.25)
        self.assertEqual(metadata["optimized_d2d_late_freshness_exponent"], 1.0)
        self.assertEqual(metadata["optimized_d2d_adaptive_switch_fraction"], 0.30)
        self.assertEqual(metadata["optimized_d2d_adaptive_switch_gain"], 12.0)
        self.assertEqual(metadata["optimized_d2d_load_target_factor"], 1.1)
        self.assertEqual(
            metadata["optimized_d2d_load_allocation_mode"],
            "selective_water_filling",
        )
        self.assertEqual(metadata["optimized_d2d_redistribution_fraction"], 0.5)
        self.assertEqual(metadata["optimized_d2d_redistribution_trigger_ratio"], 0.95)
        self.assertEqual(metadata["optimized_d2d_density_trigger_threshold"], 0.95)
        self.assertEqual(metadata["optimized_d2d_dense_trigger_ratio"], 0.90)
        self.assertEqual(metadata["optimized_d2d_throughput_ewma_decay"], 0.90)


if __name__ == "__main__":
    unittest.main()
