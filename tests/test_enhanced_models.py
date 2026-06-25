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
                cluster_head_channel_score_mode="rayleigh_outage",
                d2d_member_link_success_mode="rayleigh_outage",
                d2d_member_pathloss_exponent=2.3,
                d2d_member_reference_snr=34567.0,
                d2d_member_snr_threshold=0.6,
                d2d_ch_bs_success_mode="rayleigh_outage",
                d2d_ch_bs_min_success_probability=0.3,
                d2d_ch_bs_pathloss_exponent=2.5,
                d2d_ch_bs_battery_exponent=0.25,
                d2d_ch_bs_reference_snr=12345.0,
                d2d_ch_bs_snr_threshold=0.8,
                device_bs_success_mode="rayleigh_outage",
                device_bs_min_success_probability=0.4,
                device_bs_pathloss_exponent=2.2,
                device_bs_battery_exponent=0.10,
                device_bs_reference_snr=23456.0,
                device_bs_snr_threshold=0.7,
                energy_drain_mode="dynamic",
                energy_model="first_order_radio",
                battery_feasibility_mode="required_energy",
                energy_direct_bs_cost=0.01,
                energy_d2d_member_cost=0.005,
                energy_ch_bs_cost=0.02,
                energy_electronics_cost=0.0003,
                energy_bs_amplifier_cost=3e-8,
                energy_d2d_amplifier_cost=2e-6,
                energy_bs_pathloss_exponent=2.1,
                energy_d2d_pathloss_exponent=2.2,
                energy_aggregation_cost=0.00003,
                energy_update_size=1.2,
                energy_aggregate_size=0.8,
                energy_rotation_control_cost=0.00001,
                d2d_ch_rotation_mode="energy_aware",
                d2d_ch_rotation_interval=2,
                d2d_ch_rotation_trigger_mode="interval_or_aoi_or_member_aoi",
                d2d_ch_rotation_aoi_threshold_fraction=0.80,
                d2d_ch_rotation_member_threshold_fraction=0.65,
                d2d_ch_rotation_member_link_weight=0.40,
                d2d_energy_efficiency_level="eco",
                cluster_split_mode="pressure_safe_max_size",
                cluster_split_max_size=3,
                cluster_split_min_subcluster_size=2,
                cluster_split_budget_fraction=0.25,
                cluster_split_pressure_member_weight=1.5,
                cluster_split_pressure_ch_weight=0.25,
                optimized_d2d_load_allocation_mode="proportional_clip",
                optimized_d2d_redistribution_fraction=0.0,
                optimized_d2d_redistribution_trigger_ratio=0.90,
                optimized_d2d_density_trigger_threshold=0.93,
                optimized_d2d_dense_trigger_ratio=0.85,
                optimized_d2d_throughput_ewma_decay=0.80,
                optimized_d2d_aoi_weight=0.75,
                optimized_d2d_aoi_exponent=1.5,
                optimized_d2d_aoi_threshold_fraction=0.70,
                optimized_d2d_aoi_channel_exponent=1.25,
                optimized_d2d_aoi_battery_exponent=0.75,
                optimized_d2d_member_refresh_floor_fraction=0.15,
                optimized_d2d_member_quota_cap_fraction=0.65,
                optimized_d2d_member_deficit_decay=0.82,
                optimized_d2d_member_deficit_weight=0.35,
                optimized_d2d_member_collision_target_fraction=0.04,
                optimized_d2d_member_collision_gain=2.5,
                optimized_d2d_member_collision_min_quota_scale=0.30,
                optimized_d2d_member_schedule_fraction=0.20,
                optimized_d2d_member_schedule_deficit_weight=0.15,
                optimized_d2d_member_schedule_control_cost=0.0007,
            )
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(metadata["optimized_d2d_access_mode"], "utility")
        self.assertEqual(metadata["optimized_d2d_load_target_factor"], 1.2)
        self.assertEqual(metadata["cluster_head_selection_mode"], "quality")
        self.assertEqual(metadata["cluster_head_degree_weight"], 0.5)
        self.assertEqual(metadata["cluster_head_channel_weight"], 0.3)
        self.assertEqual(metadata["cluster_head_battery_weight"], 0.2)
        self.assertEqual(metadata["cluster_head_channel_score_mode"], "rayleigh_outage")
        self.assertEqual(metadata["d2d_member_link_success_mode"], "rayleigh_outage")
        self.assertEqual(metadata["d2d_member_pathloss_exponent"], 2.3)
        self.assertEqual(metadata["d2d_member_reference_snr"], 34567.0)
        self.assertEqual(metadata["d2d_member_snr_threshold"], 0.6)
        self.assertEqual(metadata["d2d_ch_bs_success_mode"], "rayleigh_outage")
        self.assertEqual(metadata["d2d_ch_bs_min_success_probability"], 0.3)
        self.assertEqual(metadata["d2d_ch_bs_pathloss_exponent"], 2.5)
        self.assertEqual(metadata["d2d_ch_bs_battery_exponent"], 0.25)
        self.assertEqual(metadata["d2d_ch_bs_reference_snr"], 12345.0)
        self.assertEqual(metadata["d2d_ch_bs_snr_threshold"], 0.8)
        self.assertEqual(metadata["device_bs_success_mode"], "rayleigh_outage")
        self.assertEqual(metadata["device_bs_min_success_probability"], 0.4)
        self.assertEqual(metadata["device_bs_pathloss_exponent"], 2.2)
        self.assertEqual(metadata["device_bs_battery_exponent"], 0.10)
        self.assertEqual(metadata["device_bs_reference_snr"], 23456.0)
        self.assertEqual(metadata["device_bs_snr_threshold"], 0.7)
        self.assertEqual(metadata["energy_drain_mode"], "dynamic")
        self.assertEqual(metadata["energy_model"], "first_order_radio")
        self.assertEqual(metadata["battery_feasibility_mode"], "required_energy")
        self.assertEqual(metadata["energy_direct_bs_cost"], 0.01)
        self.assertEqual(metadata["energy_d2d_member_cost"], 0.005)
        self.assertEqual(metadata["energy_ch_bs_cost"], 0.02)
        self.assertEqual(metadata["energy_electronics_cost"], 0.0003)
        self.assertEqual(metadata["energy_bs_amplifier_cost"], 3e-8)
        self.assertEqual(metadata["energy_d2d_amplifier_cost"], 2e-6)
        self.assertEqual(metadata["energy_bs_pathloss_exponent"], 2.1)
        self.assertEqual(metadata["energy_d2d_pathloss_exponent"], 2.2)
        self.assertEqual(metadata["energy_aggregation_cost"], 0.00003)
        self.assertEqual(metadata["energy_update_size"], 1.2)
        self.assertEqual(metadata["energy_aggregate_size"], 0.8)
        self.assertEqual(metadata["energy_rotation_control_cost"], 0.00001)
        self.assertEqual(metadata["d2d_ch_rotation_mode"], "energy_aware")
        self.assertEqual(metadata["d2d_ch_rotation_interval"], 2)
        self.assertEqual(
            metadata["d2d_ch_rotation_trigger_mode"],
            "interval_or_aoi_or_member_aoi",
        )
        self.assertEqual(metadata["d2d_ch_rotation_aoi_threshold_fraction"], 0.80)
        self.assertEqual(
            metadata["d2d_ch_rotation_member_threshold_fraction"],
            0.65,
        )
        self.assertEqual(metadata["d2d_ch_rotation_member_link_weight"], 0.40)
        self.assertEqual(metadata["d2d_energy_efficiency_level"], "eco")
        self.assertEqual(metadata["cluster_split_mode"], "pressure_safe_max_size")
        self.assertEqual(metadata["cluster_split_max_size"], 3)
        self.assertEqual(metadata["cluster_split_min_subcluster_size"], 2)
        self.assertEqual(metadata["cluster_split_budget_fraction"], 0.25)
        self.assertEqual(metadata["cluster_split_pressure_member_weight"], 1.5)
        self.assertEqual(metadata["cluster_split_pressure_ch_weight"], 0.25)
        self.assertEqual(
            metadata["d2d_energy_efficiency_profile_weights"],
            {"channel": 0.45, "battery": 0.45, "stability": 0.10},
        )
        self.assertEqual(
            metadata["optimized_d2d_load_allocation_mode"],
            "proportional_clip",
        )
        self.assertEqual(metadata["optimized_d2d_redistribution_fraction"], 0.0)
        self.assertEqual(metadata["optimized_d2d_redistribution_trigger_ratio"], 0.90)
        self.assertEqual(metadata["optimized_d2d_density_trigger_threshold"], 0.93)
        self.assertEqual(metadata["optimized_d2d_dense_trigger_ratio"], 0.85)
        self.assertEqual(metadata["optimized_d2d_throughput_ewma_decay"], 0.80)
        self.assertEqual(metadata["optimized_d2d_aoi_weight"], 0.75)
        self.assertEqual(metadata["optimized_d2d_aoi_exponent"], 1.5)
        self.assertEqual(metadata["optimized_d2d_aoi_threshold_fraction"], 0.70)
        self.assertEqual(metadata["optimized_d2d_aoi_channel_exponent"], 1.25)
        self.assertEqual(metadata["optimized_d2d_aoi_battery_exponent"], 0.75)
        self.assertEqual(
            metadata["optimized_d2d_member_refresh_floor_fraction"],
            0.15,
        )
        self.assertEqual(metadata["optimized_d2d_member_quota_cap_fraction"], 0.65)
        self.assertEqual(metadata["optimized_d2d_member_deficit_decay"], 0.82)
        self.assertEqual(metadata["optimized_d2d_member_deficit_weight"], 0.35)
        self.assertEqual(
            metadata["optimized_d2d_member_collision_target_fraction"],
            0.04,
        )
        self.assertEqual(metadata["optimized_d2d_member_collision_gain"], 2.5)
        self.assertEqual(
            metadata["optimized_d2d_member_collision_min_quota_scale"],
            0.30,
        )
        self.assertEqual(metadata["optimized_d2d_member_schedule_fraction"], 0.20)
        self.assertEqual(
            metadata["optimized_d2d_member_schedule_deficit_weight"],
            0.15,
        )
        self.assertEqual(
            metadata["optimized_d2d_member_schedule_control_cost"],
            0.0007,
        )
        self.assertIn("optimized_aloha_d2d_energy_used_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_energy_efficiency_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_clusterhead_energy_used_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_peak_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_p75_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_p90_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_p95_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_stale_fraction_50_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_stale_fraction_75_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_stale_fraction_100_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_peak_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_p75_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_p90_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_p95_aoi_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_stale_fraction_50_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_stale_fraction_75_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_stale_fraction_100_mean", rows[0])
        self.assertIn("optimized_aloha_d2d_member_participation_p05_mean", rows[0])
        self.assertIn(
            "optimized_aloha_d2d_member_zero_participation_fraction_mean",
            rows[0],
        )
        self.assertIn(
            "optimized_aloha_d2d_member_stale_compute_failure_fraction_mean",
            rows[0],
        )
        self.assertIn(
            "optimized_aloha_d2d_member_stale_link_failure_fraction_mean",
            rows[0],
        )
        self.assertIn(
            "optimized_aloha_d2d_member_stale_member_energy_failure_fraction_mean",
            rows[0],
        )
        self.assertIn(
            "optimized_aloha_d2d_member_stale_ch_no_attempt_fraction_mean",
            rows[0],
        )
        self.assertIn(
            "optimized_aloha_d2d_member_stale_collision_fraction_mean",
            rows[0],
        )
        self.assertIn(
            "optimized_aloha_d2d_member_stale_ch_bs_failure_fraction_mean",
            rows[0],
        )
        self.assertIn(
            "optimized_aloha_d2d_member_stale_other_failure_fraction_mean",
            rows[0],
        )

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
