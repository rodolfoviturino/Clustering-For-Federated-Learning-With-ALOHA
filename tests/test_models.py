import math
import unittest

import numpy as np

import Models.jax_models_arrangement as jax_models
from Models.models_arrangement import (
    error_calculator,
    error_calculator_trace_jax,
    prepare_clusters_for_jax,
)


JAX_AVAILABLE = jax_models.jax is not None


@unittest.skipUnless(JAX_AVAILABLE, "JAX is not installed in this interpreter")
class JaxModelTests(unittest.TestCase):
    def test_prepare_clusters_for_jax_returns_padded_arrays(self):
        clusters = prepare_clusters_for_jax([[0, 1, 2], [3]])

        members = np.asarray(clusters.cluster_members)
        sizes = np.asarray(clusters.cluster_sizes)
        heads = np.asarray(clusters.cluster_heads)

        self.assertEqual(members.dtype, np.int32)
        self.assertEqual(sizes.tolist(), [3, 1])
        self.assertEqual(heads.tolist(), [0, 3])
        self.assertEqual(members.tolist(), [[0, 1, 2], [3, -1, -1]])

    def test_error_trace_returns_finite_outputs_at_checkpoints(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3]])
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            max_iterations_t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=5,
            checkpoints=[1, 3],
        )

        error_norms = np.asarray(result.error_norms)
        uploads = np.asarray(result.successful_uploads)
        mean_aoi = np.asarray(result.mean_aoi)
        peak_aoi = np.asarray(result.peak_aoi)
        p75_aoi = np.asarray(result.p75_aoi)
        p90_aoi = np.asarray(result.p90_aoi)
        p95_aoi = np.asarray(result.p95_aoi)
        stale_fraction_50 = np.asarray(result.stale_fraction_50)
        stale_fraction_75 = np.asarray(result.stale_fraction_75)
        stale_fraction_100 = np.asarray(result.stale_fraction_100)

        self.assertEqual(error_norms.shape, (2, 6))
        self.assertEqual(uploads.shape, (2, 6))
        self.assertEqual(mean_aoi.shape, (2, 6))
        self.assertEqual(peak_aoi.shape, (2, 6))
        self.assertEqual(p75_aoi.shape, (2, 6))
        self.assertEqual(p90_aoi.shape, (2, 6))
        self.assertEqual(p95_aoi.shape, (2, 6))
        self.assertEqual(stale_fraction_50.shape, (2, 6))
        self.assertEqual(stale_fraction_75.shape, (2, 6))
        self.assertEqual(stale_fraction_100.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(error_norms)))
        self.assertTrue(np.all(uploads >= 0))
        self.assertTrue(np.all(np.isfinite(mean_aoi)))
        self.assertTrue(np.all(np.isfinite(peak_aoi)))
        self.assertTrue(np.all(np.isfinite(p75_aoi)))
        self.assertTrue(np.all(np.isfinite(p90_aoi)))
        self.assertTrue(np.all(np.isfinite(p95_aoi)))
        self.assertTrue(np.all(np.isfinite(stale_fraction_50)))
        self.assertTrue(np.all(np.isfinite(stale_fraction_75)))
        self.assertTrue(np.all(np.isfinite(stale_fraction_100)))
        self.assertTrue(np.all(mean_aoi >= 1.0))
        self.assertTrue(np.all(peak_aoi >= 1.0))
        self.assertTrue(np.all(p75_aoi >= 1.0))
        self.assertTrue(np.all(p90_aoi >= 1.0))
        self.assertTrue(np.all(p95_aoi >= 1.0))
        self.assertTrue(np.all(stale_fraction_50 >= 0.0))
        self.assertTrue(np.all(stale_fraction_75 >= 0.0))
        self.assertTrue(np.all(stale_fraction_100 >= 0.0))
        self.assertTrue(np.all(stale_fraction_50 <= 1.0))
        self.assertTrue(np.all(stale_fraction_75 <= 1.0))
        self.assertTrue(np.all(stale_fraction_100 <= 1.0))

    def test_legacy_error_calculator_tuple_shape(self):
        result = error_calculator(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            number_of_iterations__t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=[[0, 1], [2, 3]],
            seed=5,
        )

        self.assertEqual(len(result), 16)
        for value in result[:7]:
            self.assertTrue(math.isfinite(value))

    def test_d2d_link_success_controls_member_upload_accounting(self):
        kwargs = dict(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            number_of_iterations__t=1,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=[[0, 1, 2], [3]],
            seed=123,
        )

        thesis_default = error_calculator(**kwargs)
        no_member_links = error_calculator(
            **kwargs,
            d2d_member_link_success_probability=0.0,
        )

        self.assertEqual(thesis_default[10], 4)
        self.assertEqual(no_member_links[10], 2)

    def test_ch_bs_success_probability_uses_channel_and_battery(self):
        jnp = jax_models.jnp
        probability = jax_models._d2d_ch_bs_success_probability(
            cluster_heads=jnp.asarray([0, 1, 2], dtype=jnp.int32),
            cluster_mask=jnp.asarray([True, True, False]),
            number_of_devices=3,
            dtype=jnp.float32,
            success_mode="channel_quality",
            min_success_probability=jnp.asarray(0.2, dtype=jnp.float32),
            pathloss_exponent=jnp.asarray(2.0, dtype=jnp.float32),
            battery_exponent=jnp.asarray(1.0, dtype=jnp.float32),
            device_distance_to_bs=jnp.asarray([1.0, 10.0, 20.0], dtype=jnp.float32),
            device_battery=jnp.asarray([100.0, 50.0, 10.0], dtype=jnp.float32),
        )

        probability = np.asarray(probability)

        self.assertAlmostEqual(probability[0], 1.0, places=6)
        self.assertGreater(probability[1], 0.2)
        self.assertLess(probability[1], probability[0])
        self.assertEqual(probability[2], 0.0)

    def test_device_bs_success_probability_uses_channel_and_battery(self):
        jnp = jax_models.jnp
        probability = jax_models._device_bs_success_probability(
            number_of_devices=3,
            dtype=jnp.float32,
            success_mode="channel_quality",
            min_success_probability=jnp.asarray(0.25, dtype=jnp.float32),
            pathloss_exponent=jnp.asarray(2.0, dtype=jnp.float32),
            battery_exponent=jnp.asarray(1.0, dtype=jnp.float32),
            device_distance_to_bs=jnp.asarray([1.0, 10.0, 20.0], dtype=jnp.float32),
            device_battery=jnp.asarray([100.0, 50.0, 10.0], dtype=jnp.float32),
        )

        probability = np.asarray(probability)

        self.assertEqual(probability.shape, (3,))
        self.assertAlmostEqual(probability[0], 1.0, places=6)
        self.assertGreater(probability[1], 0.25)
        self.assertLess(probability[1], probability[0])
        self.assertGreaterEqual(probability[2], 0.25)
        self.assertLess(probability[2], probability[1])

    def test_rayleigh_outage_probability_is_monotonic(self):
        jnp = jax_models.jnp
        distances = jnp.asarray([1.0, 10.0, 50.0], dtype=jnp.float32)
        probability = np.asarray(
            jax_models._rayleigh_outage_success_probability(
                number_of_devices=3,
                dtype=jnp.float32,
                device_distance_to_bs=distances,
                pathloss_exponent=2.0,
                reference_snr=1000.0,
                snr_threshold=1.0,
            )
        )
        higher_snr = np.asarray(
            jax_models._rayleigh_outage_success_probability(
                number_of_devices=3,
                dtype=jnp.float32,
                device_distance_to_bs=distances,
                pathloss_exponent=2.0,
                reference_snr=2000.0,
                snr_threshold=1.0,
            )
        )
        higher_threshold = np.asarray(
            jax_models._rayleigh_outage_success_probability(
                number_of_devices=3,
                dtype=jnp.float32,
                device_distance_to_bs=distances,
                pathloss_exponent=2.0,
                reference_snr=1000.0,
                snr_threshold=2.0,
            )
        )

        self.assertGreater(probability[0], probability[1])
        self.assertGreater(probability[1], probability[2])
        self.assertTrue(np.all(higher_snr >= probability))
        self.assertTrue(np.all(higher_threshold <= probability))

    def test_first_order_radio_energy_increases_with_distance_and_size(self):
        jnp = jax_models.jnp
        distances = jnp.asarray([1.0, 10.0, 20.0], dtype=jnp.float32)
        energy = np.asarray(
            jax_models._first_order_bs_tx_energy(
                distances,
                packet_size=1.0,
                electronics_cost=0.0002,
                amplifier_cost=1e-6,
                pathloss_exponent=2.0,
            )
        )
        larger_packet = np.asarray(
            jax_models._first_order_bs_tx_energy(
                distances,
                packet_size=2.0,
                electronics_cost=0.0002,
                amplifier_cost=1e-6,
                pathloss_exponent=2.0,
            )
        )
        d2d_energy = np.asarray(
            jax_models._first_order_d2d_tx_energy(
                distances,
                packet_size=1.0,
                electronics_cost=0.0002,
                amplifier_cost=1e-6,
                pathloss_exponent=2.0,
            )
        )

        self.assertGreater(energy[1], energy[0])
        self.assertGreater(energy[2], energy[1])
        self.assertTrue(np.all(larger_packet > energy))
        np.testing.assert_allclose(d2d_energy, energy, rtol=1e-6)

    def test_rayleigh_outage_trace_returns_finite_outputs(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3]])
        jnp = jax_models.jnp
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            max_iterations_t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=21,
            d2d_ch_bs_success_mode="rayleigh_outage",
            d2d_ch_bs_reference_snr=100000.0,
            d2d_ch_bs_snr_threshold=1.0,
            device_bs_success_mode="rayleigh_outage",
            device_bs_reference_snr=100000.0,
            device_bs_snr_threshold=1.0,
            device_distance_to_bs=jnp.asarray([5.0, 80.0, 10.0, 70.0]),
            checkpoints=[1, 3],
        )

        self.assertEqual(np.asarray(result.error_norms).shape, (2, 6))
        self.assertEqual(np.asarray(result.mean_aoi).shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.error_norms))))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.mean_aoi))))

    def test_ch_bs_channel_quality_trace_returns_finite_outputs(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3]])
        jnp = jax_models.jnp
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            max_iterations_t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=17,
            d2d_ch_bs_success_mode="channel_quality",
            d2d_ch_bs_min_success_probability=0.2,
            d2d_ch_bs_pathloss_exponent=2.0,
            d2d_ch_bs_battery_exponent=0.5,
            device_distance_to_bs=jnp.asarray([5.0, 80.0, 10.0, 70.0]),
            device_battery=jnp.asarray([100.0, 30.0, 90.0, 20.0]),
            checkpoints=[1, 3],
        )

        error_norms = np.asarray(result.error_norms)
        uploads = np.asarray(result.successful_uploads)
        clusterhead_uploads = np.asarray(result.successful_clusterhead_uploads)

        self.assertEqual(error_norms.shape, (2, 6))
        self.assertEqual(uploads.shape, (2, 6))
        self.assertEqual(clusterhead_uploads.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(error_norms)))
        self.assertTrue(np.all(uploads >= 0))
        self.assertTrue(np.all(clusterhead_uploads >= 0))

    def test_device_bs_channel_quality_trace_returns_finite_outputs(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3]])
        jnp = jax_models.jnp
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            max_iterations_t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=19,
            device_bs_success_mode="channel_quality",
            device_bs_min_success_probability=0.3,
            device_bs_pathloss_exponent=2.0,
            device_bs_battery_exponent=0.5,
            device_distance_to_bs=jnp.asarray([5.0, 80.0, 10.0, 70.0]),
            device_battery=jnp.asarray([100.0, 30.0, 90.0, 20.0]),
            checkpoints=[1, 3],
        )

        error_norms = np.asarray(result.error_norms)
        uploads = np.asarray(result.successful_uploads)

        self.assertEqual(error_norms.shape, (2, 6))
        self.assertEqual(uploads.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(error_norms)))
        self.assertTrue(np.all(uploads >= 0))

    def test_dynamic_energy_drain_records_battery_metrics(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3]])
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            max_iterations_t=2,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=23,
            device_battery=jax_models.jnp.asarray([100.0, 100.0, 100.0, 100.0]),
            energy_drain_mode="dynamic",
            energy_direct_bs_cost=0.01,
            energy_d2d_member_cost=0.005,
            energy_ch_bs_cost=0.02,
            checkpoints=[1, 2],
        )

        mean_battery = np.asarray(result.mean_battery)
        mean_clusterhead_battery = np.asarray(result.mean_clusterhead_battery)
        mean_energy_used = np.asarray(result.mean_energy_used)
        energy_efficiency = np.asarray(result.energy_efficiency)
        mean_clusterhead_energy_used = np.asarray(result.mean_clusterhead_energy_used)

        self.assertEqual(mean_battery.shape, (2, 6))
        self.assertEqual(mean_clusterhead_battery.shape, (2, 3))
        self.assertEqual(mean_energy_used.shape, (2, 6))
        self.assertEqual(energy_efficiency.shape, (2, 6))
        self.assertEqual(mean_clusterhead_energy_used.shape, (2, 3))
        self.assertLess(mean_battery[0, 0], 1.0)
        self.assertTrue(np.all(mean_battery >= 0.0))
        self.assertTrue(np.all(mean_battery <= 1.0))
        self.assertTrue(np.all(mean_clusterhead_battery >= 0.0))
        self.assertTrue(np.all(mean_clusterhead_battery <= 1.0))
        self.assertGreater(mean_clusterhead_battery[0, 0], 0.90)
        self.assertTrue(np.all(mean_energy_used >= 0.0))
        self.assertTrue(np.all(np.isfinite(energy_efficiency)))
        self.assertTrue(np.all(mean_clusterhead_energy_used >= 0.0))

    def test_battery_feasibility_blocks_attempts_when_energy_is_insufficient(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3]])
        jnp = jax_models.jnp
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=4,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            max_iterations_t=2,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=25,
            device_coords=jnp.asarray(
                [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
                dtype=jnp.float32,
            ),
            device_distance_to_bs=jnp.asarray([100.0, 100.0, 100.0, 100.0]),
            device_battery=jnp.zeros(4, dtype=jnp.float32),
            energy_drain_mode="dynamic",
            energy_model="first_order_radio",
            battery_feasibility_mode="required_energy",
            checkpoints=[1, 2],
        )

        uploads = np.asarray(result.successful_uploads)
        self.assertEqual(uploads.shape, (2, 6))
        self.assertTrue(np.all(uploads == 0))

    def test_ch_bs_success_parameters_are_validated(self):
        invalid_kwargs = (
            {"d2d_ch_bs_success_mode": "invalid"},
            {"d2d_ch_bs_min_success_probability": -0.1},
            {"d2d_ch_bs_min_success_probability": 1.1},
            {"d2d_ch_bs_pathloss_exponent": -0.1},
            {"d2d_ch_bs_battery_exponent": -0.1},
            {"d2d_ch_bs_reference_snr": 0.0},
            {"d2d_ch_bs_snr_threshold": -0.1},
        )

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=4,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=1.0,
                        number_of_iterations__t=1,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3]],
                        seed=17,
                        **kwargs,
                    )

    def test_device_bs_success_parameters_are_validated(self):
        invalid_kwargs = (
            {"device_bs_success_mode": "invalid"},
            {"device_bs_min_success_probability": -0.1},
            {"device_bs_min_success_probability": 1.1},
            {"device_bs_pathloss_exponent": -0.1},
            {"device_bs_battery_exponent": -0.1},
            {"device_bs_reference_snr": 0.0},
            {"device_bs_snr_threshold": -0.1},
        )

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=4,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=1.0,
                        number_of_iterations__t=1,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3]],
                        seed=17,
                        **kwargs,
                    )

    def test_energy_drain_parameters_are_validated(self):
        invalid_kwargs = (
            {"energy_drain_mode": "invalid"},
            {"energy_model": "invalid"},
            {"battery_feasibility_mode": "invalid"},
            {"energy_direct_bs_cost": -0.1},
            {"energy_d2d_member_cost": -0.1},
            {"energy_ch_bs_cost": -0.1},
            {"energy_electronics_cost": -0.1},
            {"energy_bs_amplifier_cost": -0.1},
            {"energy_d2d_amplifier_cost": -0.1},
            {"energy_bs_pathloss_exponent": -0.1},
            {"energy_d2d_pathloss_exponent": -0.1},
            {"energy_aggregation_cost": -0.1},
            {"energy_update_size": 0.0},
            {"energy_aggregate_size": 0.0},
            {"energy_rotation_control_cost": -0.1},
        )

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=4,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=1.0,
                        number_of_iterations__t=1,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3]],
                        seed=17,
                        **kwargs,
                    )

    def test_d2d_energy_efficiency_profiles_are_named_weights(self):
        self.assertEqual(
            jax_models._d2d_energy_efficiency_profile_weights("performance"),
            (0.85, 0.10, 0.05),
        )
        self.assertEqual(
            jax_models._d2d_energy_efficiency_profile_weights("balanced"),
            (0.65, 0.25, 0.10),
        )
        self.assertEqual(
            jax_models._d2d_energy_efficiency_profile_weights("eco"),
            (0.45, 0.45, 0.10),
        )
        with self.assertRaises(ValueError):
            jax_models._d2d_energy_efficiency_profile_weights("invalid")

    def test_d2d_ch_rotation_parameters_are_validated(self):
        base_kwargs = dict(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            number_of_iterations__t=1,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=[[0, 1], [2, 3]],
            seed=17,
        )
        invalid_kwargs = (
            {"d2d_ch_rotation_mode": "invalid"},
            {"d2d_ch_rotation_interval": 0},
            {"d2d_ch_rotation_trigger_mode": "invalid"},
            {"d2d_ch_rotation_aoi_threshold_fraction": -0.1},
            {"d2d_ch_rotation_aoi_threshold_fraction": 1.1},
            {
                "d2d_ch_rotation_mode": "energy_aware",
                "energy_drain_mode": "none",
                "device_coords": [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
                "device_radius": 2.0,
            },
            {
                "d2d_ch_rotation_mode": "energy_aware",
                "energy_drain_mode": "dynamic",
            },
            {"d2d_energy_efficiency_level": "invalid"},
        )

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    error_calculator(**base_kwargs, **kwargs)

    def test_energy_aware_d2d_ch_rotation_records_energy_metrics(self):
        clusters = prepare_clusters_for_jax([[0, 1, 2], [3]])
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            max_iterations_t=2,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=29,
            device_coords=jax_models.jnp.asarray(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]
            ),
            device_radius=2.0,
            device_distance_to_bs=jax_models.jnp.asarray([80.0, 5.0, 60.0, 10.0]),
            device_battery=jax_models.jnp.asarray([100.0, 100.0, 100.0, 100.0]),
            energy_drain_mode="dynamic",
            energy_direct_bs_cost=0.01,
            energy_d2d_member_cost=0.005,
            energy_ch_bs_cost=0.02,
            d2d_ch_rotation_mode="energy_aware",
            d2d_ch_rotation_interval=1,
            d2d_ch_rotation_trigger_mode="interval",
            d2d_energy_efficiency_level="balanced",
            d2d_ch_bs_success_mode="channel_quality",
            d2d_ch_bs_min_success_probability=0.35,
            d2d_ch_bs_pathloss_exponent=2.0,
            d2d_ch_bs_battery_exponent=0.25,
            checkpoints=[1, 2],
        )

        mean_energy_used = np.asarray(result.mean_energy_used)
        energy_efficiency = np.asarray(result.energy_efficiency)
        mean_clusterhead_energy_used = np.asarray(result.mean_clusterhead_energy_used)

        self.assertEqual(mean_energy_used.shape, (2, 6))
        self.assertEqual(energy_efficiency.shape, (2, 6))
        self.assertEqual(mean_clusterhead_energy_used.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(mean_energy_used)))
        self.assertTrue(np.all(np.isfinite(energy_efficiency)))
        self.assertTrue(np.all(np.isfinite(mean_clusterhead_energy_used)))
        self.assertTrue(np.all(mean_energy_used >= 0.0))
        self.assertTrue(np.all(mean_clusterhead_energy_used >= 0.0))

    def test_aoi_triggered_d2d_ch_rotation_returns_finite_outputs(self):
        clusters = prepare_clusters_for_jax([[0, 1, 2], [3, 4, 5]])
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=6,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            max_iterations_t=4,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=37,
            device_coords=jax_models.jnp.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [10.0, 10.0],
                    [11.0, 10.0],
                    [10.0, 11.0],
                ]
            ),
            device_radius=2.0,
            device_distance_to_bs=jax_models.jnp.asarray(
                [80.0, 5.0, 60.0, 70.0, 6.0, 65.0]
            ),
            device_battery=jax_models.jnp.asarray(
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
            ),
            energy_drain_mode="dynamic",
            energy_direct_bs_cost=0.01,
            energy_d2d_member_cost=0.005,
            energy_ch_bs_cost=0.02,
            d2d_ch_rotation_mode="energy_aware",
            d2d_ch_rotation_interval=10,
            d2d_ch_rotation_trigger_mode="aoi",
            d2d_ch_rotation_aoi_threshold_fraction=0.75,
            d2d_energy_efficiency_level="performance",
            d2d_ch_bs_success_mode="channel_quality",
            d2d_ch_bs_min_success_probability=0.35,
            d2d_ch_bs_pathloss_exponent=2.0,
            checkpoints=[1, 4],
        )

        self.assertEqual(np.asarray(result.error_norms).shape, (2, 6))
        self.assertEqual(np.asarray(result.mean_clusterhead_battery).shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.error_norms))))
        self.assertTrue(
            np.all(np.isfinite(np.asarray(result.mean_clusterhead_battery)))
        )

    def test_static_d2d_ch_rotation_preserves_default_behavior(self):
        clusters = prepare_clusters_for_jax([[0, 1, 2], [3]])
        base_kwargs = dict(
            number_of_mobile_devices__k=4,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            max_iterations_t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=31,
            device_battery=jax_models.jnp.asarray([100.0, 100.0, 100.0, 100.0]),
            energy_drain_mode="dynamic",
            energy_direct_bs_cost=0.01,
            energy_d2d_member_cost=0.005,
            energy_ch_bs_cost=0.02,
            checkpoints=[1, 2, 3],
        )
        default_result = error_calculator_trace_jax(**base_kwargs)
        static_with_coords = error_calculator_trace_jax(
            **base_kwargs,
            d2d_ch_rotation_mode="static",
            device_coords=jax_models.jnp.asarray(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]
            ),
            device_radius=2.0,
        )

        np.testing.assert_allclose(
            np.asarray(default_result.error_norms),
            np.asarray(static_with_coords.error_norms),
        )
        np.testing.assert_array_equal(
            np.asarray(default_result.successful_uploads),
            np.asarray(static_with_coords.successful_uploads),
        )
        np.testing.assert_allclose(
            np.asarray(default_result.mean_battery),
            np.asarray(static_with_coords.mean_battery),
        )

    def test_normalization_parameter_changes_learning_scale(self):
        kwargs = dict(
            number_of_mobile_devices__k=6,
            data_dimension__L=3,
            number_of_parallel_channels__M=3,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            number_of_iterations__t=2,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=[[0, 1], [2, 3], [4, 5]],
            seed=77,
        )

        normalized = error_calculator(**kwargs, normalize_by_k=True)
        unnormalized = error_calculator(**kwargs, normalize_by_k=False)

        self.assertNotEqual(normalized[1], unnormalized[1])
        self.assertTrue(math.isfinite(normalized[1]))
        self.assertTrue(math.isfinite(unnormalized[1]))

    def test_more_than_255_contenders_do_not_overflow_indices(self):
        clusters = [[device] for device in range(300)]
        result = error_calculator(
            number_of_mobile_devices__k=300,
            data_dimension__L=2,
            number_of_parallel_channels__M=300,
            probability_that_user_can_compute_its_local_update__pcomp=1.0,
            number_of_iterations__t=1,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=clusters,
            seed=9,
        )

        self.assertEqual(len(result), 16)
        self.assertGreaterEqual(result[8], 0)

    def test_optimized_access_floor_parameters_are_supported(self):
        result = error_calculator(
            number_of_mobile_devices__k=6,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            number_of_iterations__t=2,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=[[0, 1], [2, 3], [4, 5]],
            seed=13,
            optimized_access_floor_fraction=1.0,
            optimized_d2d_access_floor_fraction=1.0,
        )

        self.assertEqual(len(result), 16)
        for value in result[:7]:
            self.assertTrue(math.isfinite(value))

    def test_utility_optimized_d2d_access_mode_is_supported(self):
        result = error_calculator(
            number_of_mobile_devices__k=6,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            number_of_iterations__t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=[[0, 1], [2, 3], [4, 5]],
            seed=21,
            optimized_d2d_access_mode="utility",
            optimized_d2d_access_floor_fraction=0.25,
            optimized_d2d_norm_exponent=1.0,
            optimized_d2d_cluster_size_exponent=1.0,
            optimized_d2d_freshness_exponent=0.5,
            optimized_d2d_load_target_factor=1.2,
        )

        self.assertEqual(len(result), 16)
        for value in result[:7]:
            self.assertTrue(math.isfinite(value))

    def test_utility_load_target_factor_must_be_positive(self):
        with self.assertRaises(ValueError):
            error_calculator(
                number_of_mobile_devices__k=6,
                data_dimension__L=2,
                number_of_parallel_channels__M=2,
                probability_that_user_can_compute_its_local_update__pcomp=0.5,
                number_of_iterations__t=3,
                learning_rate__u1=0.01,
                step_size__u=0.1,
                clusters_list=[[0, 1], [2, 3], [4, 5]],
                seed=21,
                optimized_d2d_access_mode="utility",
                optimized_d2d_load_target_factor=0.0,
            )

    def test_water_filling_allocator_preserves_limits_and_target_load(self):
        jnp = jax_models.jnp
        pcomp = jnp.asarray(0.6, dtype=jnp.float32)
        fixed_probability = jnp.asarray(0.1, dtype=jnp.float32)
        probability = jax_models._load_controlled_access_from_utility(
            utility=jnp.asarray([10.0, 5.0, 1.0, 0.5], dtype=jnp.float32),
            cluster_mask=jnp.asarray([True, True, True, True]),
            n_channels=2,
            pcomp=pcomp,
            fixed_access_probability=fixed_probability,
            floor_fraction=jnp.asarray(0.2, dtype=jnp.float32),
            load_target_factor=jnp.asarray(1.0, dtype=jnp.float32),
            load_allocation_mode="water_filling",
            redistribution_fraction=jnp.asarray(1.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.0, dtype=jnp.float32),
            fixed_success_target=jnp.asarray(1.0, dtype=jnp.float32),
        )

        probability = np.asarray(probability)
        floor = float(fixed_probability * 0.2)

        self.assertEqual(probability.shape, (4,))
        self.assertTrue(np.all(probability >= floor - 1e-6))
        self.assertTrue(np.all(probability <= float(pcomp) + 1e-6))
        self.assertAlmostEqual(float(np.sum(probability)), 2.0, places=5)

    def test_proportional_clip_allocator_preserves_legacy_clipping(self):
        jnp = jax_models.jnp
        utility = jnp.asarray([10.0, 5.0, 1.0, 0.5], dtype=jnp.float32)
        cluster_mask = jnp.asarray([True, True, True, True])
        probability = jax_models._load_controlled_access_from_utility(
            utility=utility,
            cluster_mask=cluster_mask,
            n_channels=2,
            pcomp=jnp.asarray(0.6, dtype=jnp.float32),
            fixed_access_probability=jnp.asarray(0.1, dtype=jnp.float32),
            floor_fraction=jnp.asarray(0.2, dtype=jnp.float32),
            load_target_factor=jnp.asarray(1.0, dtype=jnp.float32),
            load_allocation_mode="proportional_clip",
            redistribution_fraction=jnp.asarray(0.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.0, dtype=jnp.float32),
            fixed_success_target=jnp.asarray(1.0, dtype=jnp.float32),
        )

        floor = 0.1 * 0.2
        remaining_load = 2.0 - 4.0 * floor
        expected = floor + remaining_load * np.asarray(utility) / np.sum(
            np.asarray(utility)
        )
        expected = np.minimum(expected, 0.6)

        np.testing.assert_allclose(np.asarray(probability), expected, rtol=1e-6)

    def test_selective_water_filling_interpolates_between_allocators(self):
        jnp = jax_models.jnp
        common_kwargs = dict(
            utility=jnp.asarray([10.0, 5.0, 1.0, 0.5], dtype=jnp.float32),
            cluster_mask=jnp.asarray([True, True, True, True]),
            n_channels=2,
            pcomp=jnp.asarray(0.6, dtype=jnp.float32),
            fixed_access_probability=jnp.asarray(0.1, dtype=jnp.float32),
            floor_fraction=jnp.asarray(0.2, dtype=jnp.float32),
            load_target_factor=jnp.asarray(1.0, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.0, dtype=jnp.float32),
            fixed_success_target=jnp.asarray(1.0, dtype=jnp.float32),
        )

        clipped = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="proportional_clip",
            redistribution_fraction=jnp.asarray(0.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
        )
        selective = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="selective_water_filling",
            redistribution_fraction=jnp.asarray(0.5, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
        )
        waterfilled = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="water_filling",
            redistribution_fraction=jnp.asarray(1.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
        )

        clipped_load = float(np.sum(np.asarray(clipped)))
        selective_load = float(np.sum(np.asarray(selective)))
        waterfilled_load = float(np.sum(np.asarray(waterfilled)))

        self.assertGreater(selective_load, clipped_load)
        self.assertLess(selective_load, waterfilled_load)
        self.assertAlmostEqual(
            selective_load,
            clipped_load + 0.5 * (2.0 - clipped_load),
            places=5,
        )

    def test_conditional_selective_water_filling_activates_when_throughput_is_low(self):
        jnp = jax_models.jnp
        common_kwargs = dict(
            utility=jnp.asarray([10.0, 5.0, 1.0, 0.5], dtype=jnp.float32),
            cluster_mask=jnp.asarray([True, True, True, True]),
            n_channels=2,
            pcomp=jnp.asarray(0.6, dtype=jnp.float32),
            fixed_access_probability=jnp.asarray(0.1, dtype=jnp.float32),
            floor_fraction=jnp.asarray(0.2, dtype=jnp.float32),
            load_target_factor=jnp.asarray(1.0, dtype=jnp.float32),
            fixed_success_target=jnp.asarray(1.0, dtype=jnp.float32),
        )

        clipped = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="proportional_clip",
            redistribution_fraction=jnp.asarray(0.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.4, dtype=jnp.float32),
        )
        conditional = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="conditional_selective_water_filling",
            redistribution_fraction=jnp.asarray(0.5, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.4, dtype=jnp.float32),
        )

        clipped_load = float(np.sum(np.asarray(clipped)))
        conditional_load = float(np.sum(np.asarray(conditional)))

        self.assertGreater(conditional_load, clipped_load)
        self.assertAlmostEqual(
            conditional_load,
            clipped_load + 0.5 * (2.0 - clipped_load),
            places=5,
        )

    def test_conditional_selective_water_filling_skips_when_throughput_is_high(self):
        jnp = jax_models.jnp
        common_kwargs = dict(
            utility=jnp.asarray([2.0, 1.8, 1.6, 1.4], dtype=jnp.float32),
            cluster_mask=jnp.asarray([True, True, True, True]),
            n_channels=2,
            pcomp=jnp.asarray(0.6, dtype=jnp.float32),
            fixed_access_probability=jnp.asarray(0.1, dtype=jnp.float32),
            floor_fraction=jnp.asarray(0.2, dtype=jnp.float32),
            load_target_factor=jnp.asarray(1.0, dtype=jnp.float32),
            fixed_success_target=jnp.asarray(1.0, dtype=jnp.float32),
        )

        clipped = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="proportional_clip",
            redistribution_fraction=jnp.asarray(0.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(1.0, dtype=jnp.float32),
        )
        conditional = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="conditional_selective_water_filling",
            redistribution_fraction=jnp.asarray(0.5, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(1.0, dtype=jnp.float32),
        )

        np.testing.assert_allclose(
            np.asarray(conditional),
            np.asarray(clipped),
            rtol=1e-6,
        )

    def test_conditional_trigger_uses_throughput_not_inflated_target(self):
        jnp = jax_models.jnp
        common_kwargs = dict(
            utility=jnp.asarray([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32),
            cluster_mask=jnp.asarray([True, True, True, True]),
            n_channels=2,
            pcomp=jnp.asarray(0.7, dtype=jnp.float32),
            fixed_access_probability=jnp.asarray(0.1, dtype=jnp.float32),
            floor_fraction=jnp.asarray(0.0, dtype=jnp.float32),
            load_target_factor=jnp.asarray(1.1, dtype=jnp.float32),
            fixed_success_target=jnp.asarray(1.0, dtype=jnp.float32),
        )

        clipped = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="proportional_clip",
            redistribution_fraction=jnp.asarray(0.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(1.0, dtype=jnp.float32),
        )
        conditional = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="conditional_selective_water_filling",
            redistribution_fraction=jnp.asarray(0.5, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(1.0, dtype=jnp.float32),
        )
        selective = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="selective_water_filling",
            redistribution_fraction=jnp.asarray(0.5, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(1.0, dtype=jnp.float32),
        )

        # Even with an inflated target load, high observed throughput should
        # make the conditional mode skip redistribution.  This is the K=3000
        # regression case: extra contenders reduced useful CH successes.
        self.assertGreaterEqual(float(np.sum(np.asarray(clipped))), 1.9)
        np.testing.assert_allclose(
            np.asarray(conditional),
            np.asarray(clipped),
            rtol=1e-6,
        )
        self.assertGreater(
            float(np.sum(np.asarray(selective))),
            float(np.sum(np.asarray(conditional))),
        )

    def test_conditional_dense_regime_uses_conservative_trigger(self):
        jnp = jax_models.jnp
        common_kwargs = dict(
            utility=jnp.asarray([10.0, 5.0, 1.0, 0.5], dtype=jnp.float32),
            cluster_mask=jnp.asarray([True, True, True, True]),
            n_channels=2,
            pcomp=jnp.asarray(0.6, dtype=jnp.float32),
            fixed_access_probability=jnp.asarray(0.1, dtype=jnp.float32),
            floor_fraction=jnp.asarray(0.2, dtype=jnp.float32),
            load_target_factor=jnp.asarray(1.0, dtype=jnp.float32),
            fixed_success_target=jnp.asarray(1.0, dtype=jnp.float32),
        )

        clipped = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="proportional_clip",
            redistribution_fraction=jnp.asarray(0.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.92, dtype=jnp.float32),
        )
        conditional = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="conditional_selective_water_filling",
            redistribution_fraction=jnp.asarray(0.5, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.92, dtype=jnp.float32),
            density_trigger_threshold=jnp.asarray(0.95, dtype=jnp.float32),
            dense_trigger_ratio=jnp.asarray(0.90, dtype=jnp.float32),
            clusterized_devices_fraction=jnp.asarray(0.99, dtype=jnp.float32),
        )

        # With 99% clusterization the effective trigger is 0.90, so an observed
        # throughput ratio of 0.92 is good enough and the allocator must fall
        # back exactly to proportional clipping.
        np.testing.assert_allclose(
            np.asarray(conditional),
            np.asarray(clipped),
            rtol=1e-6,
        )

    def test_conditional_sparse_regime_keeps_base_trigger(self):
        jnp = jax_models.jnp
        common_kwargs = dict(
            utility=jnp.asarray([10.0, 5.0, 1.0, 0.5], dtype=jnp.float32),
            cluster_mask=jnp.asarray([True, True, True, True]),
            n_channels=2,
            pcomp=jnp.asarray(0.6, dtype=jnp.float32),
            fixed_access_probability=jnp.asarray(0.1, dtype=jnp.float32),
            floor_fraction=jnp.asarray(0.2, dtype=jnp.float32),
            load_target_factor=jnp.asarray(1.0, dtype=jnp.float32),
            fixed_success_target=jnp.asarray(1.0, dtype=jnp.float32),
        )

        clipped = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="proportional_clip",
            redistribution_fraction=jnp.asarray(0.0, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.92, dtype=jnp.float32),
        )
        conditional = jax_models._load_controlled_access_from_utility(
            **common_kwargs,
            load_allocation_mode="conditional_selective_water_filling",
            redistribution_fraction=jnp.asarray(0.5, dtype=jnp.float32),
            redistribution_trigger_ratio=jnp.asarray(0.95, dtype=jnp.float32),
            optimized_success_ewma=jnp.asarray(0.92, dtype=jnp.float32),
            density_trigger_threshold=jnp.asarray(0.95, dtype=jnp.float32),
            dense_trigger_ratio=jnp.asarray(0.90, dtype=jnp.float32),
            clusterized_devices_fraction=jnp.asarray(0.80, dtype=jnp.float32),
        )

        # With 80% clusterization the allocator is still in the sparse regime,
        # so the base 0.95 trigger remains active for a 0.92 throughput ratio.
        self.assertGreater(
            float(np.sum(np.asarray(conditional))),
            float(np.sum(np.asarray(clipped))),
        )

    def test_utility_load_allocation_mode_must_be_valid(self):
        with self.assertRaises(ValueError):
            error_calculator(
                number_of_mobile_devices__k=6,
                data_dimension__L=2,
                number_of_parallel_channels__M=2,
                probability_that_user_can_compute_its_local_update__pcomp=0.5,
                number_of_iterations__t=3,
                learning_rate__u1=0.01,
                step_size__u=0.1,
                clusters_list=[[0, 1], [2, 3], [4, 5]],
                seed=21,
                optimized_d2d_access_mode="utility",
                optimized_d2d_load_allocation_mode="invalid",
            )

    def test_selective_redistribution_fraction_must_be_in_unit_interval(self):
        for redistribution_fraction in (-0.1, 1.1):
            with self.subTest(redistribution_fraction=redistribution_fraction):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=6,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=0.5,
                        number_of_iterations__t=3,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3], [4, 5]],
                        seed=21,
                        optimized_d2d_access_mode="utility",
                        optimized_d2d_load_allocation_mode="selective_water_filling",
                        optimized_d2d_redistribution_fraction=redistribution_fraction,
                    )

    def test_conditional_redistribution_trigger_must_be_in_unit_interval(self):
        for trigger_ratio in (-0.1, 1.1):
            with self.subTest(trigger_ratio=trigger_ratio):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=6,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=0.5,
                        number_of_iterations__t=3,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3], [4, 5]],
                        seed=21,
                        optimized_d2d_access_mode="utility",
                        optimized_d2d_load_allocation_mode=(
                            "conditional_selective_water_filling"
                        ),
                        optimized_d2d_redistribution_trigger_ratio=trigger_ratio,
                    )

    def test_conditional_density_trigger_threshold_must_be_in_unit_interval(self):
        for density_threshold in (-0.1, 1.1):
            with self.subTest(density_threshold=density_threshold):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=6,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=0.5,
                        number_of_iterations__t=3,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3], [4, 5]],
                        seed=21,
                        optimized_d2d_access_mode="utility",
                        optimized_d2d_load_allocation_mode=(
                            "conditional_selective_water_filling"
                        ),
                        optimized_d2d_density_trigger_threshold=density_threshold,
                    )

    def test_conditional_dense_trigger_ratio_must_be_in_unit_interval(self):
        for dense_trigger_ratio in (-0.1, 1.1):
            with self.subTest(dense_trigger_ratio=dense_trigger_ratio):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=6,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=0.5,
                        number_of_iterations__t=3,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3], [4, 5]],
                        seed=21,
                        optimized_d2d_access_mode="utility",
                        optimized_d2d_load_allocation_mode=(
                            "conditional_selective_water_filling"
                        ),
                        optimized_d2d_dense_trigger_ratio=dense_trigger_ratio,
                    )

    def test_conditional_throughput_ewma_decay_must_be_in_unit_interval(self):
        for ewma_decay in (-0.1, 1.1):
            with self.subTest(ewma_decay=ewma_decay):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=6,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=0.5,
                        number_of_iterations__t=3,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3], [4, 5]],
                        seed=21,
                        optimized_d2d_access_mode="utility",
                        optimized_d2d_load_allocation_mode=(
                            "conditional_selective_water_filling"
                        ),
                        optimized_d2d_throughput_ewma_decay=ewma_decay,
                    )

    def test_max_weight_optimized_d2d_access_mode_is_supported(self):
        result = error_calculator(
            number_of_mobile_devices__k=6,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            number_of_iterations__t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=[[0, 1], [2, 3], [4, 5]],
            seed=34,
            optimized_d2d_access_mode="max_weight",
            optimized_d2d_access_floor_fraction=0.10,
            optimized_d2d_norm_exponent=2.0,
            optimized_d2d_cluster_size_exponent=1.0,
            optimized_d2d_freshness_exponent=1.0,
            optimized_d2d_threshold_gain=8.0,
        )

        self.assertEqual(len(result), 16)
        for value in result[:7]:
            self.assertTrue(math.isfinite(value))

    def test_hybrid_optimized_d2d_access_mode_is_supported(self):
        result = error_calculator(
            number_of_mobile_devices__k=6,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            number_of_iterations__t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters_list=[[0, 1], [2, 3], [4, 5]],
            seed=55,
            optimized_d2d_access_mode="hybrid",
            optimized_d2d_access_floor_fraction=0.10,
            optimized_d2d_norm_exponent=2.0,
            optimized_d2d_cluster_size_exponent=1.0,
            optimized_d2d_freshness_exponent=1.0,
            optimized_d2d_novelty_exponent=1.0,
            optimized_d2d_novelty_floor=0.25,
            optimized_d2d_reference_decay=0.90,
        )

        self.assertEqual(len(result), 16)
        for value in result[:7]:
            self.assertTrue(math.isfinite(value))

    def test_adaptive_diversity_optimized_d2d_access_mode_is_supported(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3], [4, 5]])
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=6,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            max_iterations_t=3,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=89,
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
            optimized_d2d_adaptive_switch_fraction=0.30,
            optimized_d2d_adaptive_switch_gain=12.0,
            checkpoints=[1, 3],
        )

        error_norms = np.asarray(result.error_norms)
        uploads = np.asarray(result.successful_uploads)
        clusterhead_uploads = np.asarray(result.successful_clusterhead_uploads)

        self.assertEqual(error_norms.shape, (2, 6))
        self.assertEqual(uploads.shape, (2, 6))
        self.assertEqual(clusterhead_uploads.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(error_norms)))
        self.assertTrue(np.all(uploads >= 0))
        self.assertTrue(np.all(clusterhead_uploads >= 0))

    def test_adaptive_diversity_switch_fraction_must_be_in_unit_interval(self):
        for switch_fraction in (-0.1, 1.1):
            with self.subTest(switch_fraction=switch_fraction):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=6,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=0.5,
                        number_of_iterations__t=3,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3], [4, 5]],
                        seed=89,
                        optimized_d2d_access_mode="adaptive_diversity",
                        optimized_d2d_adaptive_switch_fraction=switch_fraction,
                    )

    def test_adaptive_diversity_switch_gain_must_be_positive(self):
        with self.assertRaises(ValueError):
            error_calculator(
                number_of_mobile_devices__k=6,
                data_dimension__L=2,
                number_of_parallel_channels__M=2,
                probability_that_user_can_compute_its_local_update__pcomp=0.5,
                number_of_iterations__t=3,
                learning_rate__u1=0.01,
                step_size__u=0.1,
                clusters_list=[[0, 1], [2, 3], [4, 5]],
                seed=89,
                optimized_d2d_access_mode="adaptive_diversity",
                optimized_d2d_adaptive_switch_gain=0.0,
            )

    def test_adaptive_diversity_late_exponents_must_be_non_negative(self):
        invalid_kwargs = (
            {"optimized_d2d_late_norm_exponent": -0.1},
            {"optimized_d2d_late_freshness_exponent": -0.1},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=6,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=0.5,
                        number_of_iterations__t=3,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3], [4, 5]],
                        seed=89,
                        optimized_d2d_access_mode="adaptive_diversity",
                        **kwargs,
                    )

    def test_aoi_aware_utility_trace_returns_finite_outputs(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3], [4, 5]])
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=6,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            max_iterations_t=4,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=91,
            optimized_d2d_access_mode="aoi_aware_utility",
            optimized_d2d_norm_exponent=2.0,
            optimized_d2d_cluster_size_exponent=1.0,
            optimized_d2d_freshness_exponent=0.25,
            optimized_d2d_aoi_weight=0.75,
            optimized_d2d_aoi_exponent=1.5,
            optimized_d2d_aoi_threshold_fraction=0.70,
            checkpoints=[1, 4],
        )

        self.assertEqual(np.asarray(result.error_norms).shape, (2, 6))
        self.assertEqual(np.asarray(result.p75_aoi).shape, (2, 6))
        self.assertEqual(np.asarray(result.p90_aoi).shape, (2, 6))
        self.assertEqual(np.asarray(result.p95_aoi).shape, (2, 6))
        self.assertEqual(np.asarray(result.stale_fraction_50).shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.error_norms))))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.p75_aoi))))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.p90_aoi))))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.p95_aoi))))
        self.assertTrue(
            np.all(np.isfinite(np.asarray(result.stale_fraction_50)))
        )

    def test_aoi_floor_utility_trace_returns_finite_outputs(self):
        clusters = prepare_clusters_for_jax([[0, 1], [2, 3], [4, 5]])
        result = error_calculator_trace_jax(
            number_of_mobile_devices__k=6,
            data_dimension__L=2,
            number_of_parallel_channels__M=2,
            probability_that_user_can_compute_its_local_update__pcomp=0.5,
            max_iterations_t=4,
            learning_rate__u1=0.01,
            step_size__u=0.1,
            clusters=clusters,
            seed=93,
            optimized_d2d_access_mode="aoi_floor_utility",
            optimized_d2d_norm_exponent=2.0,
            optimized_d2d_cluster_size_exponent=1.0,
            optimized_d2d_freshness_exponent=0.25,
            optimized_d2d_aoi_weight=0.25,
            optimized_d2d_aoi_exponent=1.0,
            optimized_d2d_aoi_threshold_fraction=0.85,
            checkpoints=[1, 4],
        )

        self.assertEqual(np.asarray(result.error_norms).shape, (2, 6))
        self.assertEqual(np.asarray(result.p90_aoi).shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.error_norms))))
        self.assertTrue(np.all(np.isfinite(np.asarray(result.p90_aoi))))

    def test_aoi_aware_utility_parameters_are_validated(self):
        invalid_kwargs = (
            {"optimized_d2d_aoi_weight": -0.1},
            {"optimized_d2d_aoi_exponent": -0.1},
            {"optimized_d2d_aoi_threshold_fraction": -0.1},
            {"optimized_d2d_aoi_threshold_fraction": 1.0},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    error_calculator(
                        number_of_mobile_devices__k=6,
                        data_dimension__L=2,
                        number_of_parallel_channels__M=2,
                        probability_that_user_can_compute_its_local_update__pcomp=0.5,
                        number_of_iterations__t=3,
                        learning_rate__u1=0.01,
                        step_size__u=0.1,
                        clusters_list=[[0, 1], [2, 3], [4, 5]],
                        seed=91,
                        optimized_d2d_access_mode="aoi_aware_utility",
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
