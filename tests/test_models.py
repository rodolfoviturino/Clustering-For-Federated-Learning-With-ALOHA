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

        self.assertEqual(error_norms.shape, (2, 6))
        self.assertEqual(uploads.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(error_norms)))
        self.assertTrue(np.all(uploads >= 0))

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


if __name__ == "__main__":
    unittest.main()
