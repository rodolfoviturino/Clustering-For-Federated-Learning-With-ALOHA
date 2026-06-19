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


if __name__ == "__main__":
    unittest.main()
