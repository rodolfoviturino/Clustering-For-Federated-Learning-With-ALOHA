import unittest

import numpy as np

from Clustering.enhanced_clustering_algorithm import devices_clusterizer_v2
import Clustering.jax_clustering_algorithm as jax_clustering
from Clustering.jax_clustering_algorithm import (
    clusterizer_jax,
    devices_generator_jax,
    validate_jax_cluster_result,
)
from Clustering.proposed_clustering_algorithm import (
    devices_clusterizer,
    devices_generator,
    validate_clusters,
)


def _clustered_rate(clusters, n_devices):
    return (1.0 - sum(1 for cluster in clusters if len(cluster) == 1) / n_devices) * 100.0


class EnhancedClusteringTests(unittest.TestCase):
    def test_v2_output_satisfies_cluster_invariants(self):
        devices, _ = devices_generator(160, 300, seed=123)
        clusters = devices_clusterizer_v2(15, 10, 4, devices, 300, seed=321)

        self.assertTrue(validate_clusters(clusters, devices, 15, 10, 160))

    def test_v2_is_deterministic_with_seed(self):
        devices, _ = devices_generator(120, 300, seed=77)
        clusters_a = devices_clusterizer_v2(15, 10, 4, devices, 300, seed=99)
        clusters_b = devices_clusterizer_v2(15, 10, 4, devices, 300, seed=99)

        self.assertEqual(clusters_a, clusters_b)

    def test_v2_smoke_cluster_rate_is_not_worse_than_baseline(self):
        devices, _ = devices_generator(120, 40, seed=14)
        baseline = devices_clusterizer(15, 10, 4, devices, seed=14)
        v2 = devices_clusterizer_v2(15, 10, 4, devices, 40, seed=14)

        self.assertGreaterEqual(_clustered_rate(v2, 120), _clustered_rate(baseline, 120))

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_geometric_clusters_satisfy_one_hop_invariants(self):
        devices = devices_generator_jax(32, 80, seed=1)
        clusters = clusterizer_jax(
            devices=devices,
            device_radius=15.0,
            max_devices_per_cluster=8,
            min_devices_per_cluster=1,
            clustering_mode="geometric",
        )

        self.assertTrue(
            validate_jax_cluster_result(
                clusters,
                devices.coords,
                device_radius=15.0,
                max_devices_per_cluster=8,
                n_devices=32,
            )
        )

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_dense_pair_first_configuration_is_valid(self):
        devices = devices_generator_jax(32, 80, seed=5)
        clusters = clusterizer_jax(
            devices=devices,
            device_radius=15.0,
            max_devices_per_cluster=8,
            min_devices_per_cluster=1,
            clustering_mode="geometric",
            initial_cluster_size=2,
            repair_passes=1,
        )

        self.assertTrue(
            validate_jax_cluster_result(
                clusters,
                devices.coords,
                device_radius=15.0,
                max_devices_per_cluster=8,
                n_devices=32,
            )
        )

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_no_d2d_returns_singletons(self):
        devices = devices_generator_jax(10, 50, seed=2)
        clusters = clusterizer_jax(
            devices=devices,
            device_radius=15.0,
            max_devices_per_cluster=4,
            clustering_mode="no_d2d",
        )

        self.assertEqual(int(clusters.number_of_clusters), 10)
        self.assertEqual(int(clusters.singleton_count), 10)
        self.assertEqual(float(clusters.clusterized_devices_rate), 0.0)

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_local_repair_absorbs_reachable_singleton(self):
        jnp = jax_clustering.jnp
        cluster_members = jnp.asarray(
            [
                [0, 1, -1],
                [2, -1, -1],
                [3, -1, -1],
                [-1, -1, -1],
            ],
            dtype=jnp.int32,
        )
        cluster_sizes = jnp.asarray([2, 1, 1, 0], dtype=jnp.int32)
        coords = jnp.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [20.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        repaired_members, repaired_sizes = jax_clustering._repair_singleton_join_requests(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=coords,
            device_radius=2.0,
            max_devices_per_cluster=3,
            repair_passes=1,
        )

        self.assertEqual(np.asarray(repaired_sizes).tolist(), [3, 1, 0, 0])
        self.assertEqual(np.asarray(repaired_members[0]).tolist(), [0, 1, 2])

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_pair_rotation_absorbs_singleton_reachable_from_member(self):
        jnp = jax_clustering.jnp
        cluster_members = jnp.asarray(
            [
                [0, 1, -1],
                [2, -1, -1],
                [-1, -1, -1],
            ],
            dtype=jnp.int32,
        )
        cluster_sizes = jnp.asarray([2, 1, 0], dtype=jnp.int32)
        coords = jnp.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        repaired_members, repaired_sizes = jax_clustering._repair_singleton_pair_rotations(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=coords,
            device_radius=1.1,
            rotation_repair_passes=1,
        )

        self.assertEqual(np.asarray(repaired_sizes).tolist(), [3, 0, 0])
        self.assertEqual(np.asarray(repaired_members[0]).tolist(), [1, 0, 2])

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_local_cluster_merge_combines_one_hop_unions(self):
        jnp = jax_clustering.jnp
        cluster_members = jnp.asarray(
            [
                [0, 1, -1, -1],
                [2, 3, -1, -1],
                [4, -1, -1, -1],
            ],
            dtype=jnp.int32,
        )
        cluster_sizes = jnp.asarray([2, 2, 1], dtype=jnp.int32)
        coords = jnp.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [1.1, 0.0],
                [10.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        merged_members, merged_sizes = jax_clustering._merge_local_cluster_heads(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=coords,
            device_radius=1.05,
            merge_passes=1,
        )

        self.assertEqual(np.asarray(merged_sizes).tolist(), [4, 1, 0])
        self.assertEqual(np.asarray(merged_members[0]).tolist(), [2, 3, 0, 1])

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_quality_ch_rotation_preserves_one_hop_coverage(self):
        jnp = jax_clustering.jnp
        cluster_members = jnp.asarray(
            [
                [0, 1, 2, -1],
                [3, 4, 5, -1],
            ],
            dtype=jnp.int32,
        )
        cluster_sizes = jnp.asarray([3, 3], dtype=jnp.int32)
        coords = jnp.asarray(
            [
                [0.0, 0.0],
                [0.9, 0.0],
                [-0.9, 0.0],
                [10.0, 0.0],
                [10.2, 0.0],
                [10.4, 0.0],
            ],
            dtype=jnp.float32,
        )
        quality_score = jnp.asarray([0.1, 0.9, 0.2, 0.1, 0.8, 0.3], dtype=jnp.float32)

        rotated_members = jax_clustering._rotate_cluster_heads_by_quality(
            cluster_members=cluster_members,
            cluster_sizes=cluster_sizes,
            coords=coords,
            device_radius=1.0,
            quality_score=quality_score,
        )

        # Row 0 keeps device 0 as CH even though device 1 has the highest
        # quality, because device 1 cannot directly cover device 2.
        self.assertEqual(np.asarray(rotated_members[0]).tolist(), [0, 1, 2, -1])
        # Row 1 rotates to device 4 because it has the highest quality among
        # members that can still cover the whole one-hop cluster.
        self.assertEqual(np.asarray(rotated_members[1]).tolist(), [4, 3, 5, -1])

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_quality_ch_selection_clusters_remain_valid(self):
        devices = devices_generator_jax(40, 80, seed=8)
        clusters = clusterizer_jax(
            devices=devices,
            device_radius=15.0,
            max_devices_per_cluster=8,
            min_devices_per_cluster=1,
            clustering_mode="geometric",
            cluster_head_selection_mode="quality",
            cluster_head_degree_weight=0.4,
            cluster_head_channel_weight=0.4,
            cluster_head_battery_weight=0.2,
        )

        self.assertTrue(
            validate_jax_cluster_result(
                clusters,
                devices.coords,
                device_radius=15.0,
                max_devices_per_cluster=8,
                n_devices=40,
            )
        )

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_quality_ch_selection_accepts_rayleigh_channel_score(self):
        devices = devices_generator_jax(40, 80, seed=9)
        clusters = clusterizer_jax(
            devices=devices,
            device_radius=15.0,
            max_devices_per_cluster=8,
            min_devices_per_cluster=1,
            clustering_mode="geometric",
            cluster_head_selection_mode="quality",
            cluster_head_degree_weight=0.0,
            cluster_head_channel_weight=1.0,
            cluster_head_battery_weight=0.0,
            cluster_head_channel_score_mode="rayleigh_outage",
            cluster_head_reference_snr=100000.0,
            cluster_head_snr_threshold=1.0,
        )

        self.assertTrue(
            validate_jax_cluster_result(
                clusters,
                devices.coords,
                device_radius=15.0,
                max_devices_per_cluster=8,
                n_devices=40,
            )
        )

    @unittest.skipUnless(jax_clustering.jax is not None, "JAX is not installed in this interpreter")
    def test_jax_quality_ch_selection_rejects_invalid_weights(self):
        devices = devices_generator_jax(10, 50, seed=3)

        with self.assertRaises(ValueError):
            clusterizer_jax(
                devices=devices,
                device_radius=15.0,
                max_devices_per_cluster=4,
                cluster_head_selection_mode="quality",
                cluster_head_degree_weight=0.0,
                cluster_head_channel_weight=0.0,
                cluster_head_battery_weight=0.0,
            )

        with self.assertRaises(ValueError):
            clusterizer_jax(
                devices=devices,
                device_radius=15.0,
                max_devices_per_cluster=4,
                cluster_head_selection_mode="quality",
                cluster_head_degree_weight=-0.1,
            )

        with self.assertRaises(ValueError):
            clusterizer_jax(
                devices=devices,
                device_radius=15.0,
                max_devices_per_cluster=4,
                cluster_head_selection_mode="quality",
                cluster_head_channel_score_mode="invalid",
            )

        with self.assertRaises(ValueError):
            clusterizer_jax(
                devices=devices,
                device_radius=15.0,
                max_devices_per_cluster=4,
                cluster_head_selection_mode="quality",
                cluster_head_channel_score_mode="rayleigh_outage",
                cluster_head_reference_snr=0.0,
            )

        with self.assertRaises(ValueError):
            clusterizer_jax(
                devices=devices,
                device_radius=15.0,
                max_devices_per_cluster=4,
                cluster_head_selection_mode="quality",
                cluster_head_channel_score_mode="rayleigh_outage",
                cluster_head_snr_threshold=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
