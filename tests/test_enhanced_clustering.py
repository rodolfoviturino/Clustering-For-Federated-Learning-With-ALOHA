import unittest

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


if __name__ == "__main__":
    unittest.main()
