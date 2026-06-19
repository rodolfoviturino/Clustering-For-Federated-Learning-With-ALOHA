import math
import unittest

from Clustering.proposed_clustering_algorithm import (
    devices_clusterizer,
    devices_generator,
    validate_clusters,
)


class ClusteringTests(unittest.TestCase):
    def test_generator_uses_radians_and_radius_range(self):
        devices, _ = devices_generator(100, 300, seed=123)

        for info in devices.values():
            self.assertGreaterEqual(info["device_angle"], 0.0)
            self.assertLess(info["device_angle"], 2.0 * math.pi)
            self.assertGreaterEqual(info["distance_from_device_to_BS"], 1.0)
            self.assertLessEqual(info["distance_from_device_to_BS"], 300.0)

    def test_seeded_generation_and_clustering_are_deterministic(self):
        devices_a, _ = devices_generator(120, 300, seed=42)
        devices_b, _ = devices_generator(120, 300, seed=42)
        self.assertEqual(devices_a, devices_b)

        clusters_a = devices_clusterizer(15, 10, 4, devices_a, seed=99)
        clusters_b = devices_clusterizer(15, 10, 4, devices_b, seed=99)
        self.assertEqual(clusters_a, clusters_b)

    def test_random_cluster_output_satisfies_invariants(self):
        devices, _ = devices_generator(250, 300, seed=7)
        clusters = devices_clusterizer(15, 10, 4, devices, seed=11)

        self.assertTrue(validate_clusters(clusters, devices, 15, 10, 250))

    def test_constructed_merge_overflow_case_stays_within_cmax(self):
        coords = {
            0: (0.0, 0.0),
            1: (0.1, 0.0),
            2: (0.9, 0.0),
            3: (-0.9, 0.0),
            4: (-0.5, 0.0),
            5: (-0.8, 0.5),
        }
        devices = {
            device: {
                "device_angle": 0.0,
                "distance_from_device_to_BS": 1.0,
                "x_y_coord": coord,
                "device_battery": 100,
                "stats_product": 100.0,
            }
            for device, coord in coords.items()
        }

        clusters = devices_clusterizer(
            device_radius=1.0,
            max_devices_per_cluster=5,
            min_devices_per_cluster=1,
            devices_information_dict=devices,
            seed=1,
            shuffle_clusters=False,
        )

        self.assertLessEqual(max(len(cluster) for cluster in clusters), 5)
        self.assertTrue(validate_clusters(clusters, devices, 1.0, 5, 6))

    def test_balancing_preserves_uniqueness_and_one_hop(self):
        devices, _ = devices_generator(180, 300, seed=101)
        clusters = devices_clusterizer(18, 8, 4, devices, seed=202)

        flat_devices = [device for cluster in clusters for device in cluster]
        self.assertEqual(len(flat_devices), len(set(flat_devices)))
        self.assertTrue(validate_clusters(clusters, devices, 18, 8, 180))


if __name__ == "__main__":
    unittest.main()
