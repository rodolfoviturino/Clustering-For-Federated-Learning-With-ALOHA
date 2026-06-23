import unittest

from experiments.run_energy_rotation_sweep import (
    annotate_static_ratios,
    energy_rotation_candidates,
    rank_candidate_summaries,
    select_candidate_slice,
    summarize_candidate,
)


def _fake_rows(final_error, final_energy, final_efficiency, final_ch_battery):
    rows = []
    for t, error in ((1, 4.0), (100, final_error * 1000.0), (200, final_error)):
        rows.append(
            {
                "t": str(t),
                "clusterized_devices_rate_mean": "88.0",
                "optimized_aloha_d2d_error_norm_mean": str(error),
                "optimized_aloha_d2d_error_norm_ci95": str(error * 0.1),
                "optimized_aloha_d2d_clusterhead_uploads_mean": "240.0",
                "fixed_aloha_d2d_clusterhead_uploads_mean": "260.0",
                "optimized_aloha_d2d_uploads_mean": "1200.0",
                "fixed_aloha_d2d_uploads_mean": "700.0",
                "optimized_aloha_d2d_energy_used_mean": str(final_energy),
                "optimized_aloha_d2d_energy_efficiency_mean": str(final_efficiency),
                "optimized_aloha_d2d_clusterhead_battery_mean": str(
                    final_ch_battery
                ),
                "optimized_aloha_d2d_clusterhead_energy_used_mean": "0.018",
                "optimized_aloha_d2d_aoi_mean": "120.0",
                "optimized_aloha_d2d_p75_aoi_mean": "150.0",
                "optimized_aloha_d2d_p90_aoi_mean": "180.0",
                "optimized_aloha_d2d_p95_aoi_mean": "190.0",
                "optimized_aloha_d2d_stale_fraction_75_mean": "0.4",
            }
        )
    return rows


class EnergyRotationSweepTests(unittest.TestCase):
    def test_energy_rotation_candidates_include_static_and_profiles(self):
        candidates = energy_rotation_candidates()

        self.assertEqual(
            [candidate["candidate_id"] for candidate in candidates],
            ["static", "energy_performance", "energy_balanced", "energy_eco"],
        )
        self.assertEqual(candidates[0]["d2d_ch_rotation_mode"], "static")
        self.assertEqual(candidates[1]["d2d_ch_rotation_mode"], "energy_aware")
        self.assertEqual(candidates[1]["d2d_ch_rotation_trigger_mode"], "interval")
        self.assertEqual(candidates[-1]["d2d_energy_efficiency_level"], "eco")

    def test_energy_rotation_candidates_can_include_aoi_triggered_profiles(self):
        candidates = energy_rotation_candidates(include_aoi_triggered=True)

        candidate_ids = [candidate["candidate_id"] for candidate in candidates]
        self.assertIn("energy_performance_aoi", candidate_ids)
        self.assertIn("energy_performance_interval_or_aoi", candidate_ids)
        aoi_candidate = next(
            candidate
            for candidate in candidates
            if candidate["candidate_id"] == "energy_performance_aoi"
        )
        self.assertEqual(aoi_candidate["d2d_ch_rotation_trigger_mode"], "aoi")
        self.assertEqual(aoi_candidate["d2d_energy_efficiency_level"], "performance")

    def test_candidate_slice_validation(self):
        candidates = energy_rotation_candidates()

        self.assertEqual(
            [candidate["candidate_id"] for candidate in select_candidate_slice(candidates, 1, 2)],
            ["energy_performance", "energy_balanced"],
        )
        with self.assertRaises(ValueError):
            select_candidate_slice(candidates, -1)
        with self.assertRaises(ValueError):
            select_candidate_slice(candidates, 0, 0)

    def test_summary_and_ranking_prefer_feasible_ch_battery_gain(self):
        static_candidate = energy_rotation_candidates()[0]
        eco_candidate = {
            "candidate_id": "energy_eco",
            "d2d_ch_rotation_mode": "energy_aware",
            "d2d_energy_efficiency_level": "eco",
        }
        risky_candidate = {
            "candidate_id": "energy_risky",
            "d2d_ch_rotation_mode": "energy_aware",
            "d2d_energy_efficiency_level": "performance",
        }
        static_summary = summarize_candidate(
            static_candidate,
            _fake_rows(final_error=2e-6, final_energy=0.010, final_efficiency=120.0, final_ch_battery=0.48),
            "static/results.csv",
            candidate_index=0,
        )
        eco_summary = summarize_candidate(
            eco_candidate,
            _fake_rows(final_error=1.9e-6, final_energy=0.0102, final_efficiency=122.0, final_ch_battery=0.56),
            "eco/results.csv",
            candidate_index=1,
        )
        risky_summary = summarize_candidate(
            risky_candidate,
            _fake_rows(final_error=1.7e-6, final_energy=0.0120, final_efficiency=100.0, final_ch_battery=0.58),
            "risky/results.csv",
            candidate_index=2,
        )

        ranked = rank_candidate_summaries(
            [static_summary, eco_summary, risky_summary],
            tolerance=1.05,
        )

        self.assertEqual(ranked[0]["candidate_id"], "energy_eco")
        self.assertTrue(ranked[0]["energy_feasible"])
        self.assertFalse(ranked[-1]["energy_feasible"])
        self.assertGreater(ranked[0]["ch_battery_gain_vs_static"], 0.0)

    def test_static_ratio_tolerance_validation(self):
        with self.assertRaises(ValueError):
            annotate_static_ratios([], tolerance=0.99)


if __name__ == "__main__":
    unittest.main()
