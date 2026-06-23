import unittest

from experiments.run_ch_quality_weight_sweep import (
    quality_weight_candidates,
    rank_candidate_summaries,
    select_candidate_slice,
    summarize_candidate,
)


def _row(
    t,
    optimized_error,
    direct_optimized_error=1e-1,
    fixed_ch=100.0,
    optimized_ch=100.0,
):
    return {
        "t": str(t),
        "clusterized_devices_rate_mean": "99.0",
        "optimized_aloha_error_norm_mean": str(direct_optimized_error),
        "optimized_aloha_uploads_mean": "80.0",
        "optimized_aloha_d2d_error_norm_mean": str(optimized_error),
        "optimized_aloha_d2d_clusterhead_uploads_mean": str(optimized_ch),
        "fixed_aloha_d2d_clusterhead_uploads_mean": str(fixed_ch),
        "optimized_aloha_d2d_uploads_mean": "160.0",
        "fixed_aloha_d2d_uploads_mean": "100.0",
    }


class ChQualityWeightSweepTests(unittest.TestCase):
    def test_finalist_candidates_are_named_and_ordered(self):
        candidates = quality_weight_candidates("finalists")

        self.assertEqual(
            [candidate["candidate_id"] for candidate in candidates],
            ["w001000_channel_only", "w009010_channel_battery"],
        )
        self.assertEqual(candidates[0]["cluster_head_channel_weight"], 1.0)
        self.assertEqual(candidates[1]["cluster_head_battery_weight"], 0.1)

    def test_candidate_slice_validation(self):
        candidates = quality_weight_candidates("channel_heavy")

        self.assertEqual(len(select_candidate_slice(candidates, 1, 2)), 2)
        with self.assertRaises(ValueError):
            select_candidate_slice(candidates, candidate_start=-1)
        with self.assertRaises(ValueError):
            select_candidate_slice(candidates, candidate_count=0)

    def test_summary_and_ranking_use_curve_auc_first(self):
        candidate_a = {
            "candidate_id": "a",
            "cluster_head_degree_weight": 0.0,
            "cluster_head_channel_weight": 1.0,
            "cluster_head_battery_weight": 0.0,
        }
        candidate_b = {
            "candidate_id": "b",
            "cluster_head_degree_weight": 0.0,
            "cluster_head_channel_weight": 0.9,
            "cluster_head_battery_weight": 0.1,
        }
        summary_a = summarize_candidate(
            candidate_a,
            [_row(1, 1e-2), _row(100, 1e-6), _row(200, 1e-9)],
            "a.csv",
        )
        summary_b = summarize_candidate(
            candidate_b,
            [_row(1, 1e-1), _row(100, 1e-4), _row(200, 1e-8)],
            "b.csv",
        )

        ranked = rank_candidate_summaries([summary_b, summary_a])

        self.assertEqual(ranked[0]["candidate_id"], "a")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["t_to_1e_minus_6"], 100)
        self.assertEqual(ranked[0]["t_to_1e_minus_9"], 200)
        self.assertAlmostEqual(ranked[0]["ch_upload_ratio"], 1.0)
        self.assertGreater(ranked[0]["optimized_d2d_log10_gain_vs_direct_t200"], 0.0)
        self.assertAlmostEqual(
            ranked[0]["optimized_d2d_upload_ratio_vs_direct_t200"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
