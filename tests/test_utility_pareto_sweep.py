import unittest

from experiments.run_utility_pareto_sweep import (
    candidate_grid,
    rank_candidate_summaries,
    select_candidate_slice,
    summarize_candidate,
)


def _row(t, optimized_error, optimized_ch, fixed_ch, optimized_uploads, fixed_uploads):
    return {
        "t": str(t),
        "optimized_aloha_d2d_error_norm_mean": str(optimized_error),
        "optimized_aloha_d2d_clusterhead_uploads_mean": str(optimized_ch),
        "fixed_aloha_d2d_clusterhead_uploads_mean": str(fixed_ch),
        "optimized_aloha_d2d_uploads_mean": str(optimized_uploads),
        "fixed_aloha_d2d_uploads_mean": str(fixed_uploads),
    }


class UtilityParetoSweepTests(unittest.TestCase):
    def test_candidate_grid_profiles_have_expected_sizes(self):
        self.assertEqual(len(candidate_grid("coarse")), 243)
        self.assertEqual(len(candidate_grid("refined")), 162)

    def test_candidate_slice_uses_zero_based_grid_indices(self):
        candidates = candidate_grid()
        sliced = select_candidate_slice(candidates, candidate_start=3, candidate_count=2)

        self.assertEqual(len(sliced), 2)
        self.assertEqual(sliced[0], candidates[3])
        self.assertEqual(sliced[1], candidates[4])

    def test_candidate_slice_rejects_invalid_bounds(self):
        candidates = candidate_grid()

        with self.assertRaises(ValueError):
            select_candidate_slice(candidates, candidate_start=-1)
        with self.assertRaises(ValueError):
            select_candidate_slice(candidates, candidate_start=0, candidate_count=0)

    def test_candidate_summary_computes_pareto_metrics(self):
        candidate = {
            "candidate_id": "candidate_a",
            "optimized_d2d_access_floor_fraction": 0.10,
            "optimized_d2d_norm_exponent": 2.0,
            "optimized_d2d_cluster_size_exponent": 1.0,
            "optimized_d2d_freshness_exponent": 1.0,
            "optimized_d2d_load_target_factor": 1.0,
        }
        rows = [
            _row(1, 1.0, 10, 10, 50, 40),
            _row(100, 1e-8, 500, 500, 2500, 2000),
            _row(200, 1e-12, 1000, 1000, 5000, 4000),
        ]

        summary = summarize_candidate(candidate, rows, "candidate_a/results.csv", 12)

        self.assertTrue(summary["pareto_feasible"])
        self.assertEqual(summary["candidate_grid_index"], 12)
        self.assertEqual(summary["candidate_id"], "candidate_a")
        self.assertEqual(summary["ch_upload_ratio"], 1.0)
        self.assertEqual(summary["device_upload_gain"], 0.25)
        self.assertEqual(summary["t_to_1e_minus_6"], 100)
        self.assertEqual(summary["t_to_1e_minus_9"], 200)
        self.assertEqual(summary["t_to_1e_minus_12"], 200)
        self.assertEqual(summary["target_time_score"], 500)
        self.assertEqual(summary["t100_error"], 1e-8)
        self.assertEqual(summary["t200_error"], 1e-12)

    def test_ranking_prefers_feasible_then_lower_target_time(self):
        feasible_better = {
            "candidate_id": "feasible_better",
            "pareto_feasible": True,
            "target_time_score": 120,
            "t_to_1e_minus_12": 60,
            "t_to_1e_minus_9": 40,
            "log_error_auc": -9.0,
            "t100_error": 1e-9,
            "t200_error": 1e-12,
            "device_upload_gain": 0.20,
        }
        feasible_worse = {
            "candidate_id": "feasible_worse",
            "pareto_feasible": True,
            "target_time_score": 180,
            "t_to_1e_minus_12": 90,
            "t_to_1e_minus_9": 60,
            "log_error_auc": -7.0,
            "t100_error": 1e-7,
            "t200_error": 1e-10,
            "device_upload_gain": 0.30,
        }
        infeasible_best_error = {
            "candidate_id": "infeasible_best_error",
            "pareto_feasible": False,
            "target_time_score": 30,
            "t_to_1e_minus_12": 10,
            "t_to_1e_minus_9": 10,
            "log_error_auc": -12.0,
            "t100_error": 1e-12,
            "t200_error": 1e-14,
            "device_upload_gain": 0.60,
        }

        ranked = rank_candidate_summaries(
            [feasible_worse, infeasible_best_error, feasible_better]
        )

        self.assertEqual(ranked[0]["candidate_id"], "feasible_better")
        self.assertEqual(ranked[1]["candidate_id"], "feasible_worse")
        self.assertEqual(ranked[2]["candidate_id"], "infeasible_best_error")
        self.assertEqual([row["rank"] for row in ranked], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
