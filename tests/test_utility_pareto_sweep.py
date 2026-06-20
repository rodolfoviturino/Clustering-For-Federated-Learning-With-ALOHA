import unittest

from experiments.run_utility_pareto_sweep import (
    rank_candidate_summaries,
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

        summary = summarize_candidate(candidate, rows, "candidate_a/results.csv")

        self.assertTrue(summary["pareto_feasible"])
        self.assertEqual(summary["candidate_id"], "candidate_a")
        self.assertEqual(summary["ch_upload_ratio"], 1.0)
        self.assertEqual(summary["device_upload_gain"], 0.25)
        self.assertEqual(summary["t100_error"], 1e-8)
        self.assertEqual(summary["t200_error"], 1e-12)

    def test_ranking_prefers_feasible_then_lower_auc(self):
        feasible_better = {
            "candidate_id": "feasible_better",
            "pareto_feasible": True,
            "log_error_auc": -9.0,
            "t100_error": 1e-9,
            "t200_error": 1e-12,
            "device_upload_gain": 0.20,
        }
        feasible_worse = {
            "candidate_id": "feasible_worse",
            "pareto_feasible": True,
            "log_error_auc": -7.0,
            "t100_error": 1e-7,
            "t200_error": 1e-10,
            "device_upload_gain": 0.30,
        }
        infeasible_best_error = {
            "candidate_id": "infeasible_best_error",
            "pareto_feasible": False,
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
