import csv
import tempfile
import unittest
from pathlib import Path

from experiments.merge_utility_pareto_summaries import merge_summary_files
from experiments.run_utility_pareto_sweep import SUMMARY_FIELDS


def _summary_row(candidate_id, feasible, auc):
    return {
        "rank": 0,
        "pareto_feasible": feasible,
        "candidate_id": candidate_id,
        "optimized_d2d_access_floor_fraction": 0.1,
        "optimized_d2d_norm_exponent": 2.0,
        "optimized_d2d_cluster_size_exponent": 1.0,
        "optimized_d2d_freshness_exponent": 1.0,
        "optimized_d2d_load_target_factor": 1.0,
        "log_error_auc": auc,
        "t100_error": 1e-8,
        "t200_error": 1e-12,
        "ch_upload_ratio": 1.0,
        "device_upload_gain": 0.2,
        "optimized_d2d_ch_uploads_t200": 100.0,
        "fixed_d2d_ch_uploads_t200": 100.0,
        "optimized_d2d_uploads_t200": 500.0,
        "fixed_d2d_uploads_t200": 400.0,
        "result_csv": f"{candidate_id}/results.csv",
    }


def _write_summary(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class MergeUtilityParetoSummariesTests(unittest.TestCase):
    def test_merge_summary_files_reranks_combined_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "part1.csv"
            second = Path(tmpdir) / "part2.csv"
            _write_summary(first, [_summary_row("candidate_a", True, -6.0)])
            _write_summary(second, [_summary_row("candidate_b", True, -8.0)])

            merged = merge_summary_files([first, second])

            self.assertEqual([row["candidate_id"] for row in merged], ["candidate_b", "candidate_a"])
            self.assertEqual([row["rank"] for row in merged], [1, 2])

    def test_merge_summary_files_rejects_duplicate_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "part1.csv"
            second = Path(tmpdir) / "part2.csv"
            _write_summary(first, [_summary_row("candidate_a", True, -6.0)])
            _write_summary(second, [_summary_row("candidate_a", True, -8.0)])

            with self.assertRaises(ValueError):
                merge_summary_files([first, second])


if __name__ == "__main__":
    unittest.main()
