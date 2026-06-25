import csv
import shutil
import unittest
from pathlib import Path

from experiments.compare_runs import summarize_run, write_comparison


class CompareRunsTests(unittest.TestCase):
    def _clean_test_root(self, name):
        root = Path.cwd() / "Runs" / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        return root

    def _write_run(self, root, name, final_error):
        run_dir = Path(root) / name
        run_dir.mkdir()
        csv_path = run_dir / "results.csv"
        rows = [
            {
                "t": 1,
                "optimized_aloha_d2d_error_norm_mean": 1.0,
                "optimized_aloha_d2d_error_norm_ci95": 0.01,
                "optimized_aloha_d2d_uploads_mean": 2.0,
                "optimized_aloha_d2d_uploads_ci95": 0.2,
                "optimized_aloha_d2d_aoi_mean": 1.0,
                "optimized_aloha_d2d_aoi_ci95": 0.1,
                "optimized_aloha_d2d_p90_aoi_mean": 1.0,
                "optimized_aloha_d2d_member_aoi_mean": 1.2,
                "optimized_aloha_d2d_member_aoi_ci95": 0.12,
                "optimized_aloha_d2d_member_stale_fraction_75_mean": 0.3,
                "optimized_aloha_d2d_member_stale_fraction_75_ci95": 0.03,
                "optimized_aloha_d2d_member_zero_participation_fraction_mean": 0.4,
                "optimized_aloha_d2d_member_zero_participation_fraction_ci95": 0.04,
                "optimized_aloha_d2d_energy_efficiency_mean": 10.0,
                "optimized_aloha_d2d_energy_efficiency_ci95": 1.0,
            },
            {
                "t": 2,
                "optimized_aloha_d2d_error_norm_mean": final_error,
                "optimized_aloha_d2d_error_norm_ci95": final_error / 10.0,
                "optimized_aloha_d2d_uploads_mean": 4.0,
                "optimized_aloha_d2d_uploads_ci95": 0.4,
                "optimized_aloha_d2d_aoi_mean": 1.5,
                "optimized_aloha_d2d_aoi_ci95": 0.15,
                "optimized_aloha_d2d_p90_aoi_mean": 2.0,
                "optimized_aloha_d2d_member_aoi_mean": 1.7,
                "optimized_aloha_d2d_member_aoi_ci95": 0.17,
                "optimized_aloha_d2d_member_stale_fraction_75_mean": 0.2,
                "optimized_aloha_d2d_member_stale_fraction_75_ci95": 0.02,
                "optimized_aloha_d2d_member_zero_participation_fraction_mean": 0.1,
                "optimized_aloha_d2d_member_zero_participation_fraction_ci95": 0.01,
                "optimized_aloha_d2d_energy_efficiency_mean": 12.0,
                "optimized_aloha_d2d_energy_efficiency_ci95": 1.2,
            },
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return run_dir

    def test_summarize_run_extracts_threshold_and_final_metrics(self):
        tmpdir = self._clean_test_root("_test_compare_runs_summary")
        try:
            run_dir = self._write_run(tmpdir, "candidate", 1e-12)

            summary = summarize_run(run_dir)

            self.assertEqual(summary["run"], "candidate")
            self.assertEqual(summary["final_t"], 2)
            self.assertEqual(summary["t_to_1e-12"], 2)
            self.assertEqual(summary["final_error_norm"], 1e-12)
            self.assertEqual(summary["final_error_norm_ci95"], 1e-13)
            self.assertEqual(summary["final_aoi"], 1.5)
            self.assertEqual(summary["final_aoi_ci95"], 0.15)
            self.assertEqual(summary["final_member_aoi"], 1.7)
            self.assertEqual(summary["final_member_aoi_ci95"], 0.17)
            self.assertEqual(
                summary["final_member_zero_participation_fraction"],
                0.1,
            )
            self.assertEqual(
                summary["final_member_zero_participation_fraction_ci95"],
                0.01,
            )
            self.assertIn("log_error_auc", summary)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_write_comparison_creates_csv_and_markdown(self):
        tmpdir = self._clean_test_root("_test_compare_runs_write")
        try:
            run_a = self._write_run(tmpdir, "baseline", 1e-9)
            run_b = self._write_run(tmpdir, "candidate", 1e-12)
            summaries = [summarize_run(run_a), summarize_run(run_b)]
            output_dir = Path(tmpdir) / "comparison"

            csv_path, markdown_path = write_comparison(summaries, output_dir)

            self.assertTrue(csv_path.exists())
            self.assertTrue(markdown_path.exists())
            text = markdown_path.read_text(encoding="utf-8")
            self.assertIn("baseline", text)
            self.assertIn("candidate", text)
            self.assertIn("member zero", text)
            self.assertIn("+/-", text)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
