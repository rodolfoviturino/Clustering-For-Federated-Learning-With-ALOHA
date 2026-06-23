import csv
import tempfile
import unittest
from pathlib import Path

from experiments.compare_runs import summarize_run, write_comparison


class CompareRunsTests(unittest.TestCase):
    def _write_run(self, root, name, final_error):
        run_dir = Path(root) / name
        run_dir.mkdir()
        csv_path = run_dir / "results.csv"
        rows = [
            {
                "t": 1,
                "optimized_aloha_d2d_error_norm_mean": 1.0,
                "optimized_aloha_d2d_uploads_mean": 2.0,
                "optimized_aloha_d2d_aoi_mean": 1.0,
                "optimized_aloha_d2d_p90_aoi_mean": 1.0,
                "optimized_aloha_d2d_energy_efficiency_mean": 10.0,
            },
            {
                "t": 2,
                "optimized_aloha_d2d_error_norm_mean": final_error,
                "optimized_aloha_d2d_uploads_mean": 4.0,
                "optimized_aloha_d2d_aoi_mean": 1.5,
                "optimized_aloha_d2d_p90_aoi_mean": 2.0,
                "optimized_aloha_d2d_energy_efficiency_mean": 12.0,
            },
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return run_dir

    def test_summarize_run_extracts_threshold_and_final_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._write_run(tmpdir, "candidate", 1e-12)

            summary = summarize_run(run_dir)

            self.assertEqual(summary["run"], "candidate")
            self.assertEqual(summary["final_t"], 2)
            self.assertEqual(summary["t_to_1e-12"], 2)
            self.assertEqual(summary["final_error_norm"], 1e-12)
            self.assertEqual(summary["final_aoi"], 1.5)
            self.assertIn("log_error_auc", summary)

    def test_write_comparison_creates_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
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


if __name__ == "__main__":
    unittest.main()
