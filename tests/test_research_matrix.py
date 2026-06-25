import csv
import io
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from experiments import run_research_matrix as research_matrix


class ResearchMatrixTests(unittest.TestCase):
    def _clean_test_root(self, name):
        root = Path.cwd() / "Runs" / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        return root

    def _write_minimal_results(self, runs_dir, run_name, final_error=1e-12):
        run_dir = Path(runs_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "results.csv"
        rows = [
            {
                "t": 1,
                "optimized_aloha_d2d_error_norm_mean": 1.0,
                "optimized_aloha_d2d_error_norm_ci95": 0.1,
                "optimized_aloha_d2d_aoi_mean": 2.0,
                "optimized_aloha_d2d_member_aoi_mean": 3.0,
                "optimized_aloha_d2d_member_aoi_ci95": 0.3,
                "optimized_aloha_d2d_member_stale_fraction_75_mean": 0.5,
                "optimized_aloha_d2d_member_stale_fraction_75_ci95": 0.05,
                "optimized_aloha_d2d_member_zero_participation_fraction_mean": 0.4,
                "optimized_aloha_d2d_member_zero_participation_fraction_ci95": 0.04,
                "optimized_aloha_d2d_energy_efficiency_mean": 10.0,
                "optimized_aloha_d2d_energy_efficiency_ci95": 1.0,
            },
            {
                "t": 2,
                "optimized_aloha_d2d_error_norm_mean": final_error,
                "optimized_aloha_d2d_error_norm_ci95": final_error / 10.0,
                "optimized_aloha_d2d_aoi_mean": 1.5,
                "optimized_aloha_d2d_member_aoi_mean": 2.5,
                "optimized_aloha_d2d_member_aoi_ci95": 0.25,
                "optimized_aloha_d2d_member_stale_fraction_75_mean": 0.3,
                "optimized_aloha_d2d_member_stale_fraction_75_ci95": 0.03,
                "optimized_aloha_d2d_member_zero_participation_fraction_mean": 0.2,
                "optimized_aloha_d2d_member_zero_participation_fraction_ci95": 0.02,
                "optimized_aloha_d2d_energy_efficiency_mean": 12.0,
                "optimized_aloha_d2d_energy_efficiency_ci95": 1.2,
            },
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_all_matrices_have_unique_run_names(self):
        for matrix_name in research_matrix.MATRICES:
            with self.subTest(matrix=matrix_name):
                matrix = research_matrix.get_matrix(matrix_name)

                run_names = [entry.run_name for entry in matrix]
                self.assertEqual(len(run_names), len(set(run_names)))

    def test_minimal_robustness_matrices_have_expected_profiles(self):
        k1000 = research_matrix.get_matrix("k1000_minimal")
        k5000 = research_matrix.get_matrix("k5000_minimal")
        combined = research_matrix.get_matrix("k1000_k5000_minimal")

        self.assertEqual(len(k1000), 4)
        self.assertEqual(len(k5000), 4)
        self.assertEqual(len(combined), 8)
        self.assertTrue(all(entry.overrides["devices"] == 1000 for entry in k1000))
        self.assertTrue(all(entry.overrides["devices"] == 5000 for entry in k5000))
        self.assertIn(
            "member_capped_quota_k1000_w015_cap010_physical_r100",
            {entry.run_name for entry in k1000},
        )
        self.assertIn(
            "member_semischedule_k5000_s030_cc0010_physical_r100",
            {entry.run_name for entry in k5000},
        )

    def test_compare_only_cli_writes_comparison_and_paper_summary(self):
        tmpdir = self._clean_test_root("_test_research_matrix_compare")
        try:
            for index, entry in enumerate(research_matrix.get_matrix("k3000_core")):
                self._write_minimal_results(
                    tmpdir,
                    entry.run_name,
                    final_error=10.0 ** (-(index + 6)),
                )
            output_dir = tmpdir / "comparison"

            with redirect_stdout(io.StringIO()):
                research_matrix.main(
                    [
                        "--matrix",
                        "k3000_core",
                        "--compare-only",
                        "--runs-dir",
                        str(tmpdir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertTrue((output_dir / "run_comparison_summary.csv").exists())
            self.assertTrue((output_dir / "run_comparison_summary.md").exists())
            paper_summary = output_dir / "paper_claim_summary.md"
            self.assertTrue(paper_summary.exists())
            text = paper_summary.read_text(encoding="utf-8")
            self.assertIn("main pure-ALOHA baseline", text)
            self.assertIn("coordinated upper-bound", text)
            self.assertIn("+/-", text)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_combined_robustness_summary_lists_density_baselines(self):
        tmpdir = self._clean_test_root("_test_research_matrix_robustness_compare")
        try:
            for index, entry in enumerate(
                research_matrix.get_matrix("k1000_k5000_minimal")
            ):
                self._write_minimal_results(
                    tmpdir,
                    entry.run_name,
                    final_error=10.0 ** (-(index + 6)),
                )
            output_dir = tmpdir / "comparison"

            with redirect_stdout(io.StringIO()):
                research_matrix.main(
                    [
                        "--matrix",
                        "k1000_k5000_minimal",
                        "--compare-only",
                        "--runs-dir",
                        str(tmpdir),
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            text = (output_dir / "paper_claim_summary.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("K=1000/K=5000", text)
            self.assertIn(
                "member_quota_k1000_w015_floor000_physical_r100",
                text,
            )
            self.assertIn(
                "member_quota_k5000_w015_floor000_physical_r100",
                text,
            )
            self.assertIn("respective densities", text)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_execute_missing_skips_existing_runs(self):
        tmpdir = self._clean_test_root("_test_research_matrix_execute_missing")
        try:
            matrix = research_matrix.get_matrix("k3000_core")
            for entry in matrix:
                self._write_minimal_results(tmpdir, entry.run_name)

            with mock.patch.object(research_matrix, "_execute_entry") as execute_entry:
                generated = research_matrix.execute_missing(
                    matrix,
                    tmpdir,
                    matrix_name="k3000_core",
                )

            self.assertEqual(generated, [])
            execute_entry.assert_not_called()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
