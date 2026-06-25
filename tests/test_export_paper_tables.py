import csv
import io
import shutil
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from experiments import export_paper_tables


class ExportPaperTablesTests(unittest.TestCase):
    def _clean_test_root(self, name):
        root = Path.cwd() / "Runs" / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        return root

    def _write_comparison_csv(self, root):
        csv_path = Path(root) / "run_comparison_summary.csv"
        rows = [
            {
                "run": "member_quota_k3000_w015_floor000_physical_r100",
                "claim_role": "main pure-ALOHA baseline",
                "optimized_d2d_access_mode": "member_quota_utility",
                "t_to_1e-12": "100",
                "final_error_norm": "4.216e-13",
                "final_error_norm_ci95": "1e-14",
                "final_member_aoi": "74.23",
                "final_member_aoi_ci95": "0.12",
                "final_member_stale_fraction_75": "0.618",
                "final_member_stale_fraction_75_ci95": "0.003",
                "final_member_zero_participation_fraction": "0.554",
                "final_member_zero_participation_fraction_ci95": "0.004",
                "final_energy_efficiency": "653.8",
                "final_energy_efficiency_ci95": "4.2",
            },
            {
                "run": "member_collision_quota_k3000_w015_t002_g2_min050_physical_r100",
                "claim_role": "pure-ALOHA convergence/energy ablation",
                "optimized_d2d_access_mode": "member_collision_aware_quota",
                "t_to_1e-12": "88",
                "final_error_norm": "1.816e-14",
                "final_error_norm_ci95": "2e-15",
                "final_member_aoi": "73.97",
                "final_member_aoi_ci95": "0.11",
                "final_member_stale_fraction_75": "0.614",
                "final_member_stale_fraction_75_ci95": "0.002",
                "final_member_zero_participation_fraction": "0.556",
                "final_member_zero_participation_fraction_ci95": "0.005",
                "final_energy_efficiency": "688.7",
                "final_energy_efficiency_ci95": "3.1",
            },
            {
                "run": "member_split_k3000_s5_w015_physical_r100",
                "claim_role": "negative structural ablation",
                "optimized_d2d_access_mode": "member_quota_utility",
                "t_to_1e-12": "",
                "final_error_norm": "3.120e-02",
                "final_error_norm_ci95": "1e-03",
                "final_member_aoi": "93.79",
                "final_member_aoi_ci95": "0.20",
                "final_member_stale_fraction_75": "0.891",
                "final_member_stale_fraction_75_ci95": "0.010",
                "final_member_zero_participation_fraction": "0.835",
                "final_member_zero_participation_fraction_ci95": "0.015",
                "final_energy_efficiency": "103.9",
                "final_energy_efficiency_ci95": "2.5",
            },
            {
                "run": "member_semischedule_k3000_s030_cc0010_physical_r100",
                "claim_role": "coordinated upper-bound / future work",
                "optimized_d2d_access_mode": "semi_scheduled_member_refresh",
                "t_to_1e-12": "59",
                "final_error_norm": "2.093e-16",
                "final_error_norm_ci95": "1e-16",
                "final_member_aoi": "63.86",
                "final_member_aoi_ci95": "0.09",
                "final_member_stale_fraction_75": "0.437",
                "final_member_stale_fraction_75_ci95": "0.002",
                "final_member_zero_participation_fraction": "0.312",
                "final_member_zero_participation_fraction_ci95": "0.003",
                "final_energy_efficiency": "825.7",
                "final_energy_efficiency_ci95": "6.0",
            },
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_cli_writes_markdown_latex_and_claim_bullets(self):
        tmpdir = self._clean_test_root("_test_export_paper_tables")
        try:
            comparison_csv = self._write_comparison_csv(tmpdir)

            with redirect_stdout(io.StringIO()):
                export_paper_tables.main(
                    [
                        "--comparison-csv",
                        str(comparison_csv),
                        "--output-dir",
                        str(tmpdir),
                    ]
                )

            markdown = tmpdir / "paper_results_table.md"
            latex = tmpdir / "paper_results_table.tex"
            claims = tmpdir / "paper_claim_bullets.md"
            self.assertTrue(markdown.exists())
            self.assertTrue(latex.exists())
            self.assertTrue(claims.exists())

            markdown_text = markdown.read_text(encoding="utf-8")
            self.assertIn("Quota baseline", markdown_text)
            self.assertIn("Collision-aware quota", markdown_text)
            self.assertIn("Semi-scheduled upper bound", markdown_text)
            self.assertIn("mean +/- ci95", markdown_text)
            self.assertIn("74.230 +/- 0.120", markdown_text)
            self.assertIn("+5.3%", markdown_text)

            latex_text = latex.read_text(encoding="utf-8")
            self.assertIn("\\begin{table}", latex_text)
            self.assertIn("pure-ALOHA", latex_text)
            self.assertIn("$\\pm$", latex_text)
            self.assertIn("\\%", latex_text)

            claim_text = claims.read_text(encoding="utf-8")
            self.assertIn("no tested ablation dominates the baseline", claim_text)
            self.assertIn("Coordinated upper-bound", claim_text)
            self.assertIn("outside the pure-ALOHA contribution", claim_text)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_explicit_missing_baseline_raises(self):
        rows = [
            {
                "run": "baseline",
                "claim_role": "main pure-ALOHA baseline",
            }
        ]

        with self.assertRaises(ValueError):
            export_paper_tables.select_baseline(
                rows,
                baseline_run="does_not_exist",
            )


if __name__ == "__main__":
    unittest.main()
