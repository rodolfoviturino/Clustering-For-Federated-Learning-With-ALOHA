import csv
import shutil
import unittest
from pathlib import Path

from experiments.plot_research_matrix import plot_research_matrix


class PlotResearchMatrixTests(unittest.TestCase):
    def test_plotter_creates_canonical_matrix_figures(self):
        output_dir = Path.cwd() / "Runs" / "_test_plot_research_matrix"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        try:
            csv_path = output_dir / "run_comparison_summary.csv"
            rows = [
                {
                    "run": "member_quota_k3000_w015_floor000_physical_r100",
                    "claim_role": "main pure-ALOHA baseline",
                    "optimized_d2d_access_mode": "member_quota_utility",
                    "t_to_1e-12": "100",
                    "final_member_stale_fraction_75": "0.618",
                    "final_member_zero_participation_fraction": "0.554",
                    "final_energy_efficiency": "653.8",
                },
                {
                    "run": "member_collision_quota_k3000_w015_t002_g2_min050_physical_r100",
                    "claim_role": "pure-ALOHA convergence/energy ablation",
                    "optimized_d2d_access_mode": "member_collision_aware_quota",
                    "t_to_1e-12": "88",
                    "final_member_stale_fraction_75": "0.614",
                    "final_member_zero_participation_fraction": "0.556",
                    "final_energy_efficiency": "688.7",
                },
                {
                    "run": "member_split_k3000_s5_w015_physical_r100",
                    "claim_role": "negative structural ablation",
                    "optimized_d2d_access_mode": "member_quota_utility",
                    "t_to_1e-12": "",
                    "final_member_stale_fraction_75": "0.891",
                    "final_member_zero_participation_fraction": "0.835",
                    "final_energy_efficiency": "103.9",
                },
                {
                    "run": "member_semischedule_k3000_s030_cc0010_physical_r100",
                    "claim_role": "coordinated upper-bound / future work",
                    "optimized_d2d_access_mode": "semi_scheduled_member_refresh",
                    "t_to_1e-12": "59",
                    "final_member_stale_fraction_75": "0.437",
                    "final_member_zero_participation_fraction": "0.312",
                    "final_energy_efficiency": "825.7",
                },
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            generated = plot_research_matrix(csv_path, formats=("png",))
            generated_names = {path.name for path in generated}

            self.assertIn("research_matrix_member_stale75.png", generated_names)
            self.assertIn(
                "research_matrix_member_zero_participation.png",
                generated_names,
            )
            self.assertIn("research_matrix_energy_efficiency.png", generated_names)
            self.assertIn("research_matrix_t_to_1e12.png", generated_names)
            self.assertIn(
                "research_matrix_stale75_energy_pareto.png",
                generated_names,
            )
            for path in generated:
                self.assertGreater(path.stat().st_size, 0)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
