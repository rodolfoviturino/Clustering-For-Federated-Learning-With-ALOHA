import csv
import tempfile
import unittest
from pathlib import Path

from experiments.plot_gpu_sweep import plot_sweep_csv


class PlotGpuSweepTests(unittest.TestCase):
    def test_plotter_creates_aoi_figures_when_columns_exist(self):
        """AoI plotting should be schema-driven, just like energy plots.

        The runner writes many optional metric families.  The plotter should
        not need a separate flag for each enhanced run; if the CSV contains
        `<scenario>_aoi_*` and `<scenario>_peak_aoi_*` columns, it should emit
        the corresponding figures when regenerating plots from an existing run.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            csv_path = output_dir / "results.csv"
            rows = [
                {
                    "t": 1,
                    "clustering_mode": "geometric",
                    "clusterized_devices_rate_mean": 80.0,
                    "clusterized_devices_rate_ci95": 0.0,
                    "polling_error_norm_mean": 1.0,
                    "polling_error_norm_ci95": 0.0,
                    "polling_aoi_mean": 1.2,
                    "polling_aoi_ci95": 0.1,
                    "polling_peak_aoi_mean": 2.0,
                    "polling_peak_aoi_ci95": 0.0,
                    "polling_p75_aoi_mean": 1.6,
                    "polling_p75_aoi_ci95": 0.0,
                    "polling_p90_aoi_mean": 1.7,
                    "polling_p90_aoi_ci95": 0.0,
                    "polling_p95_aoi_mean": 1.8,
                    "polling_p95_aoi_ci95": 0.0,
                    "polling_stale_fraction_50_mean": 0.25,
                    "polling_stale_fraction_50_ci95": 0.0,
                    "polling_stale_fraction_75_mean": 0.10,
                    "polling_stale_fraction_75_ci95": 0.0,
                    "polling_stale_fraction_100_mean": 0.0,
                    "polling_stale_fraction_100_ci95": 0.0,
                },
                {
                    "t": 2,
                    "clustering_mode": "geometric",
                    "clusterized_devices_rate_mean": 80.0,
                    "clusterized_devices_rate_ci95": 0.0,
                    "polling_error_norm_mean": 0.8,
                    "polling_error_norm_ci95": 0.0,
                    "polling_aoi_mean": 1.5,
                    "polling_aoi_ci95": 0.1,
                    "polling_peak_aoi_mean": 3.0,
                    "polling_peak_aoi_ci95": 0.0,
                    "polling_p75_aoi_mean": 2.0,
                    "polling_p75_aoi_ci95": 0.0,
                    "polling_p90_aoi_mean": 2.2,
                    "polling_p90_aoi_ci95": 0.0,
                    "polling_p95_aoi_mean": 2.5,
                    "polling_p95_aoi_ci95": 0.0,
                    "polling_stale_fraction_50_mean": 0.30,
                    "polling_stale_fraction_50_ci95": 0.0,
                    "polling_stale_fraction_75_mean": 0.15,
                    "polling_stale_fraction_75_ci95": 0.0,
                    "polling_stale_fraction_100_mean": 0.0,
                    "polling_stale_fraction_100_ci95": 0.0,
                },
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            generated = plot_sweep_csv(csv_path, formats=("png",))
            generated_names = {path.name for path in generated}

            self.assertIn("results_aoi.png", generated_names)
            self.assertIn("results_peak_aoi.png", generated_names)
            self.assertIn("results_p75_aoi.png", generated_names)
            self.assertIn("results_p90_aoi.png", generated_names)
            self.assertIn("results_p95_aoi.png", generated_names)
            self.assertIn("results_stale_fraction_50.png", generated_names)
            self.assertIn("results_stale_fraction_75.png", generated_names)
            self.assertIn("results_stale_fraction_100.png", generated_names)


if __name__ == "__main__":
    unittest.main()
