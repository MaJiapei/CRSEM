"""Smoke test for CRSEM with the bundled real-case dataset."""
from __future__ import annotations

from pathlib import Path
import unittest

from CRSEM.batch_runner import run_parameter_batch
from CRSEM.driver import BasinDriver
from CRSEM.parameters import CRSEMParameters


class RealDataSmokeTest(unittest.TestCase):
    def test_real_data_hillslope_batch_runs(self):
        """Test model runs with repository-bundled prepared NC data files."""
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "example" / "zhimenda_sample"
        static_nc = output_dir / "drivers" / "static.nc"
        dynamic_nc = output_dir / "drivers" / "dynamic.nc"
        observations_nc = output_dir / "drivers" / "observations.nc"

        if not all(f.exists() for f in (static_nc, dynamic_nc, observations_nc)):
            self.skipTest(f"Bundled example dataset not found under: {output_dir / 'drivers'}")

        # Load driver from NC files (modern API)
        driver = BasinDriver.from_nc_files(
            static_nc=static_nc,
            dynamic_nc=dynamic_nc,
            observations_nc=observations_nc,
            station_name="zhimenda",
        ).collapse_ndvi_members()

        # Run model using batch runner
        params = CRSEMParameters.from_default().to_array()
        result = run_parameter_batch(
            model_type="crsem",
            source=driver,
            params=params,
            run_method="run_hillslope",
        )

        ds = result.to_dataset()

        self.assertIn("member", ds.sizes)
        self.assertEqual(ds.sizes["member"], 1)
        self.assertIn("E_hillslope", ds.data_vars)
        self.assertGreater(ds.sizes.get("time", 0), 0)


if __name__ == "__main__":
    unittest.main()
