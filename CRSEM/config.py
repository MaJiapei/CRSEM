from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataPaths:
    """Filesystem paths required by CRSEM loaders.

    All fields are optional so different workflows can supply only the paths
    they actually need. Callers should validate required fields at the loader
    boundary instead of relying on module-level constants.
    """

    # Original data source paths (legacy mode)
    meteo_file: Path | None = None
    ndvi_file: Path | None = None
    mask_tp_file: Path | None = None
    mask_sryr_file: Path | None = None
    mask_tth_file: Path | None = None
    ls_file: Path | None = None
    k_file: Path | None = None
    ic_file: Path | None = None
    river_obs_dir: Path | None = None
    river_excel_file: Path | None = None
    cjy_basin_shp: Path | None = None
    tth_basin_shp: Path | None = None
    era5_monthly_file: Path | None = None
    cru_pre_file: Path | None = None
    cru_tmp_file: Path | None = None
    terraclimate_ppt_glob: str | None = None
    terraclimate_tmean_zarr: Path | None = None
    terraclimate_tmin_glob: str | None = None
    terraclimate_tmax_glob: str | None = None
    terraclimate_ppt_zarr: Path | None = None
    landsat_turbidity_file: Path | None = None
    meteo_merged_file: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataPaths":
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                normalized[key] = None
            elif key.endswith("_glob"):
                normalized[key] = str(value)
            else:
                normalized[key] = Path(value)
        return cls(**normalized)

    def require(self, field_name: str) -> Path:
        value = getattr(self, field_name)
        if value is None:
            raise ValueError(f"Missing required data path: {field_name}")
        return value


@dataclass(slots=True)
class PreparedDatasetPaths:
    """Paths to pre-prepared NetCDF files for direct loading.

    This is the new recommended way to specify inputs, replacing the
    scattered raw data files with prepared NetCDF datasets.

    Files should be created using scripts/prepare_basin_drivers.py.
    """

    static_nc: Path | None = None
    """Path to static.nc containing K, LS, IC, P_f, mask."""

    dynamic_nc: Path | None = None
    """Path to dynamic.nc containing T, Pre, NDVI with time dimension."""

    observations_nc: Path | None = None
    """Path to observations.nc containing Q and SSF (optional)."""

    basin_name: str | None = None
    """Basin name for metadata."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedDatasetPaths":
        """Create from dictionary, converting path strings to Path objects."""
        return cls(
            static_nc=Path(data["static_nc"]) if data.get("static_nc") else None,
            dynamic_nc=Path(data["dynamic_nc"]) if data.get("dynamic_nc") else None,
            observations_nc=Path(data["observations_nc"]) if data.get("observations_nc") else None,
            basin_name=data.get("basin_name"),
        )

    def is_complete(self) -> bool:
        """Check if all required paths are specified."""
        return self.static_nc is not None and self.dynamic_nc is not None

    def require(self, field_name: str) -> Path:
        """Get a path, raising ValueError if not set."""
        value = getattr(self, field_name)
        if value is None:
            raise ValueError(f"Missing required prepared dataset path: {field_name}")
        return value


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime configuration passed explicitly into loaders and drivers."""

    data_paths: DataPaths
    datasets: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    prepared_dataset: PreparedDatasetPaths | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        paths_payload = data.get("data_paths", {})
        datasets = data.get("datasets", {})
        extras = data.get("extras", {})

        # Load prepared dataset paths if present
        prepared_payload = data.get("prepared_dataset", {})
        prepared_dataset = (
            PreparedDatasetPaths.from_dict(prepared_payload)
            if prepared_payload
            else None
        )

        return cls(
            data_paths=DataPaths.from_dict(paths_payload),
            datasets=dict(datasets),
            extras=dict(extras),
            prepared_dataset=prepared_dataset,
        )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load runtime configuration from JSON or YAML.

    YAML support is optional and only used when the dependency is available.
    """

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8"))
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise RuntimeError(
                "YAML config requires PyYAML. Install it or use a JSON config file."
            ) from exc
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Runtime config must deserialize to a mapping")
    return RuntimeConfig.from_dict(data)
