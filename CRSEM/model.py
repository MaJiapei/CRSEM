"""Model data containers and model factory."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, Set, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import xarray as xr

NumericArray = Union[np.ndarray, List[float], Tuple[float, ...], Sequence[float]]


@dataclass(slots=True)
class ModelInputs:
    """Container for xarray-backed model inputs."""

    T: Optional[xr.DataArray] = None
    Pre: Optional[xr.DataArray] = None
    NDVI: Optional[xr.DataArray] = None
    LS: Optional[xr.DataArray] = None
    P_f: Optional[xr.DataArray] = None
    IC: Optional[xr.DataArray] = None
    K: Optional[xr.DataArray] = None

    def __getitem__(self, key: str) -> Optional[xr.DataArray]:
        return getattr(self, key)

    def as_dict(self) -> Dict[str, Optional[xr.DataArray]]:
        return {field_info.name: getattr(self, field_info.name) for field_info in fields(self)}


@dataclass(slots=True)
class ModelOutputs:
    """A standardized container for model outputs."""

    _ALLOWED_DIMS: ClassVar[Set[str]] = {"latitude", "longitude", "y", "x", "time", "member"}

    SSF_pred: Optional[xr.DataArray] = None
    A_channel: Optional[xr.DataArray] = None
    E_hillslope: Optional[xr.DataArray] = None
    E_hillslope_rain: Optional[xr.DataArray] = None
    E_hillslope_melt: Optional[xr.DataArray] = None
    R_rain: Optional[xr.DataArray] = None
    R_melt: Optional[xr.DataArray] = None
    K_factor: Optional[xr.DataArray] = None
    C_factor: Optional[xr.DataArray] = None
    SDR: Optional[xr.DataArray] = None

    def __post_init__(self) -> None:
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if isinstance(value, xr.DataArray):
                setattr(self, field_info.name, self._trim_dims(value, field_info.name))

    def __getitem__(self, key: str) -> Optional[xr.DataArray]:
        return getattr(self, key)

    def _trim_dims(self, data_array: xr.DataArray, var_name: str) -> xr.DataArray:
        trimmed = data_array.reset_coords(drop=True)
        drop_coords = [coord for coord in trimmed.coords if coord not in self._ALLOWED_DIMS]
        if drop_coords:
            trimmed = trimmed.drop_vars(drop_coords, errors="ignore")

        remove_dims = [dim for dim in trimmed.dims if dim not in self._ALLOWED_DIMS]
        if remove_dims:
            squeezable = [dim for dim in remove_dims if trimmed.sizes.get(dim, 0) == 1]
            if squeezable:
                trimmed = trimmed.squeeze(dim=squeezable, drop=True)
            remaining = [dim for dim in trimmed.dims if dim not in self._ALLOWED_DIMS]
            if remaining:
                raise ValueError(
                    f"Variable '{var_name}' contains unsupported dims {remaining}; "
                    f"allowed dims are {sorted(self._ALLOWED_DIMS)}."
                )
        return trimmed

    def list_not_none(self) -> List[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is not None]

    @property
    def ds(self) -> xr.Dataset:
        return xr.Dataset({name: getattr(self, name) for name in self.list_not_none()})

    def __repr__(self) -> str:
        not_none_attrs = self.list_not_none()
        if not not_none_attrs:
            return "ModelOutputs()"
        lines = ["ModelOutputs("]
        for attr_name in not_none_attrs:
            lines.append(f"  {attr_name}: {type(getattr(self, attr_name)).__name__}")
        lines.append(")")
        return "\n".join(lines)


class ModelRegistry:
    """Centralized management of model classes, parameter classes and their metadata."""

    _registry: Dict[str, Dict[str, object]] = {}

    @classmethod
    def register(
        cls,
        key: str,
        model_cls: Type["BaseModel"],
        param_cls: Type,
        defaults: object,
        bounds: Iterable[Tuple[float, float]],
        param_names: Sequence[str],
    ) -> None:
        key_lower = key.lower()
        if key_lower in cls._registry:
            raise ValueError(f"Model key {key} already exists")
        cls._registry[key_lower] = {
            "model_cls": model_cls,
            "param_cls": param_cls,
            "defaults": defaults,
            "bounds": tuple(bounds),
            "param_names": tuple(param_names),
        }

    @classmethod
    def get(cls, key: str) -> Dict[str, object]:
        try:
            return cls._registry[key.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown model type: {key}") from exc

    @classmethod
    def keys(cls) -> Iterable[str]:
        return cls._registry.keys()


class ModelFactory:
    """Unified entry point for model construction and parameter metadata."""

    # Config-based parameter overrides (set via use_config method)
    _config_overrides: Dict[str, Dict[str, object]] = {}

    @staticmethod
    def register_defaults() -> None:
        if ModelRegistry.keys():
            return

        from CRSEM._model_crsem import CRSEMModel
        from CRSEM._model_rusle import RUSLEModel
        from CRSEM.parameters import CRSEMParameters, RUSLEParameters

        # Check for config overrides
        crsem_defaults = CRSEMParameters.DEFAULT_PARAMS
        crsem_bounds = tuple(CRSEMParameters.PARAM_BOUNDS[name] for name in CRSEMModel.PARAM_NAMES)
        rusle_defaults = RUSLEParameters.DEFAULT_PARAMS
        rusle_bounds = tuple(RUSLEParameters.PARAM_BOUNDS[name] for name in RUSLEModel.PARAM_NAMES)

        ModelRegistry.register(
            "crsem",
            model_cls=CRSEMModel,
            param_cls=CRSEMParameters,
            defaults=crsem_defaults,
            bounds=crsem_bounds,
            param_names=CRSEMModel.PARAM_NAMES,
        )
        ModelRegistry.register(
            "hydrosedi",
            model_cls=CRSEMModel,
            param_cls=CRSEMParameters,
            defaults=crsem_defaults,
            bounds=crsem_bounds,
            param_names=CRSEMModel.PARAM_NAMES,
        )
        ModelRegistry.register(
            "rusle",
            model_cls=RUSLEModel,
            param_cls=RUSLEParameters,
            defaults=rusle_defaults,
            bounds=rusle_bounds,
            param_names=RUSLEModel.PARAM_NAMES,
        )

    @classmethod
    def use_config(cls, config_path: str | None = None, model_type: str | None = None) -> None:
        """Load parameter configuration from a YAML file.

        Args:
            config_path: Path to YAML config file. If None, uses built-in defaults.
            model_type: Model type ('crsem' or 'rusle'). If None, auto-detected from config.

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        from CRSEM.parameters import CRSEMParameters, RUSLEParameters
        from CRSEM.parameter_config import ParameterConfigLoader

        # Clear registry to allow re-registration with new config
        ModelRegistry._registry.clear()
        CRSEMParameters.CONFIG_LOADER = None
        RUSLEParameters.CONFIG_LOADER = None

        if config_path is None:
            # Use built-in defaults
            return

        # Load config from file
        config_loader = ParameterConfigLoader.load(config_path)

        # Apply config to parameter classes
        if model_type is None:
            model_type = config_loader.get_model_type()

        model_type = model_type.lower()
        if model_type == "crsem":
            CRSEMParameters.set_config_loader(config_loader)
        elif model_type == "rusle":
            RUSLEParameters.set_config_loader(config_loader)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Re-register with updated defaults and bounds
        cls.register_defaults()

    @classmethod
    def use_builtin_config(cls, model_type: str = "crsem") -> None:
        """Use built-in default configuration for a model type.

        Args:
            model_type: Model type ('crsem' or 'rusle')
        """
        from CRSEM.parameters import CRSEMParameters, RUSLEParameters
        from CRSEM.parameter_config import ParameterConfigLoader

        # Clear registry
        ModelRegistry._registry.clear()
        CRSEMParameters.CONFIG_LOADER = None
        RUSLEParameters.CONFIG_LOADER = None

        # Load built-in config
        config_loader = ParameterConfigLoader.load_default(model_type)

        # Apply config to parameter classes
        model_type = model_type.lower()
        if model_type == "crsem":
            CRSEMParameters.set_config_loader(config_loader)
        elif model_type == "rusle":
            RUSLEParameters.set_config_loader(config_loader)

        # Re-register with updated defaults and bounds
        cls.register_defaults()

    @classmethod
    def coerce_parameter_batch(cls, model_type: str, params: Union[object, NumericArray]):
        from CRSEM.contracts import ParameterBatch

        cls.register_defaults()
        meta = ModelRegistry.get(model_type)
        param_cls = meta["param_cls"]
        param_names = tuple(meta["param_names"])

        if isinstance(params, ParameterBatch):
            return params

        if isinstance(params, param_cls):
            return ParameterBatch(values=params.to_array(), param_names=param_names)
        if isinstance(params, dict):
            return ParameterBatch(values=param_cls(**params).to_array(), param_names=param_names)
        if isinstance(params, (list, tuple, np.ndarray)):
            arr = np.asarray(params, dtype=float)
            return ParameterBatch(values=arr, param_names=param_names)
        raise TypeError(f"Unsupported parameter type: {type(params)}")

    @classmethod
    def create_base_model(cls, model_type: str, params: Union[object, NumericArray]):
        cls.register_defaults()
        meta = ModelRegistry.get(model_type)
        model_cls = meta["model_cls"]
        param_cls = meta["param_cls"]
        batch = cls.coerce_parameter_batch(model_type, params)
        param_obj = param_cls.from_array(np.asarray(batch.values[0], dtype=float))
        return model_cls(param_obj)

    @classmethod
    def create_execution(cls, model_type: str, params: Union[object, NumericArray]):
        batch = cls.coerce_parameter_batch(model_type, params)
        model = cls.create_base_model(model_type, batch)
        return model, batch

    @classmethod
    def create_model(cls, model_type: str, params: Union[object, NumericArray]):
        batch = cls.coerce_parameter_batch(model_type, params)
        if batch.n_members != 1:
            raise TypeError(
                "ModelFactory.create_model only accepts single-parameter inputs. "
                "Use ModelFactory.create_execution(...) or run_parameter_batch(...) for multi-member execution."
            )

        return cls.create_base_model(model_type, batch)

    @classmethod
    def get_parameter_template(cls, model_type: str) -> Dict[str, float]:
        cls.register_defaults()
        meta = ModelRegistry.get(model_type)
        defaults = meta["defaults"]
        if isinstance(defaults, dict):
            return dict(defaults)
        if hasattr(defaults, "to_dict"):
            return defaults.to_dict()
        raise ValueError("Default parameters cannot be converted to dictionary")

    @classmethod
    def get_parameter_info(cls, model_type: str):
        cls.register_defaults()
        meta = ModelRegistry.get(model_type)
        return meta["param_names"], meta["bounds"]

    @classmethod
    def get_parameter_class(cls, model_type: str):
        cls.register_defaults()
        meta = ModelRegistry.get(model_type)
        return meta["param_cls"]



