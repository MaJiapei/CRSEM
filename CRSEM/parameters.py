import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, ClassVar
from pathlib import Path
from abc import ABC

from .parameter_config import ParameterConfigLoader

"""
Model Parameters Management Module

This module provides a unified parameter management system for different models
using a base class pattern.

Architecture:
  - BaseParameters: Abstract base class with common functionality
    └── CRSEMParameters: Cold Region Soil Erosion Model parameters
    └── RUSLEParameters: RUSLE erosion model parameters

Key Features:
  1. Common interface for all parameter types
  2. Automatic validation against parameter bounds
  3. Multiple input/output formats: dict, JSON file, numpy array
  4. Extensible through method inheritance and overrides
  5. Built-in documentation with describe() method

Usage Examples:
  # Load from default
  params = CRSEMParameters.from_default()
  
  # Load from file
  params = CRSEMParameters.from_file('path/to/params.json')
  
  # Load from numpy array
  params = CRSEMParameters.from_array(param_array)
  
  # Load from dictionary
  params = CRSEMParameters.from_dict(param_dict)
  
  # Save to file
  params.to_file('output/params.json')
  
  # Get parameter array for model
  param_array = params.to_array()
  
  # Validate parameters
  params.validate()  # Raises ValueError if out of bounds
  
  # Print description
  print(params.describe())
"""


@dataclass
class BaseParameters(ABC):
    """Base class for model parameters with common functionality.

    Instances represent a single parameter set. Multi-member execution is handled by
    `ParameterBatch`, not by the parameter classes themselves.

    Class Attributes:
        DEFAULT_PARAMS: Default parameter values (can be overridden by config)
        PARAM_BOUNDS: Parameter bounds for optimization (can be overridden by config)
        CONFIG_LOADER: Optional ParameterConfigLoader instance for config-based initialization
    """

    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {}
    PARAM_BOUNDS: ClassVar[Dict[str, tuple]] = {}
    CONFIG_LOADER: ClassVar[Optional[ParameterConfigLoader]] = None

    @classmethod
    def set_config_loader(cls, config_loader: ParameterConfigLoader) -> None:
        """Set configuration loader to override default parameters and bounds.

        Args:
            config_loader: ParameterConfigLoader instance with custom configuration
        """
        cls.CONFIG_LOADER = config_loader
        # Override DEFAULT_PARAMS and PARAM_BOUNDS from config
        cls.DEFAULT_PARAMS = config_loader.get_defaults()
        cls.PARAM_BOUNDS = config_loader.get_bounds()

    @classmethod
    def clear_config_loader(cls) -> None:
        """Clear configuration loader and revert to built-in defaults."""
        cls.CONFIG_LOADER = None
        # Note: DEFAULT_PARAMS and PARAM_BOUNDS need to be restored by subclass
        # This is typically handled by reloading the module or restarting Python

    def __setattr__(self, name, value):
        """Ensure parameter attributes are stored as float32 where applicable."""
        # Only enforce float32 for fields defined in DEFAULT_PARAMS
        default_params = getattr(type(self), "DEFAULT_PARAMS", {})
        if name in default_params and value is not None:
            # Convert scalar values to float32
            if isinstance(value, (int, float, np.floating)):
                value = np.float32(value)
            # Convert numpy arrays to float32
            elif isinstance(value, np.ndarray):
                value = value.astype(np.float32, copy=False)
        super().__setattr__(name, value)

    @classmethod
    def from_default(cls) -> 'BaseParameters':
        """Create instance using default parameters"""
        return cls(**cls.DEFAULT_PARAMS)
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'BaseParameters':
        """Load parameters from JSON file, returning parameters and optional metrics"""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = None
        # New format check
        if 'parameters' in data and 'metrics' in data:
            params = data['parameters']
            metrics = data['metrics']
        else:
            # Legacy format support
            params = data

        # Verify all required parameters exist
        missing_params = set(cls.DEFAULT_PARAMS.keys()) - set(params.keys())
        if missing_params:
            raise ValueError(f"Parameter file missing the following parameters: {missing_params}")
        
        instance = cls(**params)
        instance.metrics = metrics
        return instance
    
    @classmethod
    def from_array(cls, arr: Union[List[float], np.ndarray]) -> 'BaseParameters':
        """Create instance from numpy array"""
        arr = np.asarray(arr, dtype=float)
        if len(arr) != len(cls.DEFAULT_PARAMS):
            raise ValueError(f"Array length must be {len(cls.DEFAULT_PARAMS)}")
        
        param_dict = dict(zip(cls.DEFAULT_PARAMS.keys(), arr))
        return cls(**param_dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BaseParameters':
        """Create instance from dictionary"""
        missing = set(cls.DEFAULT_PARAMS.keys()) - set(data.keys())
        if missing:
            raise ValueError(f"Parameter dictionary missing the following parameters: {missing}")
        return cls(**{name: data[name] for name in cls.DEFAULT_PARAMS.keys()})
    
    def to_file(self, filepath: Union[str, Path], metrics: Optional[Dict] = None) -> None:
        """Save parameters and optionally metrics to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data payload
        if metrics is not None:
            # New format: dict with parameters and metrics
            payload = {
                'parameters': self.to_dict(),
                'metrics': _convert_numpy_types(metrics)
            }
        else:
            # Old format: just parameters dict
            payload = self.to_dict()
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array of shape (n_params,)"""
        return np.array([getattr(self, name) for name in self.DEFAULT_PARAMS.keys()])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary mapping parameter names to values"""
        result = asdict(self)
        # Convert numpy types to native Python types for JSON serialization
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in result.items()}
    
    def validate(self) -> None:
        """Validate if parameters are within reasonable ranges"""
        for param_name, value in asdict(self).items():
            if param_name in self.PARAM_BOUNDS:
                lower, upper = self.PARAM_BOUNDS[param_name]
                if not lower <= value <= upper:
                    raise ValueError(
                        f"Parameter {param_name} = {value} out of range [{lower}, {upper}]"
                    )
    
    def describe(self) -> str:
        """Return parameter description information as a formatted table."""
        desc = ["Parameter Configuration:"]

        # Get categories and create a param -> category mapping
        categories = getattr(self, '_get_categories', lambda: {})()
        param_to_category = {param: cat for cat, params in categories.items() for param in params}

        # Prepare data for the table
        params = asdict(self)
        table_data = []
        for name, value in params.items():
            bounds = self.PARAM_BOUNDS.get(name, ('N/A', 'N/A'))
            
            table_data.append({
                'Category': param_to_category.get(name, 'General'),
                'Parameter': name,
                'Value': value,
                'Bounds': str(bounds)
            })

        df = pd.DataFrame(table_data)
        df.set_index(['Category', 'Parameter'], inplace=True)

        # Format float columns
        df['Value'] = df['Value'].map('{:.6f}'.format)

        # Use pandas option to display all rows
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            desc.append(df.to_string())
        
        return "\n".join(desc)

@dataclass
class CRSEMParameters(BaseParameters):
    """Cold Region Soil Erosion Model (CRSEM) parameter class"""
    
    # R-factor parameters
    a_rain: float  # Rainfall R-factor coefficient a
    r_th: float  # Rainfall R-factor intercept b
    a_melt: float  # Snowmelt R-factor coefficient a
    m_th: float  # Snowmelt R-factor intercept b
    k_melt: float  # Snowmelt coefficient k
    
    # K-factor parameters
    alpha_K: float  # Freeze-thaw effect coefficient
    K_min_r: float  # Minimum K-factor ratio
    K_max_r: float  # Maximum K-factor ratio
    
    # C-factor parameters
    alpha_C: float  # Vegetation cover effect coefficient
    
    # SDR parameters
    ic0: float     # Critical slope length
    k: float       # SDR curve shape parameter
    beta_sdr: float  # SDR sensitivity to precipitation (rainfall + snowmelt)
    
    # Channel erosion parameters (single-mechanism power-law model)
    c_base: float  # Base channel erosion coefficient
    n_chan: float  # Flow exponent for base erosion
    K_chan: float  # Channel erodibility coefficient (0-1)

    # Default parameter values
    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {
        'a_rain': 0.5,
        'r_th': 1,
        'a_melt': 0.1,
        'm_th': 0,
        'k_melt': 3.0,
        'alpha_K': 0.35,
        'K_min_r': 0.7,
        'K_max_r': 1.5,
        'alpha_C': 2,
        'ic0': 0.5,
        'k': 2.5,
        'beta_sdr': 0.5,
        'c_base': 0.1,
        'n_chan': 1.8,
        'K_chan': 0.5
    }

    # Parameter bounds
    PARAM_BOUNDS: ClassVar[Dict[str, tuple]] = {
        'a_rain': (0.5, 1),       # Increased lower bound to avoid too small R-factor
        'r_th': (1, 20),        # Intercept for linear R-factor model
        'a_melt': (0.1, 1),       # Increased lower bound to avoid too small R-factor
        'm_th': (0, 10),        # Intercept for linear R-factor model
        'k_melt': (1.0, 5.0),
        'alpha_K': (0.1, 0.8),       # Relaxed upper bound
        'K_min_r': (0.4, 1.0),
        'K_max_r': (1.0, 2),
        'alpha_C': (1.0, 5.0),       # Relaxed upper bound for stronger vegetation protection
        'ic0': (0.1, 1.0),           # Relaxed upper bound
        'k': (0.5, 4.0),             # Relaxed lower bound
        'beta_sdr': (0.3, 1.0),      # SDR sensitivity to precipitation
        
        # Simplified to Single Mechanism (Power Law)
        'c_base': (0.1, 20.0),       # Significantly increased upper bound
        'n_chan': (1.0, 2.0),        # Lowered lower bound to allow near-linear response
        'K_chan': (0.1, 1.0)         # Standard range
    }
    
    def _get_categories(self) -> Dict[str, List[str]]:
        """Get parameter categories for description"""
        return {
            "R-factor Parameters": ['a_rain', 'r_th', 'a_melt', 'm_th', 'k_melt'],
            "K-factor Parameters": ['alpha_K', 'K_min_r', 'K_max_r'],
            "C-factor Parameters": ['alpha_C'],
            "SDR Parameters": ['ic0', 'k', 'beta_sdr'],
            "Channel Erosion Parameters": ['c_base', 'n_chan', 'K_chan']
        }


@dataclass
class RUSLEParameters(BaseParameters):
    """RUSLE model parameter class"""

    a_rain: float
    b_rain: float
    c_base: float
    n_chan: float
    K_chan: float
    ic0: float
    k: float

    DEFAULT_PARAMS: ClassVar[Dict[str, float]] = {
        'a_rain': 0.05,
        'b_rain': 2.0,
        'c_base': 0.1,
        'n_chan': 1.5,
        'K_chan': 0.5,
        'ic0': 0.5,
        'k': 2.5,
    }

    PARAM_BOUNDS: ClassVar[Dict[str, tuple]] = {
        'a_rain': (1e-3, 1e-1),
        'b_rain': (1.0, 3.0),
        'c_base': (0.1, 20.0),       # Significantly increased upper bound
        'n_chan': (1.0, 2.0),        # Lowered lower bound to allow near-linear response
        'K_chan': (0.1, 1.0),
        'ic0': (0.1, 1.0),           # Relaxed upper bound
        'k': (0.5, 4.0),             # Relaxed lower bound
    }
    
    def _get_categories(self) -> Dict[str, List[str]]:
        """Get parameter categories for description"""
        return {
            "R-factor Parameters": ['a_rain', 'b_rain'],
            "SDR Parameters": ['ic0', 'k'],
            "Channel Erosion Parameters": ['c_base', 'n_chan', 'K_chan']
        }

def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization.
    
    This is a module-level helper function used by parameter serialization helpers.
    """
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj
if __name__ == "__main__":
    # Simple tests to ensure parameters are stored as float32
    print("Testing CRSEMParameters float32 enforcement...")
    
    hydro = CRSEMParameters.from_default()
    for name in CRSEMParameters.DEFAULT_PARAMS.keys():
        value = getattr(hydro, name)
        assert isinstance(value, np.float32), f"{name} is not float32: {type(value)}"
    # Reassign a scalar and check type
    hydro.a_rain = 1.23
    assert isinstance(hydro.a_rain, np.float32), f"a_rain reassignment is not float32: {type(hydro.a_rain)}"


    print("Testing RUSLEParameters float32 enforcement...")
    rusle = RUSLEParameters.from_default()
    for name in RUSLEParameters.DEFAULT_PARAMS.keys():
        value = getattr(rusle, name)
        assert isinstance(value, np.float32), f"{name} is not float32: {type(value)}"
    rusle.a_rain = 0.01
    assert isinstance(rusle.a_rain, np.float32), f"a_rain reassignment is not float32: {type(rusle.a_rain)}"

    print("All float32 tests passed.")



