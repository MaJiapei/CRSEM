"""Parameter configuration loader for CRSEM and RUSLE models.

This module provides functionality to load parameter defaults, bounds, and penalty
settings from YAML configuration files. If no config file is provided, the code
falls back to built-in hardcoded defaults.

Usage:
    # Load CRSEM config from file
    config = ParameterConfigLoader.load("config/parameter_config.crsem.yml")

    # Load default CRSEM config (built-in)
    config = ParameterConfigLoader.load_default("crsem")

    # Access values
    defaults = config.get_defaults()
    bounds = config.get_bounds()
    penalty_settings = config.get_penalty_settings()
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ParameterConfigLoader:
    """Loads and manages parameter configuration from YAML files."""

    # Built-in default configurations (fallback when no config file provided)
    _BUILTIN_CONFIGS = {
        "crsem": {
            "model_type": "crsem",
            "defaults": {
                'a_rain': 0.5, 'r_th': 1, 'a_melt': 0.1, 'm_th': 0, 'k_melt': 3.0,
                'alpha_K': 0.35, 'K_min_r': 0.7, 'K_max_r': 1.5,
                'alpha_C': 2, 'ic0': 0.5, 'k': 2.5, 'beta_sdr': 0.5,
                'c_base': 0.1, 'n_chan': 1.8, 'K_chan': 0.5
            },
            "bounds": {
                'a_rain': (0.5, 1), 'r_th': (1, 20), 'a_melt': (0.1, 1), 'm_th': (0, 10),
                'k_melt': (1.0, 5.0), 'alpha_K': (0.1, 0.8), 'K_min_r': (0.4, 1.0),
                'K_max_r': (1.0, 2), 'alpha_C': (1.0, 5.0), 'ic0': (0.1, 1.0),
                'k': (0.5, 4.0), 'beta_sdr': (0.3, 1.0),
                'c_base': (0.1, 20.0), 'n_chan': (1.0, 2.0), 'K_chan': (0.1, 1.0)
            },
            "penalties": {
                "channel_ratio": {"enabled": True, "lower_bound": -0.6, "upper_bound": 0.3},
                "annual_r_factor": {"enabled": True, "lower_normal": 100.0, "upper_normal": 200.0}
            }
        },
        "rusle": {
            "model_type": "rusle",
            "defaults": {
                'a_rain': 0.05, 'b_rain': 2.0, 'c_base': 0.1,
                'n_chan': 1.5, 'K_chan': 0.5, 'ic0': 0.5, 'k': 2.5
            },
            "bounds": {
                'a_rain': (1e-3, 1e-1), 'b_rain': (1.0, 3.0),
                'c_base': (0.1, 20.0), 'n_chan': (1.0, 2.0), 'K_chan': (0.1, 1.0),
                'ic0': (0.1, 1.0), 'k': (0.5, 4.0)
            },
            "penalties": {
                "channel_ratio": {"enabled": True, "lower_bound": -0.6, "upper_bound": 0.3}
            }
        }
    }

    def __init__(self, config_data: Dict[str, Any], source: str = "builtin"):
        """Initialize with configuration data.

        Args:
            config_data: Dictionary containing defaults, bounds, and penalties
            source: Source description (e.g., file path or 'builtin')
        """
        self.config_data = config_data
        self.source = source

    @classmethod
    def load(cls, config_path: str | Path) -> 'ParameterConfigLoader':
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            ParameterConfigLoader instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        return cls(config_data, source=str(config_path))

    @classmethod
    def load_default(cls, model_type: str = "crsem") -> 'ParameterConfigLoader':
        """Load built-in default configuration for a model type.

        Args:
            model_type: Model type ('crsem' or 'rusle')

        Returns:
            ParameterConfigLoader instance with built-in defaults

        Raises:
            ValueError: If model_type is not recognized
        """
        model_type = model_type.lower()
        if model_type not in cls._BUILTIN_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(cls._BUILTIN_CONFIGS.keys())}")

        return cls(cls._BUILTIN_CONFIGS[model_type], source=f"builtin:{model_type}")

    def get_defaults(self) -> Dict[str, float]:
        """Get default parameter values.

        Returns:
            Dictionary mapping parameter names to default values
        """
        return dict(self.config_data.get("defaults", {}))

    def get_bounds(self) -> Dict[str, tuple]:
        """Get parameter bounds.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        bounds = self.config_data.get("bounds", {})
        # Convert lists to tuples for consistency
        return {k: tuple(v) if isinstance(v, list) else v for k, v in bounds.items()}

    def get_penalty_settings(self, penalty_name: str) -> Optional[Dict[str, Any]]:
        """Get settings for a specific penalty.

        Args:
            penalty_name: Name of the penalty (e.g., 'channel_ratio', 'annual_r_factor')

        Returns:
            Dictionary with penalty settings, or None if penalty not configured
        """
        penalties = self.config_data.get("penalties", {})
        penalty_config = penalties.get(penalty_name)

        if penalty_config is None:
            return None

        # If only 'enabled' is specified, return default settings
        if penalty_config.get("enabled", True):
            return penalty_config
        return None

    def get_all_penalties(self) -> Dict[str, Dict[str, Any]]:
        """Get all penalty configurations.

        Returns:
            Dictionary mapping penalty names to their settings
        """
        penalties = self.config_data.get("penalties", {})
        return {
            name: config for name, config in penalties.items()
            if config.get("enabled", True)
        }

    def get_model_type(self) -> str:
        """Get the model type from configuration.

        Returns:
            Model type string
        """
        return self.config_data.get("model_type", "unknown")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config_data.get(key, default)
