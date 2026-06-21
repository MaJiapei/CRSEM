import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from .calibrator import Calibrator
from .contracts import ParameterBatch
from .driver import BasinDriver
from .parameter_config import ParameterConfigLoader
from .model import ModelFactory


def save_calibration_results(params_obj, metrics, save_dir=None, save_path=None):
    """Save calibration results to a directory or JSON file path.

    Args:
        params_obj: ParameterBatch to save
        metrics: Calibration metrics dictionary
        save_dir: Output directory. Writes ``params.json`` into this directory.
        save_path: Output path. A ``.json`` path writes to that file directly;
            any other path is treated as a directory and writes ``params.json`` there.

    Returns:
        Path to the saved params.json file
    """
    if not isinstance(params_obj, ParameterBatch):
        raise TypeError("save_calibration_results expects a ParameterBatch.")

    if save_dir is not None and save_path is not None:
        raise ValueError("Use either save_dir or save_path, not both.")

    if save_path is not None:
        target = Path(save_path)
    else:
        target = Path("calibration_results") if save_dir is None else Path(save_dir)

    if target.suffix.lower() == ".json":
        params_file = target
        params_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
        params_file = target / "params.json"

    params_obj.to_file(params_file, metrics=metrics)
    print(f"\nCalibration results saved to: {params_file}")
    return params_file


def refine_parameters(
    driver: BasinDriver,
    model_type="crsem",
    maxiter=40,
    optimizer="differential_evolution",
    plot_progress=True,
    ensemble_para: bool = False,
    objective_method="nse",
    config_path: Optional[str | Path] = None,
    penalty_settings: Optional[Dict[str, Any]] = None,
    calibration_output_mode: str = "full",
    **kwargs,
):
    """Refine model parameters through calibration.

    Args:
        driver: BasinDriver with model inputs and observations
        model_type: Model type ('crsem' or 'rusle')
        maxiter: Maximum number of optimization iterations
        optimizer: Optimization algorithm ('differential_evolution')
        plot_progress: Whether to plot calibration progress
        ensemble_para: Whether to use ensemble parameter selection
        objective_method: Objective function ('nse', 'kge', 'rmse', 'mae', 'r2')
        config_path: Optional path to YAML parameter config file
        penalty_settings: Optional dictionary with penalty settings overrides
        calibration_output_mode: Output detail during calibration ('full' or 'compact')
        **kwargs: Additional arguments passed to Calibrator

    Returns:
        Tuple of (ParameterBatch, metrics dict)
    """
    # Load config if provided
    if config_path is not None:
        ModelFactory.use_config(config_path, model_type)

    calibrator = Calibrator(
        driver=driver,
        model_type=model_type,
        plot_progress=plot_progress,
        ensemble_para=ensemble_para,
        objective_method=objective_method,
        penalty_settings=penalty_settings,
        calibration_output_mode=calibration_output_mode,
        **kwargs,
    )
    return calibrator.run(optimizer=optimizer, maxiter=maxiter)

