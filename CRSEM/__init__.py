"""CRSEM public package exports."""

from ._model_base import BaseModel
from ._model_crsem import CRSEMModel
from ._model_rusle import RUSLEModel
from .batch_runner import run_parameter_batch
from .calibration_api import refine_parameters, save_calibration_results
from .calibration_evaluation import (
    AnnualRFactorPenalty,
    ChannelRatioPenalty,
    DiagnosticsExtractor,
    KGEObjective,
    MAEObjective,
    MetricsExtractor,
    NSEObjective,
    ObjectiveEvaluator,
    R2Objective,
    RMSEObjective,
    create_penalties,
    register_model_penalties,
    register_objective,
    register_penalty,
)
from .calibration_optimizer import DifferentialEvolutionOptimizer, SamplingOptimizer, create_optimizer, register_optimizer
from .calibration_reporting import CalibrationReporter, CalibrationTracker
from .calibration_result import CalibrationResult
from .calibration_runner import CalibrationModelRunner
from .calibrator import Calibrator
from .config import DataPaths, RuntimeConfig, load_runtime_config
from .contracts import BatchRunResult, ParameterBatch, PreparedInputs, RunContext
from .driver import BasinDriver
from .ensemble_selector import AICSelector, BestOnlySelector, GLUESelector, create_selector, register_selector
from .model import ModelFactory, ModelInputs, ModelOutputs
from .parameters import CRSEMParameters, RUSLEParameters
from .preparation import prepare_inputs
from .result_aggregator import ResultAggregator
# Keep sensitivity tools lazy/optional. Importing SHAP can crash some Windows
# scipy builds at native-library load time, which would otherwise block core
# CRSEM model runs.
analyze_climate_ndvi_sensitivity = None
run_oat_sensitivity_analysis = None
validate_point_mode = None

__all__ = [
    "AnnualRFactorPenalty",
    "AICSelector",
    "BaseModel",
    "BasinDriver",
    "BatchRunResult",
    "BestOnlySelector",
    "CRSEMModel",
    "CRSEMParameters",
    "Calibrator",
    "CalibrationModelRunner",
    "CalibrationReporter",
    "CalibrationResult",
    "CalibrationTracker",
    "ChannelRatioPenalty",
    "DataPaths",
    "DiagnosticsExtractor",
    "DifferentialEvolutionOptimizer",
    "GLUESelector",
    "KGEObjective",
    "MAEObjective",
    "MetricsExtractor",
    "ModelFactory",
    "ModelInputs",
    "ModelOutputs",
    "NSEObjective",
    "ObjectiveEvaluator",
    "ParameterBatch",
    "PreparedInputs",
    "R2Objective",
    "RMSEObjective",
    "RUSLEModel",
    "RUSLEParameters",
    "ResultAggregator",
    "RunContext",
    "RuntimeConfig",
    "analyze_climate_ndvi_sensitivity",
    "create_optimizer",
    "create_penalties",
    "create_selector",
    "load_runtime_config",
    "prepare_inputs",
    "refine_parameters",
    "register_optimizer",
    "register_model_penalties",
    "register_objective",
    "register_penalty",
    "register_selector",
    "run_oat_sensitivity_analysis",
    "run_parameter_batch",
    "save_calibration_results",
    "SamplingOptimizer",
    "validate_point_mode",
]

