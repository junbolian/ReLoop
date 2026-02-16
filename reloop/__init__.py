"""ReLoop: Behavioral Verification for LLM-Generated Optimization Code"""

from .verification import (
    ReLoopVerifier,
    VerificationReport,
    LayerResult,
    Severity,
    Complexity,
    Diagnostic,
    verify_code,
    layer_results_to_diagnostics,
)

from .executor import CodeExecutor

from .param_utils import (
    extract_numeric_params,
    perturb_param,
    set_param,
    should_skip_param,
    get_param_value,
)

from .perturbation import (
    detect_perturbation_mode,
    extract_perturbable_params,
    perturb_code,
    get_source_code_param_names,
    run_perturbation,
)

from .generation import CodeGenerator, GenerationResult

from .repair import CodeRepairer, RepairResult

from .pipeline import ReLoopPipeline, PipelineResult, RepairContext, run_reloop

from .data_extraction import DataExtractor, extract_data_from_question

from .experiment_runner import (
    ExperimentRunner,
    ExperimentRecord,
    ExperimentSummary,
    run_experiment,
)

__version__ = "1.0.0"
__all__ = [
    # Verification
    "ReLoopVerifier",
    "VerificationReport",
    "LayerResult",
    "Severity",
    "Complexity",
    "Diagnostic",
    "verify_code",
    "layer_results_to_diagnostics",
    # Executor
    "CodeExecutor",
    # Parameter utilities
    "extract_numeric_params",
    "perturb_param",
    "set_param",
    "should_skip_param",
    "get_param_value",
    # Perturbation
    "detect_perturbation_mode",
    "extract_perturbable_params",
    "perturb_code",
    "get_source_code_param_names",
    "run_perturbation",
    # Generation
    "CodeGenerator",
    "GenerationResult",
    # Repair
    "CodeRepairer",
    "RepairResult",
    # Pipeline
    "ReLoopPipeline",
    "PipelineResult",
    "RepairContext",
    "run_reloop",
    # Data Extraction
    "DataExtractor",
    "extract_data_from_question",
    # Experiment Runner
    "ExperimentRunner",
    "ExperimentRecord",
    "ExperimentSummary",
    "run_experiment",
]
