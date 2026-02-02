"""ReLoop: Behavioral Verification for LLM-Generated Optimization Code"""

from .verification import (
    ReLoopVerifier,
    VerificationReport,
    LayerResult,
    Severity,
    Complexity,
    verify_code,
)

from .executor import CodeExecutor

from .param_utils import (
    ParameterRole,
    extract_numeric_params,
    infer_param_role,
    get_expected_direction,
    perturb_param,
    set_param,
    should_skip_param,
    get_param_value,
)

from .generation import CodeGenerator, GenerationResult

from .repair import CodeRepairer

from .pipeline import ReLoopPipeline, PipelineResult, run_reloop

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
    "verify_code",
    # Executor
    "CodeExecutor",
    # Parameter utilities
    "ParameterRole",
    "extract_numeric_params",
    "infer_param_role",
    "get_expected_direction",
    "perturb_param",
    "set_param",
    "should_skip_param",
    "get_param_value",
    # Generation
    "CodeGenerator",
    "GenerationResult",
    # Repair
    "CodeRepairer",
    # Pipeline
    "ReLoopPipeline",
    "PipelineResult",
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
