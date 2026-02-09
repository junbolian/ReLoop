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
    l2_verify_results_to_diagnostics,
    # Backward compatibility alias
    l4_verify_results_to_diagnostics,
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

from .l2_direction import (
    L2DirectionVerifier,
    L2VerifyResult,
    L2RepairDecision,
    should_exit_l2_loop,
    # Backward compatibility aliases
    L4AdversarialVerifier,
    L4VerifyResult,
    L4RepairDecision,
    should_exit_l4_loop,
)

from .pipeline import ReLoopPipeline, PipelineResult, RepairContext, run_reloop

from .specification import (
    run_l4,
    extract_specification,
    verify_specification,
    results_to_diagnostics as l4_results_to_diagnostics,
)

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
    "l2_verify_results_to_diagnostics",
    "l4_verify_results_to_diagnostics",  # backward compat alias
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
    # L2 Direction Analysis
    "L2DirectionVerifier",
    "L2VerifyResult",
    "L2RepairDecision",
    "should_exit_l2_loop",
    # Backward compatibility aliases
    "L4AdversarialVerifier",
    "L4VerifyResult",
    "L4RepairDecision",
    "should_exit_l4_loop",
    # Pipeline
    "ReLoopPipeline",
    "PipelineResult",
    "RepairContext",
    "run_reloop",
    # L4 Specification Compliance
    "run_l4",
    "extract_specification",
    "verify_specification",
    "l4_results_to_diagnostics",
    # Data Extraction
    "DataExtractor",
    "extract_data_from_question",
    # Experiment Runner
    "ExperimentRunner",
    "ExperimentRecord",
    "ExperimentSummary",
    "run_experiment",
]
