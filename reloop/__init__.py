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
    extract_numeric_params,
    perturb_param,
    set_param,
    should_skip_param,
    get_param_value,
)

from .generation import CodeGenerator, GenerationResult

from .repair import CodeRepairer, RepairResult

from .l4_adversarial import (
    L4AdversarialVerifier,
    L4VerifyResult,
    L4RepairDecision,
    should_exit_l4_loop,
)

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
    "verify_code",
    # Executor
    "CodeExecutor",
    # Parameter utilities
    "extract_numeric_params",
    "perturb_param",
    "set_param",
    "should_skip_param",
    "get_param_value",
    # Generation
    "CodeGenerator",
    "GenerationResult",
    # Repair
    "CodeRepairer",
    "RepairResult",
    # L4 Adversarial
    "L4AdversarialVerifier",
    "L4VerifyResult",
    "L4RepairDecision",
    "should_exit_l4_loop",
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
