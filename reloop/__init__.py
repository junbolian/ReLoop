"""
ReLoop: Reliable LLM-based Optimization Modeling
via Sensitivity-Based Behavioral Verification

Architecture:
  Module 1: Structured Generation (3-step)
  Module 2: Behavioral Verification (6-layer, Core)
  Module 3: Diagnosis-Guided Repair

Key Principles:
  1. No Archetype: Works for any optimization problem
  2. L3 is Core: Monotonicity check is universal
  3. L4 is Best-Effort: Skip if role cannot be inferred
  4. LLM sees Schema only: Not actual data values
  5. No Human Feedback: Fully automated

Usage:
    from reloop import ReLoop, ReLoopConfig, run_reloop, verify_code
"""

# Main pipeline
from .reloop import (
    ReLoop,
    ReLoopResult,
    ReLoopConfig,
    run_reloop,
    verify_code,
)

# Module 1: Structured Generation
from .structured_generation import (
    StructuredGenerator,
    LLMClient,
    OpenAIClient,
    get_schema_for_dataset,
    detect_dataset_type,
    RETAIL_SCHEMA,
    MAMO_SCHEMA,
    NL4OPT_SCHEMA,
)

# Prompts
from .prompts import (
    PromptGenerator,
    BASELINE_PROMPT,
    STEP1_PROMPT,
    STEP2_PROMPT,
    STEP3_PROMPT,
    REPAIR_PROMPT,
    MAMO_BASELINE_PROMPT,
    NL4OPT_BASELINE_PROMPT,
    RETAIL_SCHEMA_TEMPLATE,
)

# Module 2: Behavioral Verification
from .behavioral_verification import (
    BehavioralVerifier,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
    CodeExecutor,
)

# Module 3: Diagnosis and Repair
from .diagnosis_repair import (
    DiagnosisRepairer,
    RepairContext,
)

# Error patterns
from .error_patterns import (
    ERROR_PATTERNS,
    get_repair_hints,
    format_repair_guidance,
)

# Parameter utilities
from .param_utils import (
    ParameterRole,
    extract_numeric_params,
    infer_param_role,
    get_expected_direction,
    perturb_param,
    set_param,
    get_param_value,
    should_skip_param,
    is_effectively_zero,
)


__version__ = "1.0.0"
__all__ = [
    # Main
    "ReLoop", "ReLoopResult", "ReLoopConfig", "run_reloop", "verify_code",
    # Generation
    "StructuredGenerator", "LLMClient", "OpenAIClient", "get_schema_for_dataset", "detect_dataset_type",
    "RETAIL_SCHEMA", "MAMO_SCHEMA", "NL4OPT_SCHEMA",
    # Prompts
    "PromptGenerator", "BASELINE_PROMPT", "STEP1_PROMPT", "STEP2_PROMPT", "STEP3_PROMPT", "REPAIR_PROMPT",
    "MAMO_BASELINE_PROMPT", "NL4OPT_BASELINE_PROMPT", "RETAIL_SCHEMA_TEMPLATE",
    # Verification
    "BehavioralVerifier", "VerificationReport", "VerificationResult", "VerificationStatus", "CodeExecutor",
    # Repair
    "DiagnosisRepairer", "RepairContext", "ERROR_PATTERNS", "get_repair_hints", "format_repair_guidance",
    # Params
    "ParameterRole", "extract_numeric_params", "infer_param_role", "get_expected_direction",
    "perturb_param", "set_param", "get_param_value", "should_skip_param", "is_effectively_zero",
]
