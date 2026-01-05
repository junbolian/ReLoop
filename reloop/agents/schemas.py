from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ==============================================================================
# DATA PROFILE
# ==============================================================================

class DataFieldProfile(BaseModel):
    path: str
    kind: str  # "scalar", "list", "dict"
    type: str
    length: Optional[int] = None
    element_type: Optional[str] = None
    key_types: Optional[str] = None
    sample_keys: Optional[List[str]] = None


class DataProfile(BaseModel):
    summary: str
    fields: List[DataFieldProfile]


# ==============================================================================
# STEP 1: CONTRACT
# ==============================================================================

class SoftViolation(BaseModel):
    name: str
    penalty: Optional[str] = None
    penalty_source: Optional[str] = None


class Step1Contract(BaseModel):
    """Contract extracted in Step 1."""
    optimize: str
    controls: List[str]
    hard_constraints: List[str]
    soft_violations: List[SoftViolation] = []
    summary: Optional[str] = None
    contract_summary: Optional[str] = None


# ==============================================================================
# STEP 2: SPEC SHEET
# ==============================================================================

class SetDef(BaseModel):
    name: str
    description: str
    source: str


class DecisionDef(BaseModel):
    name: str
    type: str
    domain: str
    indices: List[str]
    meaning: str
    active_if: str = "always"


class ObjectiveTerm(BaseModel):
    name: str
    expression: str
    source: str
    active_if: str = "always"


class ConstraintFamily(BaseModel):
    prefix: str
    meaning: str
    indices: List[str]
    sense: str
    active_if: str = "always"


class EdgeCase(BaseModel):
    case: str
    handling: str


class OpenQuestion(BaseModel):
    question: str


class SpecSheet(BaseModel):
    """Spec sheet from Step 2."""
    sets: List[SetDef] = []
    decisions: List[DecisionDef] = []
    objective_terms: List[ObjectiveTerm] = []
    constraint_families: List[ConstraintFamily] = []
    edge_cases: List[EdgeCase] = []
    open_questions: List[OpenQuestion] = []


# ==============================================================================
# STEP 3: CONSTRAINT TEMPLATES
# ==============================================================================

class Equation(BaseModel):
    name_suffix: str
    sense: str
    lhs: str
    rhs: str


class ConstraintTemplate(BaseModel):
    """Constraint template from Step 3."""
    prefix: str
    template_type: str
    applies_when: str
    indices: List[str]
    equations: List[Equation]
    notes: List[str] = []


# ==============================================================================
# SANITY CHECK
# ==============================================================================

class SanityCheck(BaseModel):
    """Sanity check result."""
    id: Optional[int] = None
    name: str
    passed: bool = True
    message: str = ""
    fix_hint: str = ""
    
    class Config:
        extra = "allow"


class SanityReport(BaseModel):
    overall_pass: bool
    checks: List[SanityCheck]
    blockers: List[str] = []
    recommended_next_step: Optional[str] = None


# ==============================================================================
# CODE VERSION
# ==============================================================================

class CodeVersion(BaseModel):
    step: str
    content: str


# ==============================================================================
# STATIC AUDIT
# ==============================================================================

class StaticAuditReport(BaseModel):
    passed: bool
    issues: List[str] = []


# ==============================================================================
# SEMANTIC PROBE
# ==============================================================================

class ProbeResult(BaseModel):
    """Result of a single semantic probe."""
    probe_name: str
    result: str
    expected: Optional[Any] = None  # FIX: Changed from str to Any
    actual: Optional[Any] = None    # FIX: Changed from str to Any
    diagnosis: Optional[str] = None


class SemanticProbeReport(BaseModel):
    """Aggregate result of all semantic probes."""
    total: int
    passed: int
    failed: int
    crashed: int = 0
    pass_rate: float
    failed_probes: List[str] = []
    probe_results: List[ProbeResult] = []
    diagnoses: Dict[str, str] = {}


# ==============================================================================
# SOLVE REPORT
# ==============================================================================

class SolveReport(BaseModel):
    status: str
    obj_val: Optional[float] = Field(None, alias="objVal")
    obj_bound: Optional[float] = Field(None, alias="objBound")
    mip_gap: Optional[float] = Field(None, alias="mipGap")
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    class Config:
        populate_by_name = True


# ==============================================================================
# REPAIR BRIEF
# ==============================================================================

class RepairBrief(BaseModel):
    """Repair diagnosis from Step 5."""
    target: str = "CODEGEN"
    error_type: Optional[str] = None
    likely_cause: Optional[str] = None
    fix: Optional[str] = None
    failed_probes: List[str] = []
    diagnosis: Optional[Dict[str, Any]] = None
    repairs: Optional[List[Dict[str, Any]]] = None
    next_step_prompting: Optional[Dict[str, Any]] = None


# ==============================================================================
# CONVERSATION LOG
# ==============================================================================

class MessageEntry(BaseModel):
    role: str
    content: str
    step: Optional[str] = None


class ConversationTurn(BaseModel):
    step: str
    messages: List[MessageEntry]
    response_text: Optional[str] = None
    raw_response: Optional[Any] = None


# ==============================================================================
# EVALUATION RESULT
# ==============================================================================

class EvaluationResult(BaseModel):
    """Result of evaluating a scenario with semantic probes."""
    scenario_id: str
    syntax_valid: bool
    probes_total: int
    probes_passed: int
    probes_failed: int
    model_status: Optional[str] = None
    model_objective: Optional[float] = None
    ground_truth_objective: Optional[float] = None
    objective_error_pct: Optional[float] = None
    objective_match: bool = False
    final_verdict: str


# ==============================================================================
# AGENT STATE MODEL
# ==============================================================================

class AgentStateModel(BaseModel):
    """Pydantic model for validating agent state."""
    run_id: str
    scenario_id: str
    base_prompt_hash: str
    data: Dict[str, Any]
    scenario_text: str = ""
    base_prompt: str = ""
    data_profile: Optional[DataProfile] = None
    step1_contract: Optional[Step1Contract] = None
    step2_spec_sheet: Optional[SpecSheet] = None
    step3_templates: Optional[List[ConstraintTemplate]] = None
    sanity_report: Optional[SanityReport] = None
    code_versions: List[CodeVersion] = []
    static_audit_reports: List[StaticAuditReport] = []
    semantic_probe_reports: List[SemanticProbeReport] = []
    solve_reports: List[SolveReport] = []
    repair_briefs: List[RepairBrief] = []
    repair_count: int = 0
    last_error: Optional[str] = None
    conversation_log: List[ConversationTurn] = []
    turn_index: int = 0

    class Config:
        extra = "allow"


# ==============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# ==============================================================================

Step0Contract = Step1Contract
Step1SpecSheet = SpecSheet
Step2Templates = ConstraintTemplate
SanityCheckResult = SanityCheck