from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SoftViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    penalty_source: str


class Step0Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optimize: str
    controls: List[str]
    hard_constraints: List[str]
    soft_violations: List[SoftViolation]
    contract_summary: str


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sets: List[str] = Field(default_factory=list)
    params: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    rule_hints: List[str] = Field(default_factory=list)


class TaggedSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str
    tag: Literal[
        "SET_OBJECT",
        "PARAM_FACT",
        "DECISION",
        "RULE_CONSTRAINT_SOURCE",
        "OBJECTIVE_PREFERENCE",
    ]
    extracted: Extraction


class SetDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    source: str


class DecisionVar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: str
    domain: str
    indices: List[str]
    meaning: str
    active_if: str


class ObjectiveTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    expression: str
    source: str
    active_if: str


class ConstraintFamily(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prefix: str
    meaning: str
    indices: List[str]
    sense: str
    active_if: str


class EdgeCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case: str
    handling: str


class OpenQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str


class SpecSheet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sets: List[SetDef]
    decisions: List[DecisionVar]
    objective_terms: List[ObjectiveTerm]
    constraint_families: List[ConstraintFamily]
    edge_cases: List[EdgeCase]
    open_questions: List[OpenQuestion]


class EquationTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name_suffix: str
    sense: str
    lhs: str
    rhs: str


class ConstraintTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prefix: str
    template_type: Literal[
        "BALANCE",
        "CAPACITY",
        "DYNAMICS",
        "NETWORK",
        "SUBSTITUTION",
        "DISCRETE",
    ]
    applies_when: str
    indices: List[str]
    equations: List[EquationTemplate]
    notes: List[str] = Field(default_factory=list)


class SanityCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    name: str
    pass_: bool = Field(alias="pass")
    reason: str
    fix_hint: str

    @property
    def passed(self) -> bool:
        return self.pass_


class SanityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checks: List[SanityCheckResult]
    overall_pass: bool
    blockers: List[str]
    recommended_next_step: Literal["PROCEED_TO_CODEGEN", "REVISE_SPEC"]


class CodeVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StaticAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    failures: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SolveReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Optional[str] = None
    obj_val: Optional[float] = None
    obj_bound: Optional[float] = None
    mip_gap: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    elapsed_sec: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IISReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    constraints: List[str] = Field(default_factory=list)
    prefix_summary: Dict[str, int] = Field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RepairAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    change_type: Literal[
        "ADD",
        "REMOVE",
        "TIGHTEN",
        "RELAX",
        "ALIGN_TIME",
        "ADD_SOFT_SLACK",
        "FIX_SUBSTITUTION_COUPLING",
        "FIX_BOUNDS",
    ]
    where: Literal[
        "spec_sheet.constraint_families",
        "constraint_templates",
        "codegen_instructions",
    ]
    description: str
    acceptance_test: str


class RepairDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal[
        "INFEASIBLE",
        "UNBOUNDED",
        "WEIRD_SOLUTION",
        "FORMAT_ERROR",
        "AUDIT_FAIL",
        "RUNTIME_ERROR",
    ]
    most_likely_causes: List[str]
    evidence: List[str]
    affected_constraint_prefixes: List[str] = Field(default_factory=list)


class NextStepPrompting(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extra_instructions_to_inject: List[str] = Field(default_factory=list)


class RepairBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["SPEC", "CODEGEN"]
    diagnosis: RepairDiagnosis
    repairs: List[RepairAction]
    next_step_prompting: NextStepPrompting


class MessageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "human", "assistant"]
    content: str
    step: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: str
    messages: List[MessageEntry]
    response_text: Optional[str] = None
    raw_response: Any = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DataFieldProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    kind: Literal["scalar", "list", "dict"]
    type: str
    length: Optional[int] = None
    key_types: Optional[str] = None
    element_type: Optional[str] = None
    sample_keys: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class DataProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    fields: List[DataFieldProfile]


class AgentStateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    scenario_id: str
    base_prompt_hash: str
    data_profile: Optional[DataProfile] = None
    step0_contract: Optional[Step0Contract] = None
    step1_tags: List[TaggedSentence] = Field(default_factory=list)
    step2_spec_sheet: Optional[SpecSheet] = None
    step3_templates: List[ConstraintTemplate] = Field(default_factory=list)
    step4_sanity_report: Optional[SanityReport] = None
    code_versions: List[CodeVersion] = Field(default_factory=list)
    static_audit_reports: List[StaticAuditReport] = Field(default_factory=list)
    solve_reports: List[SolveReport] = Field(default_factory=list)
    iis_reports: List[IISReport] = Field(default_factory=list)
    repair_briefs: List[RepairBrief] = Field(default_factory=list)
    repair_count: int = 0
    last_error: Optional[str] = None
    conversation_log: List[ConversationTurn] = Field(default_factory=list)
    scenario_text: Optional[str] = None
    base_prompt: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    turn_index: int = 0
