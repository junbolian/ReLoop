from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import datetime as _dt


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format with seconds."""
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class RouteResult(BaseModel):
    """Routing decision for an input scenario."""

    archetype_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    missing_inputs: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ExtractionConfig(BaseModel):
    """Configuration for extraction and canonicalization behaviour."""

    use_llm: bool = False
    prompt_version: str = "v1"
    strict_mode: bool = False
    auto_apply_trusted: bool = True
    suggest_candidate: bool = True
    allow_candidate_auto_apply: bool = False
    max_retries: int = 1

    model_config = ConfigDict(extra="forbid")


class AllocationItem(BaseModel):
    """Generalized allocation item (diet_lp alias)."""

    name: str
    cost: float
    features: Dict[str, float]

    model_config = ConfigDict(extra="allow", populate_by_name=True, alias_generator=None, protected_namespaces=())

    @classmethod
    def model_validate(cls, data):
        if isinstance(data, dict) and "features" not in data and "nutrients" in data:
            data = dict(data)
            data["features"] = data.pop("nutrients")
        return super().model_validate(data)


class AllocationSchema(BaseModel):
    items: List[AllocationItem]
    requirements: Dict[str, float]

    model_config = ConfigDict(extra="forbid")


class CanonicalAllocationSchema(BaseModel):
    """Allocation schema with deterministic ordering and filled feature matrix."""

    items: List[AllocationItem]
    requirements: Dict[str, float]
    feature_order: List[str]
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class TransportFlowSchema(BaseModel):
    sources: List[Dict[str, Any]]  # each {name, supply}
    sinks: List[Dict[str, Any]]  # each {name, demand}
    costs: Dict[str, Dict[str, float]]  # costs[src][sink]

    model_config = ConfigDict(extra="forbid")


class AssignmentSchema(BaseModel):
    workers: List[str]
    tasks: List[str]
    costs: Dict[str, Dict[str, float]]  # costs[worker][task]

    model_config = ConfigDict(extra="forbid")


class FacilityLocationSchema(BaseModel):
    facilities: List[Dict[str, Any]]  # each {name, open_cost, capacity (optional)}
    customers: List[str]
    ship_cost: Dict[str, Dict[str, float]]  # cost[f][c]
    demand: Dict[str, float] = Field(default_factory=dict)  # optional per customer

    model_config = ConfigDict(extra="forbid")


class RetailInventorySchema(BaseModel):
    """Retail data-driven scenario; may be empty if data not provided."""

    data: Optional[Dict[str, Any]] = None
    coverage_level: str = "baseline_no_aging"
    missing_inputs: List[str] = Field(default_factory=list)
    missing_capabilities: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


ProblemSchema = Union[
    AllocationSchema,
    TransportFlowSchema,
    AssignmentSchema,
    FacilityLocationSchema,
    RetailInventorySchema,
]


class ModelHandle(BaseModel):
    """Wrap Gurobi model and variable references."""

    archetype_id: str
    model: Any
    variables: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class SolveResult(BaseModel):
    status: str
    objective: Optional[float] = None
    runtime: Optional[float] = None
    mip_gap: Optional[float] = None
    solver_log: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class AuditViolation(BaseModel):
    name: str
    lhs: float
    rhs: float
    slack: float

    model_config = ConfigDict(extra="forbid")


class AuditResult(BaseModel):
    passed: bool
    violations: List[AuditViolation] = Field(default_factory=list)
    audit_score: float = 0.0
    symptom_tags: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class Diagnosis(BaseModel):
    summary: str
    symptom_tags: List[str] = Field(default_factory=list)
    recommended_params: Dict[str, Any] = Field(default_factory=dict)
    iis_constraints: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class FixRecord(BaseModel):
    """Patch recipe representing a potential or trusted fix."""

    fix_id: str
    archetype_id: str
    symptom_tags: List[str]
    patch_type: str
    patch_payload: Dict[str, Any]
    preconditions: List[str]
    expected_effect: str
    created_at: str = Field(default_factory=utc_now_iso)
    human_label: str = "unknown"  # correct | wrong | skip | unknown
    notes: Optional[str] = None
    source_run_id: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class SandboxReport(BaseModel):
    recommendation: str  # apply | reject | inconclusive
    baseline_objective: Optional[float]
    candidate_objective: Optional[float]
    baseline_audit_pass: bool
    candidate_audit_pass: bool
    comparison_notes: str

    model_config = ConfigDict(extra="allow")


class NotYetImplemented(BaseModel):
    archetype_id: str
    missing_inputs: List[str] = Field(default_factory=list)
    missing_capabilities: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    coverage_level: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class RunRecord(BaseModel):
    """Persistent run log for reproducibility and auditing."""

    run_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    input_text: str
    archetype_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    route: Optional[RouteResult] = None
    extraction_config: ExtractionConfig
    extracted_schema: Optional[ProblemSchema] = None
    canonical_schema: Optional[Any] = None
    model_plan: Optional[Dict[str, Any]] = None
    model_plan_hash: Optional[str] = None
    model_summary: Optional[Dict[str, Any]] = None
    solve_result: Optional[SolveResult] = None
    audit_result: Optional[AuditResult] = None
    diagnosis: Optional[Diagnosis] = None
    symptoms: List[str] = Field(default_factory=list)
    candidate_fix: Optional[FixRecord] = None
    applied_fixes: List[FixRecord] = Field(default_factory=list)
    sandbox_report: Optional[SandboxReport] = None
    final_status: Optional[str] = None
    not_implemented: Optional[NotYetImplemented] = None
    expected_objective: Optional[float] = None
    expected_tolerance: Optional[float] = None
    is_correct: Optional[bool] = None
    prompt_injections: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class RunState(BaseModel):
    """
    In-memory orchestration state carried through the LangGraph pipeline.

    This is the single source of truth for the current run. It is persisted
    into a RunRecord upon completion.
    """

    run_id: str
    input_text: str
    extraction_config: ExtractionConfig = Field(default_factory=ExtractionConfig)
    archetype_id: Optional[str] = None
    route: Optional[RouteResult] = None
    input_data: Optional[Dict[str, Any]] = None  # external data dict (retail)
    extracted_schema: Optional[ProblemSchema] = None
    canonical_schema: Optional[Any] = None
    model_handle: Optional[ModelHandle] = None
    solve_result: Optional[SolveResult] = None
    audit_result: Optional[AuditResult] = None
    diagnosis: Optional[Diagnosis] = None
    symptoms: List[str] = Field(default_factory=list)
    candidate_fix: Optional[FixRecord] = None
    sandbox_report: Optional[SandboxReport] = None
    applied_fixes: List[FixRecord] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    final_status: Optional[str] = None
    not_implemented: Optional[NotYetImplemented] = None
    expected_objective: Optional[float] = None
    expected_tolerance: Optional[float] = None
    is_correct: Optional[bool] = None
    model_plan: Optional[Dict[str, Any]] = None
    model_plan_hash: Optional[str] = None
    prompt_injections: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


__all__ = [
    "RouteResult",
    "ExtractionConfig",
    "AllocationItem",
    "AllocationSchema",
    "CanonicalAllocationSchema",
    "TransportFlowSchema",
    "AssignmentSchema",
    "FacilityLocationSchema",
    "RetailInventorySchema",
    "ProblemSchema",
    "ModelHandle",
    "SolveResult",
    "AuditResult",
    "AuditViolation",
    "Diagnosis",
    "FixRecord",
    "SandboxReport",
    "RunRecord",
    "RunState",
    "NotYetImplemented",
    "utc_now_iso",
]
