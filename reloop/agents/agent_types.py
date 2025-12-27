from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    raw: Any = None


@dataclass
class ContractReport:
    ok: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    traceback: str
    duration: float
    status_code: Optional[int] = None
    status_str: str = ""
    feasible: Optional[bool] = None
    objective: Optional[float] = None
    constr_names: List[str] = field(default_factory=list)
    var_names: List[str] = field(default_factory=list)
    iis_constraints: List[Dict[str, Any]] = field(default_factory=list)
    license_limited: bool = False


@dataclass
class SemanticReport:
    valid: bool
    missing_prefixes: List[str] = field(default_factory=list)
    unexpected_modules: List[str] = field(default_factory=list)
    suspicious: List[str] = field(default_factory=list)


@dataclass
class AgentRound:
    round_id: int
    messages: List[LLMMessage]
    script: str
    llm_response: Optional[LLMResponse] = None
    contract_report: Optional[ContractReport] = None
    exec_result: Optional[ExecutionResult] = None
    semantic_report: Optional[SemanticReport] = None
    repair_brief: Optional[str] = None
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class AgentSummary:
    scenario_id: str
    mode: str
    compilation_ok: bool
    feasible: bool
    semantic_valid: bool
    status: str
    objective: Optional[float]
    iters: int
    runtime_s: float
    tokens_in: int
    tokens_out: int
    total_cost_est: float
    config: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
