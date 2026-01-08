from __future__ import annotations

import json
import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from .llm_client import LLMClient
from .prompt_stack import PromptStack
from pydantic import ValidationError

from .schemas import (
    AgentStateModel,
    CodeVersion,
    ConstraintTemplate,
    ConversationTurn,
    DataProfile,
    RepairBrief,
    SolveReport,
    SanityReport,
    SemanticProbeReport,
    SpecSheet,
    Step1Contract,
)
from .tools.data_profiler import profile_data
from .tools.persistence import PersistenceManager
from .tools.sanity_checker import run_sanity_checks
from .tools.script_runner import run_script, check_syntax
from .tools.static_auditor import audit_script
from .tools.semantic_probes import ProbeRunner, get_probe_diagnosis


# ==============================================================================
# STATE DEFINITION
# ==============================================================================

class AgentState(TypedDict, total=False):
    run_id: str
    scenario_id: str
    base_prompt_hash: str
    data: Dict[str, Any]
    scenario_text: str
    base_prompt: str
    
    # Data profile
    data_profile: DataProfile
    
    # Step outputs (matching user's original step numbering: 1-5)
    step1_contract: Step1Contract           # Step 1: Contract
    step2_spec_sheet: SpecSheet             # Step 2: Spec Sheet
    step3_templates: list[ConstraintTemplate]  # Step 3: Templates
    sanity_report: SanityReport             # Sanity Check (code-based)
    
    # Code and verification
    code_versions: list[CodeVersion]
    static_audit_reports: list[Any]
    semantic_probe_reports: list[SemanticProbeReport]
    solve_reports: list[Any]
    
    # Repair
    repair_briefs: list[RepairBrief]
    repair_count: int
    last_error: Optional[str]
    
    # Conversation
    conversation_log: list[ConversationTurn]
    turn_index: int
    codegen_conversation: list[BaseMessage]


def _validated_state(state: AgentState) -> AgentState:
    AgentStateModel.model_validate(state)
    return state


# ==============================================================================
# ORCHESTRATOR
# ==============================================================================

class AgentOrchestrator:
    """
    LangGraph orchestrator for the ReLoop pipeline.
    
    Flow (using user's original step numbering):
        profile_data -> step1 -> step2 -> step3 -> sanity_check -> step4_codegen
                                                                    |
                                                                 run_code
                                                                    |
                                                     (if error) repair_code
                                                                    |
                                                               static_audit
                                                                    |
                                                              semantic_probe
                                                                    |
                                                (if fail) repair_audit_probe
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_stack: PromptStack,
        persistence: PersistenceManager,
        repair_limit: int = 5,
        max_turns: int = 12,
        run_probes: bool = True,
    ):
        self.llm = llm_client
        self.prompt_stack = prompt_stack
        self.persistence = persistence
        self.repair_limit = repair_limit
        self.max_turns = max_turns
        self.run_probes = run_probes
        self.probe_runner = ProbeRunner() if run_probes else None
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("profile_data", self._profile_data)
        graph.add_node("step1", self._step1_contract)
        graph.add_node("step2", self._step2_spec_sheet)
        graph.add_node("step3", self._step3_templates)
        graph.add_node("sanity_check", self._sanity_check)
        graph.add_node("step4_codegen", self._step4_codegen)
        graph.add_node("run_code", self._run_code)
        graph.add_node("repair_code", self._repair_code)
        graph.add_node("static_audit", self._static_audit)
        graph.add_node("semantic_probe", self._semantic_probe)
        graph.add_node("repair_audit_probe", self._repair_audit_probe)

        # Set entry point and edges
        graph.set_entry_point("profile_data")
        graph.add_edge("profile_data", "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.add_edge("step3", "sanity_check")
        
        # After sanity: proceed or revise
        graph.add_conditional_edges(
            "sanity_check",
            self._route_after_sanity,
            {
                "revise_spec": "step2",
                "codegen": "step4_codegen",
                "done": END,
            },
        )
        
        graph.add_edge("step4_codegen", "run_code")
        
        # After run: repair on error, otherwise audit
        graph.add_conditional_edges(
            "run_code",
            self._route_after_run_code,
            {
                "repair_code": "repair_code",
                "audit": "static_audit",
                "done": END,
            },
        )
        
        graph.add_edge("repair_code", "run_code")
        graph.add_edge("static_audit", "semantic_probe")
        
        # After probes: repair on any failure, otherwise done
        graph.add_conditional_edges(
            "semantic_probe",
            self._route_after_probe,
            {
                "repair_audit_probe": "repair_audit_probe",
                "done": END,
            },
        )
        
        graph.add_edge("repair_audit_probe", "run_code")

        return graph.compile()

    def run(self, initial_state: AgentState) -> AgentState:
        self.persistence.init_run(
            initial_state["run_id"],
            {
                "scenario_id": initial_state["scenario_id"],
                "base_prompt_hash": initial_state["base_prompt_hash"],
            },
        )
        final_state = self.graph.invoke(
            initial_state, {"recursion_limit": max(50, self.repair_limit * 10)}
        )
        return _validated_state(final_state)

    def _next_turn_index(self, state: AgentState) -> int:
        state["turn_index"] = state.get("turn_index", 0) + 1
        return state["turn_index"]

    # ==========================================================================
    # HELPER: Strip markdown fences (FIXED - no extra LLM call)
    # ==========================================================================
    
    @staticmethod
    def _strip_markdown(code: str) -> str:
        """Remove markdown code fences from LLM response."""
        code = code.strip()
        
        # Remove ```python or ```json or ``` at start
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```json"):
            code = code[7:]
        elif code.startswith("```"):
            code = code[3:]
        
        # Remove leading newline after fence
        if code.startswith("\n"):
            code = code[1:]
        
        # Remove ``` at end
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()

    # ==========================================================================
    # NODE: profile_data
    # ==========================================================================
    
    def _profile_data(self, state: AgentState) -> AgentState:
        profile = profile_data(state["data"])
        state["data_profile"] = profile
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"], idx, "profile_data", step_outputs=profile
        )
        return state

    # ==========================================================================
    # LLM CALL HELPER
    # ==========================================================================

    def _call_llm(
        self,
        state: AgentState,
        step: str,
        prior_outputs: Optional[Dict[str, Any]] = None,
        runtime_extra: Optional[list[str]] = None,
        repair_mode: Optional[str] = None,
        previous_response: Optional[str] = None,
        probe_diagnosis: Optional[str] = None,
    ) -> tuple[str, int]:
        data_profile_dict = None
        if state.get("data_profile"):
            dp = state["data_profile"]
            if hasattr(dp, "model_dump"):
                data_profile_dict = dp.model_dump(mode="json")
            elif isinstance(dp, dict):
                data_profile_dict = dp

        use_codegen_chat = step in {"step4", "step6", "step7"}
        messages: List[BaseMessage]
        if use_codegen_chat and state.get("codegen_conversation"):
            messages = list(state["codegen_conversation"])
            update_blocks: list[str] = []
            if runtime_extra:
                update_blocks.append("\n".join(runtime_extra))
            if prior_outputs:
                update_blocks.append(
                    "Previous outputs:\n" + json.dumps(prior_outputs, indent=2)
                )
            if probe_diagnosis:
                update_blocks.append(f"Semantic Probe Diagnosis:\n{probe_diagnosis}")
            if repair_mode == "json" and self.prompt_stack.format_repair_json:
                update_blocks.append(self.prompt_stack.format_repair_json)
            if previous_response:
                update_blocks.append(f"Previous response:\n{previous_response}")
            if update_blocks:
                messages.append(HumanMessage(content="\n\n".join(update_blocks)))
        else:
            messages = self.prompt_stack.assemble_messages(
                step=step,
                scenario_text=state.get("scenario_text", ""),
                data_profile=data_profile_dict,
                prior_outputs=prior_outputs,
                runtime_extra=runtime_extra,
                repair_mode=repair_mode,
                previous_response=previous_response,
                probe_diagnosis=probe_diagnosis,
            )
            if use_codegen_chat:
                state["codegen_conversation"] = list(messages)

        response = self.llm.complete(messages)
        content = response.content

        idx = self._next_turn_index(state)
        turn = self.prompt_stack.log_turn(step, messages, response)
        state.setdefault("conversation_log", []).append(turn)
        self.persistence.persist_turn(state["run_id"], idx, step, messages=turn)

        if use_codegen_chat:
            state["codegen_conversation"] = list(messages) + [
                AIMessage(content=content)
            ]

        return content, idx

    def _parse_json_with_repair(
        self,
        text: str,
        state: AgentState,
        step: str,
        prior: Optional[Dict[str, Any]] = None,
        max_repairs: int = 2,
    ) -> tuple[Any, Optional[int]]:
        # Strip markdown fences first
        text = self._strip_markdown(text)
        
        for attempt in range(max_repairs + 1):
            try:
                return json.loads(text), None
            except json.JSONDecodeError as err:
                if attempt >= max_repairs:
                    raise ValueError(f"JSON parse failed after {max_repairs} repairs: {err}") from err
                text, repair_idx = self._call_llm(
                    state,
                    step,
                    prior_outputs=prior,
                    repair_mode="json",
                    previous_response=text,
                )
                # Strip markdown again after repair
                text = self._strip_markdown(text)
        return json.loads(text), None

    # ==========================================================================
    # NODE: step1 - Contract
    # ==========================================================================
    
    def _step1_contract(self, state: AgentState) -> AgentState:
        content, msg_idx = self._call_llm(state, "step1")
        payload, repair_idx = self._parse_json_with_repair(content, state, "step1")
        contract = Step1Contract.model_validate(payload)
        state["step1_contract"] = contract
        self.persistence.persist_turn(
            state["run_id"], repair_idx or msg_idx, "step1", step_outputs=contract
        )
        return state

    # ==========================================================================
    # NODE: step2 - Spec Sheet
    # ==========================================================================
    
    def _step2_spec_sheet(self, state: AgentState) -> AgentState:
        prior = {
            "contract": state["step1_contract"].model_dump(mode="json"),
        }
        content, msg_idx = self._call_llm(state, "step2", prior_outputs=prior)
        payload, repair_idx = self._parse_json_with_repair(content, state, "step2", prior)
        spec = SpecSheet.model_validate(payload)
        state["step2_spec_sheet"] = spec
        state.pop("codegen_conversation", None)
        self.persistence.persist_turn(
            state["run_id"], repair_idx or msg_idx, "step2", step_outputs=spec
        )
        return state

    # ==========================================================================
    # NODE: step3 - Constraint Templates
    # ==========================================================================
    
    def _step3_templates(self, state: AgentState) -> AgentState:
        prior = {
            "contract": state["step1_contract"].model_dump(mode="json"),
            "spec_sheet": state["step2_spec_sheet"].model_dump(mode="json"),
        }
        content, msg_idx = self._call_llm(state, "step3", prior_outputs=prior)
        payload, repair_idx = self._parse_json_with_repair(content, state, "step3", prior)
        
        if isinstance(payload, list):
            templates = [ConstraintTemplate.model_validate(t) for t in payload]
        else:
            templates = [ConstraintTemplate.model_validate(payload)]
        
        state["step3_templates"] = templates
        state.pop("codegen_conversation", None)
        self.persistence.persist_turn(
            state["run_id"], repair_idx or msg_idx, "step3", step_outputs=templates
        )
        return state

    # ==========================================================================
    # NODE: sanity_check
    # ==========================================================================
    
    def _sanity_check(self, state: AgentState) -> AgentState:
        contract = state.get("step1_contract")
        spec = state.get("step2_spec_sheet")
        templates = state.get("step3_templates", [])
        data_profile = state.get("data_profile")
        
        report = run_sanity_checks(
            contract=contract,
            spec_sheet=spec,
            templates=templates,
            data_profile=data_profile,
        )
        state["sanity_report"] = report
        
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"], idx, "sanity_check", step_outputs=report
        )
        return state

    # ==========================================================================
    # NODE: step4 - Code Generation
    # ==========================================================================
    
    def _step4_codegen(self, state: AgentState) -> AgentState:
        prior = {
            "spec_sheet": state["step2_spec_sheet"].model_dump(mode="json"),
            "constraint_templates": [
                t.model_dump(mode="json") for t in state.get("step3_templates", [])
            ],
            "sanity": state.get("sanity_report").model_dump(mode="json", by_alias=True)
            if state.get("sanity_report")
            else None,
        }
        
        content, msg_idx = self._call_llm(
            state, "step4", 
            prior_outputs=prior
        )
        
        # ===== FIXED: Strip markdown directly, no extra LLM call =====
        code_text = self._strip_markdown(content)
        # =============================================================
        
        state.setdefault("code_versions", []).append(
            CodeVersion(step="step4", content=code_text)
        )
        self.persistence.persist_turn(
            state["run_id"], msg_idx, "step4_codegen", code=code_text
        )
        return state

    # ==========================================================================
    # NODE: static_audit
    # ==========================================================================
    
    def _static_audit(self, state: AgentState) -> AgentState:
        code_text = state["code_versions"][-1].content
        audit = audit_script(code_text)
        state.setdefault("static_audit_reports", []).append(audit)
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"], idx, "static_audit", static_audit=audit
        )
        if not audit.passed:
            state["last_error"] = "static_audit_failed"
        return state

    # ==========================================================================
    # NODE: semantic_probe
    # ==========================================================================
    
    def _semantic_probe(self, state: AgentState) -> AgentState:
        """Run semantic probes to detect constraint errors before full benchmark."""
        
        if not self.run_probes or self.probe_runner is None:
            state.setdefault("semantic_probe_reports", []).append(
                SemanticProbeReport(
                    total=0, passed=0, failed=0, crashed=0, pass_rate=1.0
                )
            )
            return state
        
        code_text = state["code_versions"][-1].content
        
        # Check syntax first
        syntax_ok, syntax_err = check_syntax(code_text)
        if not syntax_ok:
            state.setdefault("semantic_probe_reports", []).append(
                SemanticProbeReport(
                    total=8, passed=0, failed=0, crashed=8, pass_rate=0.0,
                    diagnoses={"syntax": f"Syntax error: {syntax_err}"}
                )
            )
            state["last_error"] = "syntax_error"
            return state
        
        # Run probes
        probe_report = self.probe_runner.run_all_probes(code_text)
        state.setdefault("semantic_probe_reports", []).append(probe_report)
        
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"], idx, "semantic_probe", 
            semantic_probe_report=probe_report  # FIX: use correct parameter
        )
        
        if probe_report.failed > 0 or probe_report.crashed > 0:
            state["last_error"] = "probe_failed"
        else:
            # CRITICAL: Clear last_error when probes pass!
            state["last_error"] = None
        
        return state

    # ==========================================================================
    # NODE: run_code
    # ==========================================================================
    
    def _run_code(self, state: AgentState) -> AgentState:
        code_text = state["code_versions"][-1].content
        
        try:
            solve_report, stdout, stderr = run_script(code_text, state["data"])
        except Exception as exc:
            solve_report = SolveReport(
                status="RUNTIME_ERROR", stderr=str(exc), stdout="", exit_code=-1
            )
            stdout = ""
            stderr = str(exc)
            state["last_error"] = str(exc)
        
        state.setdefault("solve_reports", []).append(solve_report)
        
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"],
            idx,
            "run_code",
            stdout=stdout,
            stderr=stderr,
            solve_report=solve_report,
        )

        status = solve_report.status
        if status in ("2", "GRB.OPTIMAL", "OPTIMAL"):
            state["last_error"] = None
        else:
            state["last_error"] = "run_failed"
        return state

    # ==========================================================================
    # NODE: step6 - Repair Code After Runtime Error
    # ==========================================================================
    
    def _repair_code(self, state: AgentState) -> AgentState:
        last_solve = state.get("solve_reports", [])[-1] if state.get("solve_reports") else None
        prompt = self.prompt_stack.repair_code_prompt or "Fix the code based on the runtime error."
        update_blocks: List[str] = [prompt]
        if last_solve:
            update_blocks.append(
                "Runtime error details:\n"
                + json.dumps(last_solve.model_dump(mode="json", by_alias=True), indent=2)
            )
        update_blocks.append("Output only raw Python code. No markdown, no comments.")
        content, msg_idx = self._call_llm(
            state, "step6", runtime_extra=update_blocks
        )
        code_text = self._strip_markdown(content)
        state.setdefault("code_versions", []).append(
            CodeVersion(step="step6", content=code_text)
        )
        self.persistence.persist_turn(
            state["run_id"], msg_idx, "step6_repair_code", code=code_text
        )
        state["repair_count"] = state.get("repair_count", 0) + 1
        return state

    # ==========================================================================
    # NODE: step7 - Repair After Audit/Probe Failures
    # ==========================================================================
    
    def _repair_audit_probe(self, state: AgentState) -> AgentState:
        prompt = (
            self.prompt_stack.repair_audit_probe_prompt
            or "Fix the code based on static audit and semantic probe failures."
        )
        update_blocks: List[str] = [prompt]
        if state.get("static_audit_reports"):
            last_audit = state["static_audit_reports"][-1]
            issues = "\n".join(last_audit.issues) if last_audit.issues else "unspecified"
            update_blocks.append(f"Static audit issues:\n{issues}")
        if state.get("semantic_probe_reports"):
            last_probe = state["semantic_probe_reports"][-1]
            if last_probe.diagnoses:
                diag_lines = "\n".join(
                    f"- {name}: {diag}" for name, diag in last_probe.diagnoses.items()
                )
            else:
                diag_lines = "\n".join(last_probe.failed_probes)
            update_blocks.append(f"Semantic probe failures:\n{diag_lines}")
        update_blocks.append("Output only raw Python code. No markdown, no comments.")
        content, msg_idx = self._call_llm(
            state, "step7", runtime_extra=update_blocks
        )
        code_text = self._strip_markdown(content)
        state.setdefault("code_versions", []).append(
            CodeVersion(step="step7", content=code_text)
        )
        self.persistence.persist_turn(
            state["run_id"], msg_idx, "step7_repair_audit_probe", code=code_text
        )
        state["repair_count"] = state.get("repair_count", 0) + 1
        return state

    # ==========================================================================
    # ROUTING FUNCTIONS
    # ==========================================================================
    
    def _route_after_sanity(self, state: AgentState):
        if state.get("turn_index", 0) >= self.max_turns:
            return "done"
        report = state.get("sanity_report")
        if report and not report.overall_pass:
            return "revise_spec"
        return "codegen"

    def _route_after_run_code(self, state: AgentState):
        if state.get("turn_index", 0) >= self.max_turns:
            return "done"
        if state.get("last_error") is None:
            return "audit"
        if state.get("repair_count", 0) >= self.repair_limit:
            return "done"
        return "repair_code"
    
    def _route_after_probe(self, state: AgentState):
        if state.get("turn_index", 0) >= self.max_turns:
            return "done"
        audit_failed = False
        if state.get("static_audit_reports"):
            audit_failed = not state["static_audit_reports"][-1].passed
        probe_failed = False
        if state.get("semantic_probe_reports"):
            last_probe = state["semantic_probe_reports"][-1]
            probe_failed = last_probe.failed > 0 or last_probe.crashed > 0
        if not audit_failed and not probe_failed:
            return "done"
        if state.get("repair_count", 0) >= self.repair_limit:
            return "done"
        return "repair_audit_probe"


# ==============================================================================
# HELPER FUNCTION
# ==============================================================================

def build_initial_state(
    scenario_id: str, data: Dict[str, Any], base_prompt: str, scenario_text: str
) -> AgentState:
    run_id = uuid.uuid4().hex
    prompt_hash = hashlib.sha256(base_prompt.encode("utf-8")).hexdigest()
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "base_prompt_hash": prompt_hash,
        "data": data,
        "scenario_text": scenario_text,
        "base_prompt": base_prompt,
        "repair_count": 0,
        "conversation_log": [],
        "code_versions": [],
        "static_audit_reports": [],
        "semantic_probe_reports": [],
        "solve_reports": [],
        "repair_briefs": [],
        "turn_index": 0,
    }
