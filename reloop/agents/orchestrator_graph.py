from __future__ import annotations

import json
import hashlib
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, TypedDict
import ast
import re

from langchain_core.messages import BaseMessage
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
    SpecSheet,
    Step0Contract,
    TaggedSentence,
)
from .tools.data_profiler import profile_data
from .tools.persistence import PersistenceManager
from .tools.sanity_checker import run_sanity_checks
from .tools.script_runner import run_script
from .tools.static_auditor import audit_script


class AgentState(TypedDict, total=False):
    run_id: str
    scenario_id: str
    base_prompt_hash: str
    data: Dict[str, Any]
    scenario_text: str
    data_profile: DataProfile
    step0_contract: Step0Contract
    step1_tags: list[TaggedSentence]
    step2_spec_sheet: SpecSheet
    step3_templates: list[ConstraintTemplate]
    step4_sanity_report: SanityReport
    code_versions: list[CodeVersion]
    static_audit_reports: list[Any]
    solve_reports: list[Any]
    iis_reports: list[Any]
    repair_briefs: list[RepairBrief]
    repair_count: int
    last_error: Optional[str]
    conversation_log: list[ConversationTurn]
    base_prompt: str
    turn_index: int


def _validated_state(state: AgentState) -> AgentState:
    AgentStateModel.model_validate(state)
    return state


class AgentOrchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_stack: PromptStack,
        persistence: PersistenceManager,
        repair_limit: int = 5,
    ):
        self.llm = llm_client
        self.prompt_stack = prompt_stack
        self.persistence = persistence
        self.repair_limit = repair_limit
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("profile_data", self._profile_data)
        graph.add_node("step0", self._step0_contract)
        graph.add_node("step1", self._step1_tags)
        graph.add_node("step2", self._step2_spec_sheet)
        graph.add_node("step3", self._step3_templates)
        graph.add_node("step4", self._step4_sanity)
        graph.add_node("step5_codegen", self._step5_codegen)
        graph.add_node("static_audit", self._static_audit)
        graph.add_node("step6_run_and_diagnose", self._step6_run_and_diagnose)

        graph.set_entry_point("profile_data")
        graph.add_edge("profile_data", "step0")
        graph.add_edge("step0", "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", "step3")
        graph.add_edge("step3", "step4")
        graph.add_conditional_edges(
            "step4",
            self._route_after_sanity,
            {
                "revise_spec": "step2",
                "codegen": "step5_codegen",
            },
        )
        graph.add_edge("step5_codegen", "static_audit")

        graph.add_conditional_edges(
            "static_audit",
            self._route_after_audit,
            {
                "retry_codegen": "step5_codegen",
                "run": "step6_run_and_diagnose",
            },
        )

        graph.add_conditional_edges(
            "step6_run_and_diagnose",
            self._route_after_run,
            {
                "revise_spec": "step2",
                "regenerate_code": "step5_codegen",
                "done": END,
            },
        )

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

    def _profile_data(self, state: AgentState) -> AgentState:
        profile = profile_data(state["data"])
        state["data_profile"] = profile
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"], idx, "profile_data", step_outputs=profile
        )
        self.persistence.log_event(
            state["run_id"],
            {"ts": datetime.utcnow().isoformat(), "event": "profile_data"},
        )
        return state

    def _call_llm(
        self,
        state: AgentState,
        step: str,
        prior_outputs: Optional[Dict[str, Any]] = None,
        repair_mode: Optional[str] = None,
        previous_response: Optional[str] = None,
    ):
        messages: list[BaseMessage] = self.prompt_stack.assemble_messages(
            step=step,
            scenario_text=state.get("scenario_text", ""),
            data_profile=state.get("data_profile").model_dump(mode="json")
            if state.get("data_profile")
            else None,
            prior_outputs=prior_outputs,
            repair_mode=repair_mode,
            previous_response=previous_response,
        )
        response = self.llm.complete(messages)
        turn = self.prompt_stack.log_turn(step, messages, response)
        state.setdefault("conversation_log", []).append(turn)
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"],
            idx,
            step,
            messages=turn,
        )
        return response.content, idx

    def _parse_json_with_repair(
        self,
        text: str,
        state: AgentState,
        step: str,
        prior_outputs: Dict[str, Any],
        default_on_fail: Any = None,
    ) -> Any:
        def _json_candidates(raw: str):
            stripped = raw.strip()
            if stripped.startswith("```"):
                # remove fences and possible language tag
                stripped = re.sub(r"^```[a-zA-Z]*", "", stripped).strip()
                stripped = stripped.rstrip("`").strip()
            candidates = [stripped]
            # balanced brace/bracket scanning
            for start_char, end_char in (("{", "}"), ("[", "]")):
                opens = [m.start() for m in re.finditer(re.escape(start_char), stripped)]
                closes = [m.start() for m in re.finditer(re.escape(end_char), stripped)]
                if opens and closes and max(closes) > min(opens):
                    candidates.append(stripped[min(opens) : max(closes) + 1])
            # remove trailing commas
            dedup = []
            for cand in candidates:
                cleaned = re.sub(r",(\s*[}\]])", r"\1", cand)
                if cleaned not in dedup:
                    dedup.append(cleaned)
            return dedup

        def _coerce_json(raw: str):
            for candidate in _json_candidates(raw):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        val = ast.literal_eval(candidate)
                        return val
                    except Exception:
                        continue
            return None

        parsed = _coerce_json(text)
        if parsed is not None:
            return parsed, None

        repaired, idx = self._call_llm(
            state,
            step,
            prior_outputs=prior_outputs,
            repair_mode="json",
            previous_response=text,
        )
        repaired_parsed = _coerce_json(repaired)
        if repaired_parsed is None:
            state["last_error"] = f"json_parse_failed_{step}"
            if default_on_fail is not None:
                return default_on_fail, idx
            raise ValueError(f"Could not parse JSON for {step}")
        return repaired_parsed, idx

    def _step0_contract(self, state: AgentState) -> AgentState:
        content, msg_idx = self._call_llm(state, "step0", prior_outputs=None)
        payload, repair_idx = self._parse_json_with_repair(content, state, "step0", {})
        contract = Step0Contract.model_validate(payload)
        state["step0_contract"] = contract
        idx = repair_idx or msg_idx
        self.persistence.persist_turn(
            state["run_id"], idx, "step0", step_outputs=contract
        )
        self.persistence.log_event(
            state["run_id"],
            {"ts": datetime.utcnow().isoformat(), "event": "step0_contract"},
        )
        return state

    def _step1_tags(self, state: AgentState) -> AgentState:
        prior = {"step0_contract": state["step0_contract"].model_dump(mode="json")}
        content, msg_idx = self._call_llm(state, "step1", prior_outputs=prior)
        raw_payload, repair_idx = self._parse_json_with_repair(
            content, state, "step1", prior, default_on_fail=[]
        )

        def _normalize(item: Dict[str, Any]) -> Dict[str, Any]:
            if "extracted" not in item:
                item["extracted"] = {"sets": [], "params": [], "decisions": [], "rule_hints": []}
            if "rule_hints" in item and isinstance(item.get("rule_hints"), list):
                item["extracted"].setdefault("rule_hints", item["rule_hints"])
                item.pop("rule_hints", None)
            return item

        payload = [_normalize(dict(obj)) for obj in raw_payload]
        tags = [TaggedSentence.model_validate(item) for item in payload]
        state["step1_tags"] = tags
        idx = repair_idx or msg_idx
        self.persistence.persist_turn(
            state["run_id"], idx, "step1", step_outputs=[t.model_dump() for t in tags]
        )
        self.persistence.log_event(
            state["run_id"],
            {"ts": datetime.utcnow().isoformat(), "event": "step1_tags"},
        )
        return state

    def _step2_spec_sheet(self, state: AgentState) -> AgentState:
        prior = {
            "step0_contract": state["step0_contract"].model_dump(mode="json"),
            "step1_tags": [t.model_dump(mode="json") for t in state["step1_tags"]],
        }
        content, msg_idx = self._call_llm(state, "step2", prior_outputs=prior)
        payload, repair_idx = self._parse_json_with_repair(
            content, state, "step2", prior, default_on_fail={}
        )

        def _fallback_spec() -> SpecSheet:
            return SpecSheet(
                sets=[],
                decisions=[],
                objective_terms=[],
                constraint_families=[],
                edge_cases=[],
                open_questions=[],
            )

        try:
            spec = SpecSheet.model_validate(payload)
        except ValidationError:
            # Try one format-repair round with validation hint
            repaired_text, repair_idx2 = self._call_llm(
                state,
                "step2",
                prior_outputs=prior,
                repair_mode="json",
                previous_response=json.dumps(payload),
            )
            payload2, repair_idx3 = self._parse_json_with_repair(
                repaired_text, state, "step2", prior, default_on_fail={}
            )
            try:
                spec = SpecSheet.model_validate(payload2)
                repair_idx = repair_idx3 or repair_idx2 or repair_idx
            except ValidationError:
                spec = _fallback_spec()
        state["step2_spec_sheet"] = spec
        idx = repair_idx or msg_idx
        self.persistence.persist_turn(
            state["run_id"], idx, "step2", step_outputs=spec
        )
        self.persistence.log_event(
            state["run_id"],
            {"ts": datetime.utcnow().isoformat(), "event": "step2_spec_sheet"},
        )
        return state

    def _step3_templates(self, state: AgentState) -> AgentState:
        prior = {
            "spec_sheet": state["step2_spec_sheet"].model_dump(mode="json"),
        }
        content, msg_idx = self._call_llm(state, "step3", prior_outputs=prior)
        payload, repair_idx = self._parse_json_with_repair(
            content, state, "step3", prior, default_on_fail=[]
        )
        templates = [ConstraintTemplate.model_validate(item) for item in payload]
        state["step3_templates"] = templates
        idx = repair_idx or msg_idx
        self.persistence.persist_turn(
            state["run_id"],
            idx,
            "step3",
            step_outputs=[t.model_dump(mode="json") for t in templates],
        )
        self.persistence.log_event(
            state["run_id"],
            {"ts": datetime.utcnow().isoformat(), "event": "step3_constraint_templates"},
        )
        return state

    def _step4_sanity(self, state: AgentState) -> AgentState:
        report = run_sanity_checks(
            state.get("step0_contract"),
            state.get("step2_spec_sheet"),
            state.get("step3_templates", []),
            data_profile=state.get("data_profile"),
        )
        state["step4_sanity_report"] = report
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"], idx, "step4", sanity_report=report
        )
        self.persistence.log_event(
            state["run_id"],
            {
                "ts": datetime.utcnow().isoformat(),
                "event": "step4_sanity",
                "overall_pass": report.overall_pass,
            },
        )
        if not report.overall_pass:
            # Route back by setting last_error; conditional edge handles loop
            state["last_error"] = "sanity_failed"
        return state

    def _step5_codegen(self, state: AgentState) -> AgentState:
        prior = {
            "spec_sheet": state["step2_spec_sheet"].model_dump(mode="json"),
            "constraint_templates": [
                t.model_dump(mode="json") for t in state.get("step3_templates", [])
            ],
            "sanity": state.get("step4_sanity_report").model_dump(
                mode="json", by_alias=True
            )
            if state.get("step4_sanity_report")
            else None,
        }
        content, msg_idx = self._call_llm(state, "step5", prior_outputs=prior)
        code_text = content
        repair_idx = None
        if "```" in code_text:
            code_text, repair_idx = self._call_llm(
                state,
                "step5",
                prior_outputs=prior,
                repair_mode="code",
                previous_response=code_text,
            )
        state.setdefault("code_versions", []).append(
            CodeVersion(step="step5", content=code_text)
        )
        self.persistence.persist_turn(
            state["run_id"], repair_idx or msg_idx, "step5", code=code_text
        )
        self.persistence.log_event(
            state["run_id"],
            {"ts": datetime.utcnow().isoformat(), "event": "step5_codegen"},
        )
        return state

    def _static_audit(self, state: AgentState) -> AgentState:
        code_text = state["code_versions"][-1].content
        audit = audit_script(code_text)
        state.setdefault("static_audit_reports", []).append(audit)
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"], idx, "static_audit", static_audit=audit
        )
        self.persistence.log_event(
            state["run_id"],
            {
                "ts": datetime.utcnow().isoformat(),
                "event": "static_audit",
                "passed": audit.passed,
            },
        )
        if not audit.passed:
            state["last_error"] = "static_audit_failed"
        return state

    def _step6_run_and_diagnose(self, state: AgentState) -> AgentState:
        code_text = state["code_versions"][-1].content
        try:
            solve_report, iis_report, stdout, stderr = run_script(
                code_text, state["data"]
            )
        except Exception as exc:  # pragma: no cover - defensive
            solve_report = SolveReport(
                status="RUNTIME_ERROR", stderr=str(exc), stdout="", exit_code=-1
            )
            iis_report = None
            stdout = ""
            stderr = str(exc)
            state["last_error"] = str(exc)
        state.setdefault("solve_reports", []).append(solve_report)
        if iis_report:
            state.setdefault("iis_reports", []).append(iis_report)
        idx = self._next_turn_index(state)
        self.persistence.persist_turn(
            state["run_id"],
            idx,
            "step6",
            stdout=stdout,
            stderr=stderr,
            solve_report=solve_report,
            iis_report=iis_report,
        )
        self.persistence.log_event(
            state["run_id"],
            {
                "ts": datetime.utcnow().isoformat(),
                "event": "step6_run",
                "status": solve_report.status,
            },
        )

        # Build repair brief via LLM if required
        needs_repair = solve_report.status not in ("2", "GRB.OPTIMAL", "OPTIMAL")
        audit_failed = (
            state.get("static_audit_reports") and not state["static_audit_reports"][-1].passed
        )
        if needs_repair or audit_failed:
            prior = {
                "spec_sheet": state.get("step2_spec_sheet").model_dump(mode="json")
                if state.get("step2_spec_sheet")
                else None,
                "constraint_templates": [
                    t.model_dump(mode="json") for t in state.get("step3_templates", [])
                ],
                "code": code_text,
                "solve_report": solve_report.model_dump(mode="json", by_alias=True),
                "iis_report": iis_report.model_dump(mode="json", by_alias=True) if iis_report else None,
                "static_audit": state["static_audit_reports"][-1].model_dump(mode="json", by_alias=True)
                if state.get("static_audit_reports")
                else None,
            }
            content, msg_idx = self._call_llm(
                state, "step6", prior_outputs=prior
            )
            brief_payload, repair_idx = self._parse_json_with_repair(
                content, state, "step6", prior
            )
            brief = RepairBrief.model_validate(brief_payload)
            state.setdefault("repair_briefs", []).append(brief)
            state["repair_count"] = state.get("repair_count", 0) + 1
            state["last_error"] = "needs_repair"
            idx_for_brief = repair_idx or msg_idx
            self.persistence.persist_turn(
                state["run_id"], idx_for_brief, "step6_repair_brief", step_outputs=brief
            )
        else:
            state["last_error"] = None
        return state

    def _route_after_audit(self, state: AgentState):
        if state.get("static_audit_reports") and not state["static_audit_reports"][-1].passed:
            return "retry_codegen"
        return "run"

    def _route_after_sanity(self, state: AgentState):
        report = state.get("step4_sanity_report")
        if report and not report.overall_pass:
            return "revise_spec"
        return "codegen"

    def _route_after_run(self, state: AgentState):
        if state.get("repair_count", 0) >= self.repair_limit:
            return "done"
        if state.get("repair_briefs"):
            brief = state["repair_briefs"][-1]
            if brief.target == "SPEC":
                return "revise_spec"
            if brief.target == "CODEGEN":
                return "regenerate_code"
        if state.get("last_error"):
            return "regenerate_code"
        return "done"


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
        "solve_reports": [],
        "iis_reports": [],
        "repair_briefs": [],
        "turn_index": 0,
    }
