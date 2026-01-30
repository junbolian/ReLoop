from __future__ import annotations

from typing import TypedDict, Optional, Any

try:
    from langgraph.graph import StateGraph, END
except ImportError:  # pragma: no cover - lightweight fallback
    END = "__END__"

    class _Compiled:
        def __init__(self, graph):
            self.graph = graph

        def invoke(self, state):
            current = self.graph.entry
            steps = 0
            while current != END and steps < 100:
                steps += 1
                func = self.graph.nodes[current]
                update = func(state)
                state.update(update)
                if current in self.graph.cond_edges:
                    chooser, mapping = self.graph.cond_edges[current]
                    branch = chooser(state)
                    current = mapping.get(branch, mapping.get("finalize", END))
                else:
                    current = self.graph.edges.get(current, END)
            return state

    class StateGraph:
        def __init__(self, _state_type):
            self.nodes = {}
            self.edges = {}
            self.cond_edges = {}
            self.entry = None

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def set_entry_point(self, name):
            self.entry = name

        def add_edge(self, src, dst):
            self.edges[src] = dst

        def add_conditional_edges(self, src, chooser, mapping):
            self.cond_edges[src] = (chooser, mapping)

        def compile(self):
            return _Compiled(self)

from .state import (
    AuditResult,
    ExtractionConfig,
    FixRecord,
    NotYetImplemented,
    RunRecord,
    RunState,
    SolveResult,
    utc_now_iso,
)
from .utils import generate_run_id, sha256_of_text
from .llm import get_llm_client
from .skills import (
    TaskRouterSkill,
    EntityExtractionSkill,
    CanonicalizationSkill,
    ModelBuilderSkill,
    SolveSkill,
    AuditSkill,
    DiagnoseSkill,
    FixProposeSkill,
    FixSandboxTestSkill,
    FixApplySkill,
    FixMemoryStoreSkill,
    FixMemoryRetrieveSkill,
)
from .skill_runtime import SkillRuntime
from .skills.router import TaskRouterInput
from .skills.extract import EntityExtractionInput
from .skills.canonicalize import CanonicalizeInput
from .skills.build_model import ModelBuilderInput
from .skills.solve import SolveInput
from .skills.audit import AuditInput
from .skills.diagnose import DiagnoseInput
from .skills.fixes import FixProposeInput, FixSandboxTestInput, FixApplyInput


class GraphState(TypedDict):
    run: RunState


class Controller:
    """LangGraph-based orchestrator for the OR agent pipeline."""

    def __init__(self, extractor: Optional[EntityExtractionSkill] = None, runtime_kwargs: Optional[dict] = None):
        self.router = TaskRouterSkill()
        self.extractor = extractor or EntityExtractionSkill()
        self.canonicalize = CanonicalizationSkill()
        self.builder = ModelBuilderSkill()
        self.solver = SolveSkill()
        self.audit = AuditSkill()
        self.diagnose = DiagnoseSkill()
        self.fix_propose = FixProposeSkill()
        self.sandbox = FixSandboxTestSkill()
        self.fix_apply = FixApplySkill()
        self.mem_store = FixMemoryStoreSkill()
        self.mem_retrieve = FixMemoryRetrieveSkill()
        self.runtime = SkillRuntime(**(runtime_kwargs or {}))
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("route", self._node_route)
        graph.add_node("extract", self._node_extract)
        graph.add_node("canonicalize", self._node_canonicalize)
        graph.add_node("retrieve_trusted", self._node_retrieve_trusted)
        graph.add_node("model_plan", self._node_model_plan)
        graph.add_node("build_model", self._node_build_model)
        graph.add_node("solve", self._node_solve)
        graph.add_node("audit", self._node_audit)
        graph.add_node("diagnose", self._node_diagnose)
        graph.add_node("propose_fix", self._node_propose_fix)
        graph.add_node("sandbox_test", self._node_sandbox_test)
        graph.add_node("apply_fix", self._node_apply_fix)
        graph.add_node("finalize", self._node_finalize)

        graph.set_entry_point("route")
        graph.add_edge("route", "extract")
        graph.add_conditional_edges("extract", self._after_extract, {"canonicalize": "canonicalize", "finalize": "finalize"})
        graph.add_edge("canonicalize", "retrieve_trusted")
        graph.add_edge("retrieve_trusted", "model_plan")
        graph.add_edge("model_plan", "build_model")
        graph.add_edge("build_model", "solve")
        graph.add_edge("solve", "audit")
        graph.add_conditional_edges("audit", self._after_audit, {"finalize": "finalize", "diagnose": "diagnose"})
        graph.add_edge("diagnose", "propose_fix")
        graph.add_edge("propose_fix", "sandbox_test")
        graph.add_conditional_edges(
            "sandbox_test",
            self._after_sandbox,
            {"apply_fix": "apply_fix", "finalize": "finalize"},
        )
        graph.add_edge("apply_fix", "build_model")  # rerun solve after applying fix
        graph.add_edge("finalize", END)
        return graph.compile()

    # Node implementations -------------------------------------------------
    def _node_route(self, state: GraphState) -> GraphState:
        run = state["run"]
        # rule-based first
        route = self.router.run(TaskRouterInput(text=run.input_text))
        if route.confidence < 0.7:
            llm_route = self.router.run_with_llm(TaskRouterInput(text=run.input_text), self.runtime, run.prompt_injections)
            route = llm_route
        run.route = route
        run.archetype_id = route.archetype_id
        return {"run": run}

    def _node_extract(self, state: GraphState) -> GraphState:
        run = state["run"]
        extraction_attempts = 0
        success = False
        config = run.extraction_config
        while extraction_attempts <= config.max_retries and not success:
            out = self.extractor.run(
                EntityExtractionInput(
                    text=run.input_text,
                    archetype_id=run.archetype_id,
                    config=config,
                    data=run.input_data,
                    prompt_injections=run.prompt_injections,
                )
            )
            extraction_attempts += 1
            if out.schema is not None:
                run.extracted_schema = out.schema
                success = True
            else:
                run.warnings.append(out.error or "Unknown extraction error")
                if config.strict_mode:
                    break
                # escalate to strict prompt for final attempt
                config.prompt_version = "v2_strict"
                config.strict_mode = True
                config.use_llm = config.use_llm  # keep flag
        if not success:
            run.symptoms.append("EXTRACT_PARSE_ERROR")
            run.final_status = "extraction_failed"
        run.extraction_config = config
        return {"run": run}

    def _after_extract(self, state: GraphState) -> str:
        run = state["run"]
        return "canonicalize" if run.extracted_schema is not None else "finalize"

    def _node_canonicalize(self, state: GraphState) -> GraphState:
        run = state["run"]
        canon = self.canonicalize.run(CanonicalizeInput(archetype_id=run.archetype_id, schema=run.extracted_schema)).canonical
        run.canonical_schema = canon
        if hasattr(canon, "warnings"):
            run.warnings.extend(getattr(canon, "warnings"))
        elif isinstance(canon, dict) and "warnings" in canon:
            run.warnings.extend(canon.get("warnings") or [])
        return {"run": run}

    def _node_retrieve_trusted(self, state: GraphState) -> GraphState:
        run = state["run"]
        fixes = self.mem_retrieve.run(archetype_id=run.archetype_id, symptom_tags=run.symptoms).trusted_fixes
        if run.extraction_config.auto_apply_trusted and fixes:
            # apply first trusted fix immediately
            fix = fixes[0]
            apply_skill = FixApplySkill()
            run = apply_skill.run(FixApplyInput(state=run, fix=fix)).state
            # re-extract and canonicalize to reflect prompt changes
            out = self.extractor.run(
                EntityExtractionInput(
                    text=run.input_text,
                    archetype_id=run.archetype_id,
                    config=run.extraction_config,
                    data=run.input_data,
                    prompt_injections=run.prompt_injections,
                )
            )
            if out.schema:
                run.extracted_schema = out.schema
                run.canonical_schema = self.canonicalize.run(
                    CanonicalizeInput(archetype_id=run.archetype_id, schema=out.schema)
                ).canonical
        return {"run": run}

    def _node_model_plan(self, state: GraphState) -> GraphState:
        run = state["run"]
        # Only use LLM if available
        if get_llm_client() and run.canonical_schema is not None:
            dummy_state = type(
                "S",
                (),
                {
                    "input_text": run.input_text,
                    "archetype_id": run.archetype_id,
                    "canonical_schema": run.canonical_schema,
                    "prompt_injections": run.prompt_injections,
                },
            )()
            result = self.runtime.run_llm_skill("model_plan", dummy_state)
            run.prompt_injections["model_plan"] = result.get("prompt_meta")
            content = result.get("content")
            if content:
                try:
                    import json
                    from .model_ir import IRPlan
                    run.model_plan = IRPlan.model_validate(json.loads(content)).model_dump()
                except Exception:
                    run.warnings.append("model_plan LLM parsing failed; falling back to code builder")
        return {"run": run}

    def _node_build_model(self, state: GraphState) -> GraphState:
        run = state["run"]
        build_out = self.builder.run(
            ModelBuilderInput(archetype_id=run.archetype_id, canonical_schema=run.canonical_schema, model_plan=run.model_plan)
        )
        run.model_handle = build_out.handle
        run.model_plan = build_out.model_plan
        if run.model_handle.metadata.get("missing_inputs"):
            run.symptoms.append("MISSING_INPUT_DATA")
        return {"run": run}

    def _node_solve(self, state: GraphState) -> GraphState:
        run = state["run"]
        solve_out = self.solver.run(SolveInput(handle=run.model_handle))
        run.solve_result = solve_out.result
        return {"run": run}

    def _node_audit(self, state: GraphState) -> GraphState:
        run = state["run"]
        if run.canonical_schema is None:
            run.audit_result = AuditResult(passed=False, symptom_tags=["CANONICALIZE_INCOMPLETE"], violations=[], audit_score=0.0)
            run.symptoms.append("CANONICALIZE_INCOMPLETE")
            return {"run": run}
        audit_out = self.audit.run(AuditInput(archetype_id=run.archetype_id, canonical_schema=run.canonical_schema, handle=run.model_handle))
        run.audit_result = audit_out.audit
        if not audit_out.audit.passed:
            run.symptoms.extend(audit_out.audit.symptom_tags)
        return {"run": run}

    def _after_audit(self, state: GraphState) -> str:
        run = state["run"]
        if run.solve_result and run.solve_result.status == "not_yet_implemented":
            return "finalize"
        if (
            run.audit_result
            and run.audit_result.passed
            and run.solve_result
            and run.solve_result.status.lower().startswith("optimal")
        ):
            return "finalize"
        return "diagnose"

    def _node_diagnose(self, state: GraphState) -> GraphState:
        run = state["run"]
        diag_out = self.diagnose.run(DiagnoseInput(handle=run.model_handle, solve_result=run.solve_result))
        run.diagnosis = diag_out.diagnosis
        run.symptoms.extend(diag_out.diagnosis.symptom_tags)
        # Optional LLM enhancement
        if get_llm_client():
            dummy_state = type(
                "S",
                (),
                {
                    "solve_result": run.solve_result,
                    "audit_result": run.audit_result,
                    "symptoms": run.symptoms,
                    "archetype_id": run.archetype_id,
                    "prompt_injections": run.prompt_injections,
                },
            )()
            result = self.runtime.run_llm_skill("diagnose", dummy_state)
            run.prompt_injections["diagnose"] = result.get("prompt_meta")
            content = result.get("content")
            if content:
                try:
                    import json
                    data = json.loads(content)
                    run.diagnosis.summary = data.get("summary", run.diagnosis.summary)
                    run.diagnosis.symptom_tags = list(set(run.diagnosis.symptom_tags + data.get("symptom_tags", [])))
                    run.diagnosis.recommended_params.update(data.get("recommended_params", {}))
                except Exception:
                    run.warnings.append("diagnose LLM parse failed")
        return {"run": run}

    def _node_propose_fix(self, state: GraphState) -> GraphState:
        run = state["run"]
        if not run.extraction_config.suggest_candidate:
            return {"run": run}
        # LLM-based propose
        if get_llm_client():
            dummy_state = type(
                "S",
                (),
                {
                    "archetype_id": run.archetype_id,
                    "symptoms": run.symptoms,
                    "diagnosis": run.diagnosis.model_dump() if run.diagnosis else {},
                    "prompt_injections": run.prompt_injections,
                },
            )()
            result = self.runtime.run_llm_skill("fix_propose", dummy_state)
            run.prompt_injections["fix_propose"] = result.get("prompt_meta")
            content = result.get("content")
            if content:
                try:
                    import json

                    data = json.loads(content)
                    if data:
                        from ..state import FixRecord

                        run.candidate_fix = FixRecord(**data)
                except Exception:
                    run.warnings.append("fix_propose LLM parse failed")
        else:
            prop = self.fix_propose.run(
                FixProposeInput(
                    archetype_id=run.archetype_id,
                    symptoms=run.symptoms,
                    diagnosis=run.diagnosis,
                    extraction_config=run.extraction_config,
                    run_id=run.run_id,
                )
            )
            run.candidate_fix = prop.candidate
        return {"run": run}

    def _node_sandbox_test(self, state: GraphState) -> GraphState:
        run = state["run"]
        if run.candidate_fix is None:
            return {"run": run}
        report = self.sandbox.run(FixSandboxTestInput(state=run, candidate=run.candidate_fix)).report
        run.sandbox_report = report
        return {"run": run}

    def _after_sandbox(self, state: GraphState) -> str:
        run = state["run"]
        if (
            run.candidate_fix
            and run.sandbox_report
            and run.sandbox_report.recommendation == "apply"
            and run.extraction_config.allow_candidate_auto_apply
        ):
            return "apply_fix"
        return "finalize"

    def _node_apply_fix(self, state: GraphState) -> GraphState:
        run = state["run"]
        if run.candidate_fix:
            run = self.fix_apply.run(FixApplyInput(state=run, fix=run.candidate_fix)).state
            # Re-run extraction/canonicalization to reflect new configuration
            out = self.extractor.run(
                EntityExtractionInput(
                    text=run.input_text,
                    archetype_id=run.archetype_id,
                    config=run.extraction_config,
                    data=run.input_data,
                    prompt_injections=run.prompt_injections,
                )
            )
            if out.schema:
                run.extracted_schema = out.schema
                run.canonical_schema = self.canonicalize.run(
                    CanonicalizeInput(archetype_id=run.archetype_id, schema=out.schema)
                ).canonical
        return {"run": run}

    def _node_finalize(self, state: GraphState) -> GraphState:
        run = state["run"]
        if not run.final_status:
            if run.solve_result and run.solve_result.status == "not_yet_implemented":
                run.final_status = "not_yet_implemented"
            elif (
                run.audit_result
                and run.audit_result.passed
                and run.solve_result
                and run.solve_result.status.lower().startswith("optimal")
            ):
                run.final_status = "success"
            else:
                run.final_status = "needs_fix"

        # Expected objective evaluation
        if run.expected_objective is not None and run.solve_result and run.solve_result.objective is not None:
            tol = run.expected_tolerance or 1e-6
            run.is_correct = (
                run.audit_result
                and run.audit_result.passed
                and run.solve_result.status.lower().startswith("optimal")
                and abs(run.solve_result.objective - run.expected_objective) <= tol
            )

        not_impl = None
        if run.final_status == "not_yet_implemented":
            missing_inputs = []
            if run.model_handle and run.model_handle.metadata.get("missing_inputs"):
                missing_inputs = run.model_handle.metadata.get("missing_inputs")
            missing_capabilities = []
            coverage = None
            if run.canonical_schema and hasattr(run.canonical_schema, "missing_capabilities"):
                missing_capabilities = getattr(run.canonical_schema, "missing_capabilities")
            if run.canonical_schema and hasattr(run.canonical_schema, "coverage_level"):
                coverage = getattr(run.canonical_schema, "coverage_level")
            not_impl = NotYetImplemented(
                archetype_id=run.archetype_id,
                missing_inputs=missing_inputs,
                missing_capabilities=missing_capabilities,
                coverage_level=coverage,
                suggestions=["Provide data_dict via --data-json"] if "data_dict" in missing_inputs else [],
            )
            run.not_implemented = not_impl

        model_plan_hash = None
        if run.model_plan:
            try:
                import json
                model_plan_hash = sha256_of_text(json.dumps(run.model_plan, sort_keys=True))
            except Exception:
                model_plan_hash = None
        run.model_plan_hash = model_plan_hash

        record = RunRecord(
            run_id=run.run_id,
            input_text=run.input_text,
            archetype_id=run.archetype_id,
            input_data=run.input_data,
            route=run.route,
            extraction_config=run.extraction_config,
            extracted_schema=run.extracted_schema,
            canonical_schema=run.canonical_schema,
            model_plan=run.model_plan,
            model_plan_hash=model_plan_hash,
            model_summary=run.model_plan if hasattr(run, "model_plan") else None,
            solve_result=run.solve_result,
            audit_result=run.audit_result,
            diagnosis=run.diagnosis,
            symptoms=run.symptoms,
            candidate_fix=run.candidate_fix,
            applied_fixes=run.applied_fixes,
            sandbox_report=run.sandbox_report,
            final_status=run.final_status,
            warnings=run.warnings,
            not_implemented=not_impl,
            expected_objective=run.expected_objective,
            expected_tolerance=run.expected_tolerance,
            is_correct=run.is_correct,
            prompt_injections=run.prompt_injections,
        )
        # Persist run and candidate fix lifecycle
        self.mem_store.store_run(record)
        if run.candidate_fix:
            self.mem_store.append_candidate(run.candidate_fix)
        return {"run": run}

    # Public API ----------------------------------------------------------
    def run(
        self,
        text: str,
        extraction_config: Optional[ExtractionConfig] = None,
        input_data: Optional[Any] = None,
        expected_objective: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> RunRecord:
        config = extraction_config or ExtractionConfig(use_llm=False)
        run_state = RunState(
            run_id=generate_run_id(text),
            input_text=text,
            extraction_config=config,
            input_data=input_data,
            expected_objective=expected_objective,
            expected_tolerance=tolerance,
        )
        result_state: GraphState = self.app.invoke({"run": run_state})
        final_run: RunState = result_state["run"]
        # Build and return fresh RunRecord mirroring finalize
        return RunRecord(
            run_id=final_run.run_id,
            input_text=final_run.input_text,
            archetype_id=final_run.archetype_id,
            input_data=final_run.input_data,
            route=final_run.route,
            extraction_config=final_run.extraction_config,
            extracted_schema=final_run.extracted_schema,
            canonical_schema=final_run.canonical_schema,
            model_plan=final_run.model_plan,
            model_plan_hash=final_run.model_plan_hash,
            model_summary=getattr(final_run, "model_plan", None),
            solve_result=final_run.solve_result,
            audit_result=final_run.audit_result,
            diagnosis=final_run.diagnosis,
            symptoms=final_run.symptoms,
            candidate_fix=final_run.candidate_fix,
            applied_fixes=final_run.applied_fixes,
            sandbox_report=final_run.sandbox_report,
            final_status=final_run.final_status,
            not_implemented=getattr(final_run, "not_implemented", None),
            expected_objective=final_run.expected_objective,
            expected_tolerance=final_run.expected_tolerance,
            is_correct=final_run.is_correct,
            warnings=final_run.warnings,
            prompt_injections=final_run.prompt_injections,
            created_at=utc_now_iso(),
        )


__all__ = ["Controller"]
