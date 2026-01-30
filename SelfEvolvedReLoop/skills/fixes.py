from __future__ import annotations

import copy
import json
from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel

from ..state import AuditResult, Diagnosis, ExtractionConfig, FixRecord, RunRecord, RunState, SandboxReport, SolveResult, utc_now_iso
from ..utils import atomic_append_jsonl, load_jsonl, memory_paths, write_json
from .canonicalize import CanonicalizationSkill, CanonicalizeInput
from .extract import EntityExtractionInput, EntityExtractionSkill
from .build_model import ModelBuilderInput, ModelBuilderSkill
from .solve import SolveInput, SolveSkill
from .audit import AuditInput, AuditSkill


class FixProposeInput(BaseModel):
    archetype_id: str
    symptoms: List[str]
    diagnosis: Optional[Diagnosis] = None
    extraction_config: ExtractionConfig
    run_id: str


class FixProposeOutput(BaseModel):
    candidate: Optional[FixRecord] = None


class FixProposeSkill:
    """
    Purpose:
        Suggest a single patch-like fix for the observed symptoms.
    Determinism:
        Rule-based mapping; deterministic.
    Failure modes:
        If no applicable rule, returns None.
    Logging:
        Returns FixRecord with human_label=unknown.
    """

    def run(self, inp: FixProposeInput) -> FixProposeOutput:
        if inp.archetype_id not in {"diet_lp", "allocation_lp"}:
            return FixProposeOutput(candidate=None)

        fix: Optional[FixRecord] = None
        if "AUDIT_VIOLATION" in inp.symptoms or "extraction_malformed" in inp.symptoms:
            fix = FixRecord(
                fix_id=f"{inp.run_id}-prompt-strict",
                archetype_id=inp.archetype_id,
                symptom_tags=inp.symptoms,
                patch_type="prompt_patch",
                patch_payload={"prompt_version": "v2_strict", "strict_mode": True},
                preconditions=["LLM_available"],
                expected_effect="Improve extraction fidelity with stricter JSON prompt.",
                human_label="unknown",
                source_run_id=inp.run_id,
            )
        elif "slow_solve" in (inp.diagnosis.symptom_tags if inp.diagnosis else []):
            fix = FixRecord(
                fix_id=f"{inp.run_id}-canonicalize",
                archetype_id=inp.archetype_id,
                symptom_tags=inp.symptoms,
                patch_type="canonicalize_patch",
                patch_payload={"clip_negatives": True},
                preconditions=[],
                expected_effect="Strengthen normalization to stabilize solver.",
                human_label="unknown",
                source_run_id=inp.run_id,
            )
        return FixProposeOutput(candidate=fix)


class FixSandboxTestInput(BaseModel):
    state: RunState
    candidate: FixRecord


class FixSandboxTestOutput(BaseModel):
    report: SandboxReport


class FixSandboxTestSkill:
    """
    Purpose:
        Evaluate a candidate fix on a copy of the current run.
    Determinism:
        Deterministic given same skills and config; may call LLM if enabled.
    Failure modes:
        Extraction/model errors -> recommendation 'inconclusive'.
    Logging:
        Returns comparison of baseline vs candidate outcomes.
    """

    def run(self, inp: FixSandboxTestInput) -> FixSandboxTestOutput:
        baseline_obj = inp.state.solve_result.objective if inp.state.solve_result else None
        baseline_pass = inp.state.audit_result.passed if inp.state.audit_result else False

        sandbox_state = inp.state.model_copy(deep=True)
        self._apply_fix_to_state(sandbox_state, inp.candidate)

        # rerun extract -> canonicalize -> model -> solve -> audit
        extract_skill = EntityExtractionSkill()
        canon_skill = CanonicalizationSkill()
        build_skill = ModelBuilderSkill()
        solve_skill = SolveSkill()
        audit_skill = AuditSkill()

        extract_out = extract_skill.run(
            EntityExtractionInput(
                text=sandbox_state.input_text,
                archetype_id=sandbox_state.archetype_id,
                config=sandbox_state.extraction_config,
                data=sandbox_state.input_data,
            )
        )
        if extract_out.schema is None:
            report = SandboxReport(
                recommendation="inconclusive",
                baseline_objective=baseline_obj,
                candidate_objective=None,
                baseline_audit_pass=baseline_pass,
                candidate_audit_pass=False,
                comparison_notes=f"Extraction failed under sandbox: {extract_out.error}",
            )
            return FixSandboxTestOutput(report=report)

        canon = canon_skill.run(CanonicalizeInput(archetype_id=sandbox_state.archetype_id, schema=extract_out.schema)).canonical
        sandbox_state.canonical_schema = canon
        build_out = build_skill.run(ModelBuilderInput(archetype_id=sandbox_state.archetype_id, canonical_schema=canon))
        sandbox_state.model_handle = build_out.handle
        solve_out = solve_skill.run(SolveInput(handle=build_out.handle))
        sandbox_state.solve_result = solve_out.result
        audit_out = audit_skill.run(AuditInput(archetype_id=sandbox_state.archetype_id, canonical_schema=canon, handle=build_out.handle))

        candidate_pass = audit_out.audit.passed
        candidate_obj = sandbox_state.solve_result.objective

        # Simple decision: prefer passing audits and lower objective
        recommendation = "reject"
        if candidate_pass and (baseline_obj is None or candidate_obj is None or candidate_obj <= baseline_obj + 1e-6):
            recommendation = "apply"
        elif candidate_pass and not baseline_pass:
            recommendation = "apply"
        elif not baseline_pass and not candidate_pass:
            recommendation = "inconclusive"

        report = SandboxReport(
            recommendation=recommendation,
            baseline_objective=baseline_obj,
            candidate_objective=candidate_obj,
            baseline_audit_pass=baseline_pass,
            candidate_audit_pass=candidate_pass,
            comparison_notes="Sandbox evaluation complete.",
        )
        return FixSandboxTestOutput(report=report)

    def _apply_fix_to_state(self, state: RunState, fix: FixRecord) -> None:
        if fix.patch_type == "prompt_patch":
            payload = fix.patch_payload
            state.extraction_config.prompt_version = payload.get("prompt_version", state.extraction_config.prompt_version)
            if payload.get("strict_mode"):
                state.extraction_config.strict_mode = True
                state.extraction_config.use_llm = True
        elif fix.patch_type == "canonicalize_patch":
            # Currently only a placeholder; could adjust canonicalization flags
            state.warnings.append("Applied canonicalize_patch (no-op placeholder).")


class FixApplyInput(BaseModel):
    state: RunState
    fix: FixRecord


class FixApplyOutput(BaseModel):
    state: RunState


class FixApplySkill:
    """
    Purpose:
        Apply an approved fix to the main state.
    Determinism:
        Rule-based application of patch payload.
    Failure modes:
        Unknown patch types are ignored with a warning.
    Logging:
        Records applied fix in state.applied_fixes.
    """

    def run(self, inp: FixApplyInput) -> FixApplyOutput:
        state = inp.state
        applied = copy.deepcopy(inp.fix)
        if not self._validate_fix(applied, state):
            state.warnings.append("Rejected fix due to safety validation failure.")
            return FixApplyOutput(state=state)
        if applied.patch_type == "prompt_patch":
            payload = applied.patch_payload
            state.extraction_config.prompt_version = payload.get("prompt_version", state.extraction_config.prompt_version)
            state.extraction_config.strict_mode = payload.get("strict_mode", state.extraction_config.strict_mode)
            if payload.get("strict_mode"):
                state.extraction_config.use_llm = True
        elif applied.patch_type == "canonicalize_patch":
            state.warnings.append("Applied canonicalize_patch (no-op placeholder).")
        else:
            state.warnings.append(f"Unknown patch_type {applied.patch_type}")
        state.applied_fixes.append(applied)
        return FixApplyOutput(state=state)

    def _validate_fix(self, fix: FixRecord, state: RunState) -> bool:
        # Only allow known safe patch types
        if fix.patch_type not in {"prompt_patch", "canonicalize_patch"}:
            return False
        # Ensure objective/constraints not altered
        forbidden = {"change_objective", "relax_constraints"}
        if any(tag in fix.preconditions for tag in forbidden):
            return False
        return True


class FixMemoryStoreSkill:
    """
    Purpose:
        Persist run records and fix lifecycle artifacts in append-only JSONL.
    Determinism:
        Deterministic serialization using json.
    """

    def __init__(self):
        paths = memory_paths()
        self.runs_dir = paths["runs_dir"]
        self.candidates = paths["candidates"]
        self.trusted = paths["trusted"]
        self.negative = paths["negative"]

    def store_run(self, record: RunRecord) -> Path:
        path = Path(self.runs_dir) / f"{record.run_id}.json"
        write_json(path, record)
        return path

    def append_candidate(self, fix: FixRecord) -> None:
        fix.human_label = fix.human_label or "unknown"
        atomic_append_jsonl(self.candidates, fix)

    def append_trusted(self, fix: FixRecord) -> None:
        fix.human_label = "correct"
        atomic_append_jsonl(self.trusted, fix)

    def append_negative(self, fix: FixRecord) -> None:
        fix.human_label = "wrong"
        atomic_append_jsonl(self.negative, fix)


class FixMemoryRetrieveOutput(BaseModel):
    trusted_fixes: List[FixRecord]


class FixMemoryRetrieveSkill:
    """
    Purpose:
        Retrieve applicable trusted fixes for a new run.
    Determinism:
        Deterministic scoring based on tag overlap.
    """

    def __init__(self):
        self.paths = memory_paths()

    def run(self, archetype_id: str, symptom_tags: Optional[List[str]] = None, top_k: int = 3) -> FixMemoryRetrieveOutput:
        symptom_tags = symptom_tags or []
        records = []
        for line in load_jsonl(self.paths["trusted"]):
            try:
                records.append(FixRecord(**line))
            except Exception:
                continue
        # simple score: count overlap
        scored = []
        for rec in records:
            if rec.archetype_id != archetype_id:
                continue
            overlap = len(set(rec.symptom_tags) & set(symptom_tags))
            scored.append((overlap, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        trusted = [rec for _, rec in scored[:top_k]]
        return FixMemoryRetrieveOutput(trusted_fixes=trusted)


def promote_candidate_to_trusted(run_id: str, label: str, notes: Optional[str] = None) -> Optional[FixRecord]:
    """
    Promote a candidate fix (found in run record) to trusted/negative/skip.
    Returns the stored FixRecord or None if no candidate existed.
    """
    paths = memory_paths()
    run_path = Path(paths["runs_dir"]) / f"{run_id}.json"
    if not run_path.exists():
        return None
    data = json.loads(run_path.read_text(encoding="utf-8"))
    record = RunRecord(**data)
    cand = record.candidate_fix
    if cand is None:
        return None
    cand.human_label = label
    cand.notes = notes
    store = FixMemoryStoreSkill()
    if label == "correct":
        store.append_trusted(cand)
    elif label == "wrong":
        store.append_negative(cand)
    elif label == "skip":
        store.append_candidate(cand)
    return cand


__all__ = [
    "FixProposeSkill",
    "FixProposeInput",
    "FixProposeOutput",
    "FixSandboxTestSkill",
    "FixSandboxTestInput",
    "FixSandboxTestOutput",
    "FixApplySkill",
    "FixApplyInput",
    "FixApplyOutput",
    "FixMemoryStoreSkill",
    "FixMemoryRetrieveSkill",
    "FixMemoryRetrieveOutput",
    "promote_candidate_to_trusted",
]
