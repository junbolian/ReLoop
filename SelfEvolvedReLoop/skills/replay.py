from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel

from ..state import RunRecord, ExtractionConfig
from ..utils import memory_paths
from .canonicalize import CanonicalizationSkill, CanonicalizeInput
from .extract import EntityExtractionInput, EntityExtractionSkill
from .build_model import ModelBuilderInput, ModelBuilderSkill
from .solve import SolveInput, SolveSkill
from .audit import AuditInput, AuditSkill
from .fixes import FixMemoryRetrieveSkill


class ReplayReport(BaseModel):
    archetype_id: str
    total_runs: int
    pass_rate: float
    avg_objective: float = None
    avg_runtime: float = None
    details: List[Dict]


class ReplayValidationSkill:
    """
    Purpose:
        Replay historical runs to validate regression safety and fix quality.
    Determinism:
        Disables LLM usage; relies on stored schemas when available.
    """

    def __init__(self):
        self.paths = memory_paths()
        self.retrieve = FixMemoryRetrieveSkill()

    def run(self, archetype_id: str, max_runs: int = 20) -> ReplayReport:
        run_dir = Path(self.paths["runs_dir"])
        files = sorted(run_dir.glob("*.json"), reverse=True)[:max_runs]
        details = []
        objs = []
        runtimes = []
        pass_count = 0

        trusted_fixes = self.retrieve.run(archetype_id, symptom_tags=[]).trusted_fixes

        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            record = RunRecord(**data)
            if record.archetype_id != archetype_id:
                continue
            detail = {"run_id": record.run_id}
            config = record.extraction_config
            config.use_llm = False  # deterministic replay
            # Apply trusted fixes
            for fix in trusted_fixes:
                if fix.patch_type == "prompt_patch":
                    payload = fix.patch_payload
                    config.prompt_version = payload.get("prompt_version", config.prompt_version)
                    if payload.get("strict_mode"):
                        config.strict_mode = True
                        config.use_llm = True  # allowed if key present; still deterministic with temp=0
            extract_skill = EntityExtractionSkill()
            canon_skill = CanonicalizationSkill()
            build_skill = ModelBuilderSkill()
            solve_skill = SolveSkill()
            audit_skill = AuditSkill()

            if record.extracted_schema:
                extract_out_schema = record.extracted_schema
            else:
                extract_out = extract_skill.run(
                    EntityExtractionInput(text=record.input_text, archetype_id=archetype_id, config=config, data=record.input_data if hasattr(record, "input_data") else None)
                )
                extract_out_schema = extract_out.schema
            if extract_out_schema is None:
                detail["status"] = "extract_failed"
                details.append(detail)
                continue
            canonical = canon_skill.run(CanonicalizeInput(archetype_id=archetype_id, schema=extract_out_schema)).canonical
            build_out = build_skill.run(
                ModelBuilderInput(archetype_id=archetype_id, canonical_schema=canonical, model_plan=record.model_plan)
            )
            solve_out = solve_skill.run(SolveInput(handle=build_out.handle))
            audit_out = audit_skill.run(AuditInput(archetype_id=archetype_id, canonical_schema=canonical, handle=build_out.handle))
            passed = audit_out.audit.passed
            if passed:
                pass_count += 1
            if solve_out.result.objective is not None:
                objs.append(solve_out.result.objective)
            if solve_out.result.runtime is not None:
                runtimes.append(solve_out.result.runtime)
            detail.update(
                {
                    "status": solve_out.result.status,
                    "objective": solve_out.result.objective,
                    "audit_pass": passed,
                }
            )
            details.append(detail)

        total = len(details)
        avg_obj = sum(objs) / len(objs) if objs else None
        avg_runtime = sum(runtimes) / len(runtimes) if runtimes else None
        pass_rate = pass_count / total if total else 0.0
        return ReplayReport(
            archetype_id=archetype_id,
            total_runs=total,
            pass_rate=pass_rate,
            avg_objective=avg_obj,
            avg_runtime=avg_runtime,
            details=details,
        )


__all__ = ["ReplayValidationSkill", "ReplayReport"]
