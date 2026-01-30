from __future__ import annotations

import yaml
from typing import Any, Dict

from .prompt_assembler import (
    assemble_route_prompt,
    assemble_extract_prompt,
    assemble_model_plan_prompt,
    assemble_diagnose_prompt,
    assemble_fix_propose_prompt,
)
from .llm import get_llm_client
from pathlib import Path

MANIFEST = yaml.safe_load((Path(__file__).parent / "skills_manifest.yaml").read_text(encoding="utf-8"))


class SkillRuntime:
    def __init__(self, budget_override: int | None = None, llm_model: str | None = None):
        self.budget_override = budget_override
        self.llm_model = llm_model

    def run_llm_skill(self, skill_id: str, state: Any) -> Dict[str, Any]:
        assembler = {
            "route": assemble_route_prompt,
            "extract": assemble_extract_prompt,
            "model_plan": assemble_model_plan_prompt,
            "diagnose": assemble_diagnose_prompt,
            "fix_propose": assemble_fix_propose_prompt,
        }.get(skill_id)
        if assembler is None:
            return {}
        prompt_text, meta = assembler(state, override_budget=self.budget_override)
        # store injection metadata even if no LLM call
        if getattr(state, "prompt_injections", None) is not None:
            state.prompt_injections[skill_id] = meta
        llm = get_llm_client(model_name=self.llm_model, temperature=0.0)
        if llm is None:
            return {"prompt_meta": meta, "content": None}
        resp = llm.invoke(prompt_text)
        content = getattr(resp, "content", str(resp))
        return {"prompt_meta": meta, "content": content}

    def get_skill_entry(self, skill_id: str) -> dict:
        return next(s for s in MANIFEST["skills"] if s["id"] == skill_id)
