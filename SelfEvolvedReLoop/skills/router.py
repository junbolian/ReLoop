from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from ..state import RouteResult
from ..skill_runtime import SkillRuntime


class TaskRouterInput(BaseModel):
    text: str


class TaskRouterOutput(RouteResult):
    pass


class TaskRouterSkill:
    """
    Purpose:
        Route a free-form request to a supported archetype.
    Inputs:
        TaskRouterInput(text)
    Outputs:
        TaskRouterOutput(archetype_id, confidence, rationale)
    Failure modes:
        Very low confidence; falls back to retail_inventory placeholder.
    Determinism:
        Rule-based; no LLM usage.
    Logging:
        Returns structured rationale for audit.
    """

    def run(self, inp: TaskRouterInput) -> TaskRouterOutput:
        text_lower = inp.text.lower()
        if any(k in text_lower for k in ["diet", "meal", "nutrient", "nutrition", "allocation"]):
            return TaskRouterOutput(
                archetype_id="diet_lp",
                confidence=0.82,
                rationale="Detected allocation/dietary terms.",
            )
        if any(k in text_lower for k in ["transport", "shipping", "supply", "demand balance", "flow"]):
            return TaskRouterOutput(
                archetype_id="transport_flow",
                confidence=0.8,
                rationale="Detected transport/flow terminology.",
            )
        if any(k in text_lower for k in ["assignment", "matching", "worker", "task"]):
            return TaskRouterOutput(
                archetype_id="assignment",
                confidence=0.75,
                rationale="Detected bipartite assignment terminology.",
            )
        if any(k in text_lower for k in ["facility", "warehouse", "open cost", "customer service"]):
            return TaskRouterOutput(
                archetype_id="facility_location",
                confidence=0.7,
                rationale="Detected facility location terminology.",
            )
        if any(k in text_lower for k in ["retail", "inventory", "stock", "store", "sku"]):
            return TaskRouterOutput(
                archetype_id="retail_inventory",
                confidence=0.7,
                rationale="Detected retail/inventory terminology.",
            )
        return TaskRouterOutput(
            archetype_id="retail_inventory",
            confidence=0.5,
            rationale="Fallback to retail_inventory placeholder.",
        )

    def run_with_llm(self, inp: TaskRouterInput, runtime: SkillRuntime, prompt_injections: dict) -> TaskRouterOutput:
        base = self.run(inp)
        if base.confidence >= 0.7:
            return base
        # attempt LLM routing
        dummy_state = type(
            "S", (), {"input_text": inp.text, "archetype_id": base.archetype_id, "prompt_injections": prompt_injections}
        )()
        result = runtime.run_llm_skill("route", dummy_state)
        prompt_injections["route"] = result.get("prompt_meta")
        content = result.get("content")
        if not content:
            return base
        try:
            import json

            data = json.loads(content)
            return TaskRouterOutput(
                archetype_id=data.get("archetype_id", base.archetype_id),
                confidence=float(data.get("confidence", base.confidence)),
                rationale=data.get("rationale", base.rationale),
                missing_inputs=data.get("missing_inputs", []),
            )
        except Exception:
            return base


__all__ = ["TaskRouterSkill", "TaskRouterInput", "TaskRouterOutput"]
