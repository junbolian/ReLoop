from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseModel, ValidationError
import re

from ..llm import get_llm
from ..skill_runtime import SkillRuntime
from ..state import (
    AllocationSchema,
    AllocationItem,
    TransportFlowSchema,
    AssignmentSchema,
    FacilityLocationSchema,
    RetailInventorySchema,
    ExtractionConfig,
)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class EntityExtractionInput(BaseModel):
    text: str
    archetype_id: str
    config: ExtractionConfig
    data: Optional[Dict[str, Any]] = None  # for retail/injected data
    prompt_injections: Optional[Dict[str, Any]] = None


class EntityExtractionOutput(BaseModel):
    schema: Optional[Any] = None
    warnings: list[str] = []
    error: Optional[str] = None


class EntityExtractionSkill:
    """
    Purpose:
        Extract structured schema from text according to archetype.
    Determinism:
        Uses LLM when available (temperature=0). Falls back to deterministic JSON
        parsing when no key is present.
    Failure modes:
        Malformed JSON -> returns error string; caller may retry with strict
        prompt or stop gracefully.
    Logging:
        Returns warnings and error fields for audit trail.
    """

    def __init__(self):
        self.llm = None
        self.runtime = SkillRuntime()

    def run(self, inp: EntityExtractionInput) -> EntityExtractionOutput:
        archetype = inp.archetype_id
        if archetype == "diet_lp":
            archetype = "allocation_lp"

        self.llm = get_llm(temperature=0.0) if inp.config.use_llm else None

        if archetype == "allocation_lp":
            content = self._llm_or_text("extract", inp) if self.llm else inp.text
            parsed = self._parse_content(content, AllocationSchema)
            if parsed.schema is None:
                det_schema = _deterministic_allocation_from_nl4opt(inp.text)
                if det_schema:
                    return EntityExtractionOutput(schema=det_schema, warnings=["deterministic_fallback_used"])
            return parsed
        if archetype == "transport_flow":
            return self._parse_content(inp.text, TransportFlowSchema)
        if archetype == "assignment":
            return self._parse_content(inp.text, AssignmentSchema)
        if archetype == "facility_location":
            return self._parse_content(inp.text, FacilityLocationSchema)
        if archetype == "retail_inventory":
            schema = RetailInventorySchema(data=inp.data, missing_inputs=[] if inp.data else ["data_dict"])
            return EntityExtractionOutput(schema=schema)
        return EntityExtractionOutput(error=f"Archetype {archetype} not implemented")

    def _llm_or_text(self, skill_id: str, inp: EntityExtractionInput) -> str:
        # Use skill runtime to enforce progressive disclosure and capture metadata
        dummy_state = type(
            "S",
            (),
            {
                "input_text": inp.text,
                "archetype_id": inp.archetype_id,
                "prompt_injections": {},
            },
        )()
        result = self.runtime.run_llm_skill(skill_id, dummy_state)
        content = result.get("content")
        if isinstance(inp.prompt_injections, dict):
            inp.prompt_injections[skill_id] = result.get("prompt_meta")
        if content:
            return content
        return inp.text

    def _parse_content(self, content: str, schema_cls) -> EntityExtractionOutput:
        raw = _extract_json_block(content)
        if raw is None:
            # maybe content already JSON dict string? try direct
            try:
                data = json.loads(content)
                raw = content
            except Exception:
                return EntityExtractionOutput(error="No JSON found in extraction output")
        try:
            data = json.loads(raw)
            schema = schema_cls(**data)
            return EntityExtractionOutput(schema=schema)
        except (json.JSONDecodeError, ValidationError):
            # retry parsing original text as deterministic fallback
            try:
                data = json.loads(content)
                schema = schema_cls(**data)
                return EntityExtractionOutput(schema=schema, warnings=["LLM parse failed; used raw text"])
            except Exception as exc:
                return EntityExtractionOutput(error=f"Failed to parse schema: {exc}")


def _load_prompt(version: str) -> str:
    path = PROMPT_DIR / f"diet_extract_{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt {version} not found at {path}")
    return path.read_text(encoding="utf-8")


def _extract_json_block(text: str) -> Optional[str]:
    if text.strip().startswith("{"):
        return text
    # naive fenced code block search
    marker = "```"
    if marker in text:
        parts = text.split(marker)
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                return part
    # last resort search for first { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _deterministic_allocation_from_nl4opt(text: str) -> Optional[AllocationSchema]:
    # Heuristic parser for NL4OPT diet-style descriptions
    item_pattern = re.compile(
        r"-\s*(?P<name>[A-Za-z0-9_ ]+):.*?(?P<prot>\d+(?:\.\d+)?)\s*gram[s]?\s*of protein.*?"
        r"(?P<carb>\d+(?:\.\d+)?)\s*gram[s]?\s*of carbohydrates.*?"
        r"(?P<cal>\d+(?:\.\d+)?)\s*calories.*?\$(?P<cost>\d+(?:\.\d+)?)",
        re.IGNORECASE | re.DOTALL,
    )
    items = []
    for m in item_pattern.finditer(text):
        items.append(
            {
                "name": m.group("name").strip(),
                "cost": float(m.group("cost")),
                "features": {
                    "protein": float(m.group("prot")),
                    "carbohydrates": float(m.group("carb")),
                    "calories": float(m.group("cal")),
                },
            }
        )
    if not items:
        return None
    reqs = {}
    def _req(key, pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            reqs[key] = float(m.group(1))
    _req("protein", r"at least\s*(\d+(?:\.\d+)?)\s*grams of protein")
    _req("carbohydrates", r"at least\s*(\d+(?:\.\d+)?)\s*grams of carbohydrates")
    _req("calories", r"at least\s*(\d+(?:\.\d+)?)\s*calories")
    if "carbohydrates" not in reqs or "calories" not in reqs:
        combo = re.search(
            r"at least\s*(\d+(?:\.\d+)?)\s*grams of protein[^0-9]+?"
            r"(\d+(?:\.\d+)?)\s*grams of carbohydrates[^0-9]+?"
            r"(\d+(?:\.\d+)?)\s*calories",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if combo:
            reqs.setdefault("protein", float(combo.group(1)))
            reqs.setdefault("carbohydrates", float(combo.group(2)))
            reqs.setdefault("calories", float(combo.group(3)))
    if not reqs:
        return None
    try:
        return AllocationSchema(items=[AllocationItem(**it) for it in items], requirements=reqs)
    except Exception:
        return None


__all__ = ["EntityExtractionSkill", "EntityExtractionInput", "EntityExtractionOutput"]
