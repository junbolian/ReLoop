from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List

from .doc_loader import load_markdown, extract_sections, trim_to_token_budget, hash_text

BASE = Path(__file__).resolve().parent
MANIFEST_PATH = BASE / "skills_manifest.yaml"


def _load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _collect_docs(paths: List[str], section_titles: List[str], budget_tokens: int) -> Tuple[str, List[dict], int]:
    injected_parts = []
    used_tokens = 0
    for p in paths:
        text = load_markdown(str(BASE / p))
        if not text:
            continue
        sectioned = extract_sections(text, section_titles)
        trimmed, tokens = trim_to_token_budget(sectioned, budget_tokens - used_tokens)
        used_tokens += tokens
        injected_parts.append({"path": p, "sections": section_titles, "text": trimmed, "sha256": hash_text(trimmed)})
        if used_tokens >= budget_tokens:
            break
    final_text = "\n\n".join([part["text"] for part in injected_parts if part["text"]])
    return final_text, injected_parts, used_tokens


def assemble_prompt(skill_id: str, state: Any, override_budget: int | None = None) -> Tuple[str, dict]:
    manifest = _load_manifest()
    skill_entry = next(s for s in manifest["skills"] if s["id"] == skill_id)
    archetype_id = getattr(state, "archetype_id", None) or getattr(state, "route", None)
    arche_entries = manifest.get("archetypes", [])
    arche_doc_paths = []
    for a in arche_entries:
        if archetype_id in ([a["id"]] + a.get("aliases", [])):
            arche_doc_paths = a.get("doc_paths", [])
            break
    doc_paths = skill_entry.get("doc_paths", []) + arche_doc_paths
    prompt_template_path = skill_entry.get("prompt_template")
    template_text = ""
    if prompt_template_path:
        tpath = BASE / prompt_template_path
        if tpath.exists():
            template_text = tpath.read_text(encoding="utf-8")
    budget = override_budget or skill_entry.get("budget_tokens", 3000)
    docs_text, injected_meta, used_tokens = _collect_docs(
        doc_paths,
        section_titles=["Purpose", "When to use", "Inputs", "Outputs", "Hard constraints", "Examples"],
        budget_tokens=budget,
    )
    prompt_parts = [
        "You are an OR agent skill. Follow injected docs strictly.",
        docs_text,
        "TEMPLATE:",
        template_text,
        "SCENARIO:",
        getattr(state, "input_text", "") or "",
    ]
    prompt_text = "\n\n".join([p for p in prompt_parts if p])
    meta = {
        "skill_id": skill_id,
        "files": injected_meta,
        "budget_tokens": budget,
        "used_tokens_est": used_tokens,
        "sha256": hash_text(prompt_text),
    }
    return prompt_text, meta


def assemble_route_prompt(state, override_budget=None):
    return assemble_prompt("route", state, override_budget)


def assemble_extract_prompt(state, override_budget=None):
    return assemble_prompt("extract", state, override_budget)


def assemble_model_plan_prompt(state, override_budget=None):
    return assemble_prompt("model_plan", state, override_budget)


def assemble_diagnose_prompt(state, override_budget=None):
    return assemble_prompt("diagnose", state, override_budget)


def assemble_fix_propose_prompt(state, override_budget=None):
    return assemble_prompt("fix_propose", state, override_budget)
