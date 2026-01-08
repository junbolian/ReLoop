from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

from langchain_core.messages import BaseMessage

from ..schemas import ConversationTurn


def _normalize_role(role: str) -> str:
    if role == "human":
        return "user"
    return role


def _messages_to_qwen3_text(messages: Iterable[dict[str, str]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = _normalize_role(msg.get("role", "user"))
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def _base_messages_to_dicts(messages: Iterable[BaseMessage]) -> List[dict[str, str]]:
    out: List[dict[str, str]] = []
    for msg in messages:
        msg_type = getattr(msg, "type", getattr(msg, "role", ""))
        if msg_type == "ai":
            role = "assistant"
        elif msg_type == "system":
            role = "system"
        elif msg_type == "human":
            role = "user"
        else:
            role = str(msg_type) if msg_type else "user"
        out.append({"role": role, "content": getattr(msg, "content", "")})
    return out


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _ensure_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def export_pre_codegen_jsonl(
    conversation_log: Iterable[ConversationTurn],
    path: Path,
    scenario_id: str,
    run_id: str,
    steps: Optional[set[str]] = None,
) -> None:
    steps = steps or {"step1", "step2", "step3"}
    for idx, turn in enumerate(conversation_log):
        if turn.step not in steps:
            continue
        messages = [{"role": m.role, "content": m.content} for m in turn.messages]
        record = {
            "id": f"{run_id}:{turn.step}:{idx}",
            "scenario_id": scenario_id,
            "run_id": run_id,
            "phase": "pre_codegen",
            "step": turn.step,
            "text": _messages_to_qwen3_text(messages),
        }
        _append_jsonl(path, record)


def export_codegen_jsonl(
    codegen_conversation: Optional[Iterable[BaseMessage]],
    conversation_log: Iterable[ConversationTurn],
    path: Path,
    scenario_id: str,
    run_id: str,
) -> None:
    messages: Optional[List[dict[str, str]]] = None
    if codegen_conversation:
        messages = _base_messages_to_dicts(codegen_conversation)
    else:
        last_turn: Optional[ConversationTurn] = None
        for turn in conversation_log:
            if turn.step == "step4":
                last_turn = turn
        if last_turn:
            messages = [
                {"role": m.role, "content": m.content} for m in last_turn.messages
            ]
    if not messages:
        return
    record = {
        "id": f"{run_id}:step4:codegen",
        "scenario_id": scenario_id,
        "run_id": run_id,
        "phase": "codegen",
        "step": "step4",
        "text": _messages_to_qwen3_text(messages),
    }
    _append_jsonl(path, record)


def init_jsonl_paths(*paths: Path) -> None:
    for path in paths:
        _ensure_empty(path)
