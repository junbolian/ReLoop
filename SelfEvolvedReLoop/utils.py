from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, List

from .state import RunRecord


BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "memory"


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_run_id(text: str) -> str:
    """Timestamp + truncated hash to ensure uniqueness and traceability."""
    from datetime import datetime

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    h = sha256_of_text(text)[:8]
    return f"{ts}-{h}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_append_jsonl(path: Path, obj: Any) -> None:
    """Best-effort append-only JSONL writer."""
    ensure_dir(path.parent)
    line = json.dumps(_to_serializable(obj), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_jsonl(path: Path) -> List[Any]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(obj), f, indent=2)


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def memory_paths() -> dict:
    return {
        "runs_dir": MEMORY_DIR / "runs",
        "candidates": MEMORY_DIR / "candidates.jsonl",
        "trusted": MEMORY_DIR / "trusted.jsonl",
        "negative": MEMORY_DIR / "negative.jsonl",
    }


def summarize_run(record: RunRecord) -> str:
    status = record.final_status or (record.solve_result.status if record.solve_result else "unknown")
    obj = record.solve_result.objective if record.solve_result else None
    return f"run_id={record.run_id} archetype={record.archetype_id} status={status} objective={obj}"
