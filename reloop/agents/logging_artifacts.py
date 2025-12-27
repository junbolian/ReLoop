import json
import os
import time
from typing import List, Dict, Any

from .agent_types import LLMMessage


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, records: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def save_text(path: str, content: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def log_message(path: str, round_id: int, message: LLMMessage, tokens: Dict[str, int]):
    append_jsonl(
        path,
        [
            {
                "round": round_id,
                "role": message.role,
                "content": message.content,
                "timestamp": time.time(),
                **tokens,
            }
        ],
    )
