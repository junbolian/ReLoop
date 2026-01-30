from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None


def load_markdown(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def extract_sections(md_text: str, section_titles: List[str]) -> str:
    if not section_titles:
        return md_text
    lines = md_text.splitlines()
    selected = []
    current_title = None
    collecting = False
    titles_lower = [t.lower() for t in section_titles]
    for line in lines:
        if line.startswith("#"):
            current_title = line.lstrip("# ").strip().lower()
            collecting = current_title in titles_lower
            if collecting:
                selected.append(line)
            continue
        if collecting:
            selected.append(line)
    return "\n".join(selected).strip()


def _estimate_tokens(text: str) -> int:
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def trim_to_token_budget(text: str, budget_tokens: int) -> Tuple[str, int]:
    if budget_tokens <= 0:
        return "", 0
    est = _estimate_tokens(text)
    if est <= budget_tokens:
        return text, est
    # approximate trimming by character proportion
    ratio = budget_tokens / float(est)
    keep_chars = int(len(text) * ratio)
    trimmed = text[: keep_chars]
    return trimmed, _estimate_tokens(trimmed)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
