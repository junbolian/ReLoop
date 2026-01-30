from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_nl4opt_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_float_answer(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return float(x)
        except ValueError:
            return None
    return None
