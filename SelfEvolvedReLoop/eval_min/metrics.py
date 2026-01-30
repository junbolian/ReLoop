from __future__ import annotations

from typing import List, Dict, Any
import math


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    feasibility = sum(1 for r in rows if r.get("audit_pass")) / n
    optimal = sum(1 for r in rows if str(r.get("status", "")).lower().startswith("optimal")) / n
    correct = sum(1 for r in rows if r.get("correct")) / n
    errors = [abs(r["abs_error"]) for r in rows if r.get("abs_error") is not None]
    mae = sum(errors) / len(errors) if errors else None
    runtimes = [r for r in (row.get("runtime_ms") for row in rows) if r is not None]
    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else None
    return {
        "n": n,
        "feasibility_rate": feasibility,
        "optimal_rate": optimal,
        "correctness_rate": correct,
        "mae": mae,
        "avg_runtime_ms": avg_runtime,
    }
