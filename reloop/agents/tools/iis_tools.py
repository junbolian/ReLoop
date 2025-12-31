from __future__ import annotations

from collections import Counter
from typing import Dict, List


def summarize_prefix_counts(constraint_names: List[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for name in constraint_names:
        prefix = name.split("_")[0] if "_" in name else name
        counts[prefix] += 1
    return dict(counts)
