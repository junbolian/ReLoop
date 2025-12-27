from collections import defaultdict
from typing import List, Dict, Any


def cluster_iis(iis_constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
    clusters = defaultdict(list)
    for item in iis_constraints:
        name = item.get("name", "")
        prefix = name.split("_")[0] if name else "unknown"
        clusters[prefix].append(item)
    cluster_summaries = []
    for prefix, items in clusters.items():
        cluster_summaries.append(
            {"prefix": prefix, "count": len(items), "examples": items[:3]}
        )
    cluster_summaries.sort(key=lambda x: x["count"], reverse=True)
    return {"clusters": cluster_summaries}
