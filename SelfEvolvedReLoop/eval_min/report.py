from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json


def write_report(out_dir: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # summary json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # results jsonl
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    # markdown report
    lines = ["# NL4OPT Minimal Evaluation", "", "## Summary"]
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Results (first 10)")
    lines.append("| idx | status | objective | expected | abs_error | correct |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows[:10]:
        lines.append(
            f"| {row.get('idx')} | {row.get('status')} | {row.get('objective')} | {row.get('expected_obj')} | {row.get('abs_error')} | {row.get('correct')} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
