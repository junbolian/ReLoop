#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional


NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _is_number(text: str) -> bool:
    return bool(NUMBER_RE.match(text))


def _find_last_stdout(turns_dir: Path) -> Optional[Path]:
    candidates = []
    for child in turns_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            stdout_path = child / "stdout.txt"
            if stdout_path.exists():
                candidates.append((int(child.name), stdout_path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _parse_objective(stdout_path: Path) -> Optional[float]:
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    for line in text.splitlines():
        if line.startswith("SOLVE_JSON::"):
            payload = line.split("SOLVE_JSON::", 1)[1]
            try:
                data = json.loads(payload)
                obj = data.get("obj_val", data.get("objVal"))
                if obj is not None:
                    return float(obj)
            except Exception:
                break

    numeric_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _is_number(stripped):
            numeric_lines.append(stripped)
    if numeric_lines:
        try:
            return float(numeric_lines[-1])
        except Exception:
            return None
    return None


def _load_scenario_id(meta_path: Path) -> Optional[str]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    scenario_id = meta.get("scenario_id")
    if isinstance(scenario_id, str) and scenario_id.strip():
        return scenario_id.strip()
    return None


def summarize(root: Path, output_path: Path) -> None:
    rows = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        turns_dir = run_dir / "turns"
        if not meta_path.exists() or not turns_dir.exists():
            continue
        scenario_id = _load_scenario_id(meta_path)
        if not scenario_id:
            continue
        last_stdout = _find_last_stdout(turns_dir)
        obj_val = _parse_objective(last_stdout) if last_stdout else None
        rows.append(
            {
                "scenario_id": scenario_id,
                "objective": "null" if obj_val is None else obj_val,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id", "objective"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize scenario objectives from qwen3-max-result runs."
    )
    parser.add_argument(
        "--root",
        default="qwen3-max-result",
        help="Root directory containing run subfolders.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Defaults to <root>/objective_summary.csv",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_path = Path(args.out) if args.out else root / "objective_summary.csv"
    summarize(root, output_path)


if __name__ == "__main__":
    main()
