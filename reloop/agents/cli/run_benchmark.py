from __future__ import annotations

import argparse
import csv
import json
import uuid
from pathlib import Path
from typing import List

from ..llm_client import build_llm_client
from ..orchestrator_graph import AgentOrchestrator
from ..prompt_stack import PromptStack
from ..schemas import AgentStateModel
from ..tools.persistence import PersistenceManager
from ..tools.sft_exporter import (
    export_codegen_jsonl,
    export_pre_codegen_jsonl,
    init_jsonl_paths,
)
from .run_one import _extract_scenario_text, _load_text, _resolve_prompt_path


def _load_scenarios_from_suite(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run a suite of scenarios through the agent orchestrator."
    )
    parser.add_argument("--suite", required=True, help="Path to a text file of scenario ids.")
    parser.add_argument("--out", default="artifacts", help="Output root for artifacts.")
    parser.add_argument("--model", default=None, help="LLM model name.")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios.")
    parser.add_argument("--repair-limit", type=int, default=5, help="Maximum repair iterations.")
    parser.add_argument("--max-turns", type=int, default=8, help="Hard cap on agent turns.")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite file not found: {suite_path}")
    scenario_ids = _load_scenarios_from_suite(suite_path)
    if args.limit:
        scenario_ids = scenario_ids[: args.limit]

    mode = "openai"
    if args.mock_llm:
        mode = "mock"
    elif args.model and (Path(args.model).exists() or "/" in args.model):
        mode = "local"

    llm = build_llm_client(mode, model=args.model)
    persistence = PersistenceManager(Path(args.out))
    pre_codegen_path = Path(args.out) / "sft_pre_codegen.jsonl"
    codegen_path = Path(args.out) / "sft_codegen.jsonl"
    init_jsonl_paths(pre_codegen_path, codegen_path)
    objective_rows = []

    for scenario_id in scenario_ids:
        data_path = Path("scenarios") / "data" / f"{scenario_id}.json"
        if not data_path.exists():
            print(f"[WARN] Skipping missing data file: {data_path}")
            continue
        with data_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        prompt_path = _resolve_prompt_path(scenario_id)
        if prompt_path is None:
            print(f"[WARN] Skipping {scenario_id}: no prompt found.")
            continue
        base_prompt_text = _load_text(str(prompt_path))
        scenario_text = _extract_scenario_text(base_prompt_text)

        step_prompts_dir = Path(__file__).resolve().parent.parent / "step_prompts"
        prompt_stack = PromptStack(base_prompt_text, step_prompts_dir)
        orchestrator = AgentOrchestrator(
            llm_client=llm,
            prompt_stack=prompt_stack,
            persistence=persistence,
            repair_limit=args.repair_limit,
            max_turns=args.max_turns,
        )

        state = {
            "run_id": uuid.uuid4().hex,
            "scenario_id": scenario_id,
            "base_prompt_hash": prompt_stack.base_prompt_hash,
            "data": data,
            "scenario_text": scenario_text,
            "base_prompt": base_prompt_text,
            "repair_count": 0,
            "conversation_log": [],
            "code_versions": [],
            "static_audit_reports": [],
            "solve_reports": [],
            "iis_reports": [],
            "repair_briefs": [],
            "turn_index": 0,
        }
        AgentStateModel.model_validate(state)
        final_state = orchestrator.run(state)
        artifact_dir = Path(args.out) / state["run_id"]
        export_pre_codegen_jsonl(
            conversation_log=final_state.get("conversation_log", []),
            path=pre_codegen_path,
            scenario_id=scenario_id,
            run_id=state["run_id"],
        )
        export_codegen_jsonl(
            codegen_conversation=final_state.get("codegen_conversation"),
            conversation_log=final_state.get("conversation_log", []),
            path=codegen_path,
            scenario_id=scenario_id,
            run_id=state["run_id"],
        )
        obj_val = None
        if final_state.get("solve_reports"):
            last_solve = final_state["solve_reports"][-1]
            if last_solve.status in ("2", "GRB.OPTIMAL", "OPTIMAL"):
                obj_val = last_solve.obj_val
        objective_rows.append(
            {
                "scenario_id": scenario_id,
                "objective": "null" if obj_val is None else obj_val,
            }
        )
        print(f"{scenario_id}: completed -> {artifact_dir}")

    csv_path = Path(args.out) / "benchmark_objectives.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id", "objective"])
        writer.writeheader()
        writer.writerows(objective_rows)


if __name__ == "__main__":
    main()
