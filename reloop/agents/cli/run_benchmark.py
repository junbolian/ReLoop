from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import List

from ..llm_client import build_llm_client
from ..orchestrator_graph import AgentOrchestrator
from ..prompt_stack import PromptStack
from ..schemas import AgentStateModel
from ..tools.persistence import PersistenceManager
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
    args = parser.parse_args()

    suite_path = Path(args.suite)
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite file not found: {suite_path}")
    scenario_ids = _load_scenarios_from_suite(suite_path)
    if args.limit:
        scenario_ids = scenario_ids[: args.limit]

    llm = build_llm_client("mock" if args.mock_llm else "openai", model=args.model)
    persistence = PersistenceManager(Path(args.out))

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
        print(f"{scenario_id}: completed -> {artifact_dir}")


if __name__ == "__main__":
    main()
