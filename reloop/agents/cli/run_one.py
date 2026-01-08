from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Optional

from ..llm_client import build_llm_client
from ..orchestrator_graph import AgentOrchestrator
from ..prompt_stack import PromptStack
from ..schemas import AgentStateModel
from ..tools.persistence import PersistenceManager
from ..tools.sft_exporter import export_codegen_jsonl, export_pre_codegen_jsonl


def _load_text(path_or_text: str) -> str:
    path = Path(path_or_text)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return path_or_text


def _extract_scenario_text(prompt_text: str) -> str:
    marker = "Business narrative:"
    if marker in prompt_text:
        return prompt_text.split(marker, 1)[1].strip()
    return prompt_text


def _resolve_prompt_path(scenario_id: str) -> Optional[Path]:
    prompts_dir = Path("scenarios") / "prompts"
    candidates = [
        prompts_dir / f"{scenario_id}.base.txt",      # ReLoop Agent
        prompts_dir / f"{scenario_id}.scenario.txt",  # Zero-shot
        prompts_dir / f"{scenario_id}.txt",           
        prompts_dir / f"{scenario_id}.user.txt",
        prompts_dir / "system_prompt.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Run one scenario through the agent pipeline.")
    parser.add_argument("--scenario", required=True, help="Scenario id or path to scenario JSON.")
    parser.add_argument(
        "--base-prompt",
        required=False,
        help="Path to base prompt text or literal string. Defaults to scenario prompt file.",
    )
    parser.add_argument("--out", default="artifacts", help="Output root for artifacts.")
    parser.add_argument("--model", default=None, help="LLM model name (provider-specific).")
    parser.add_argument(
        "--mock-llm", action="store_true", help="Use a mock LLM that echoes the prompt."
    )
    parser.add_argument(
        "--repair-limit", type=int, default=5, help="Maximum repair iterations."
    )
    parser.add_argument(
        "--max-turns", type=int, default=12, help="Hard cap on total agent turns."
    )
    parser.add_argument(
        "--no-probes", action="store_true", help="Disable semantic probe verification."
    )
    args = parser.parse_args()

    scenario_arg = args.scenario
    data_path = Path(scenario_arg)
    if data_path.is_dir():
        raise ValueError("Scenario must be a file or id, not a directory.")

    if data_path.exists():
        if data_path.suffix != ".json":
            scenario_id = data_path.stem
            data_path = Path("scenarios") / "data" / f"{scenario_id}.json"
        else:
            scenario_id = data_path.stem
    else:
        scenario_id = scenario_arg
        data_path = Path("scenarios") / "data" / f"{scenario_id}.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find scenario data at {data_path}")

    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    prompt_path: Optional[Path] = None
    if args.base_prompt:
        base_prompt_text = _load_text(args.base_prompt)
    else:
        prompt_path = _resolve_prompt_path(scenario_id)
        if prompt_path is None:
            raise FileNotFoundError("No base prompt found for scenario.")
        base_prompt_text = prompt_path.read_text(encoding="utf-8")

    scenario_text = _extract_scenario_text(base_prompt_text)

    step_prompts_dir = Path(__file__).resolve().parent.parent / "step_prompts"
    prompt_stack = PromptStack(base_prompt=base_prompt_text, step_prompts_dir=step_prompts_dir)
    
    mode = "openai"
    if args.mock_llm:
        mode = "mock"
    elif args.model and (Path(args.model).exists() or "/" in args.model):
        mode = "local"

    llm = build_llm_client(mode, model=args.model)
    persistence = PersistenceManager(Path(args.out))
    
    orchestrator = AgentOrchestrator(
        llm_client=llm,
        prompt_stack=prompt_stack,
        persistence=persistence,
        repair_limit=args.repair_limit,
        max_turns=args.max_turns,
        run_probes=not args.no_probes,  # NEW: enable/disable probes
    )

    # UPDATED: New state fields (removed iis_reports, step1_tags; renamed steps)
    initial_state = {
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
        "semantic_probe_reports": [],  # NEW: replaces iis_reports
        "solve_reports": [],
        "repair_briefs": [],
        "turn_index": 0,
    }
    
    AgentStateModel.model_validate(initial_state)
    final_state = orchestrator.run(initial_state)
    artifact_dir = Path(args.out) / initial_state["run_id"]

    pre_codegen_path = artifact_dir / "sft_pre_codegen.jsonl"
    codegen_path = artifact_dir / "sft_codegen.jsonl"
    export_pre_codegen_jsonl(
        conversation_log=final_state.get("conversation_log", []),
        path=pre_codegen_path,
        scenario_id=scenario_id,
        run_id=initial_state["run_id"],
    )
    export_codegen_jsonl(
        codegen_conversation=final_state.get("codegen_conversation"),
        conversation_log=final_state.get("conversation_log", []),
        path=codegen_path,
        scenario_id=scenario_id,
        run_id=initial_state["run_id"],
    )
    
    # Print summary
    print(f"Run complete. Artifacts at {artifact_dir}")
    
    # Print probe results if available
    if final_state.get("semantic_probe_reports"):
        last_probe = final_state["semantic_probe_reports"][-1]
        print(f"Semantic Probes: {last_probe.passed}/{last_probe.total} passed")
        if last_probe.failed_probes:
            print(f"Failed probes: {last_probe.failed_probes}")
    
    # Print solve results if available
    if final_state.get("solve_reports"):
        last_solve = final_state["solve_reports"][-1]
        print(f"Solver status: {last_solve.status}")
        if last_solve.obj_val is not None:
            print(f"Objective: {last_solve.obj_val}")
    
    return final_state


if __name__ == "__main__":
    main()
