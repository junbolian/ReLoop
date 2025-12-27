import argparse
import os
import json
import sys
from pathlib import Path

# Support running both as `python -m reloop.agents.run_one` and `python reloop/agents/run_one.py`
try:  # pragma: no cover - small convenience shim
    from .env_utils import maybe_reexec_with_gurobi
except ImportError:  # executed when __package__ is None (direct script execution)
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root))
    from reloop.agents.env_utils import maybe_reexec_with_gurobi


def resolve_default_paths(base_dir, data_dir_arg, prompts_dir_arg):
    default_data = os.path.join(base_dir, "scenarios", "retailopt_190", "data")
    default_prompts = os.path.join(base_dir, "scenarios", "retailopt_190", "prompts")
    data_dir = data_dir_arg or default_data
    prompts_dir = prompts_dir_arg or default_prompts
    return data_dir, prompts_dir


def main():
    maybe_reexec_with_gurobi()
    try:
        from .langchain_agent import run_langchain_agent
        from .prompt_loader import load_prompt_for_scenario
    except ImportError:  # allow direct execution as a script
        from reloop.agents.langchain_agent import run_langchain_agent
        from reloop.agents.prompt_loader import load_prompt_for_scenario

    parser = argparse.ArgumentParser(description="Run a single scenario with LangChain agent.")
    parser.add_argument("--scenario", required=True, help="Scenario ID (filename without .json)")
    parser.add_argument("--data_dir", default=None, help="Data directory path")
    parser.add_argument("--prompts_dir", default=None, help="Prompts directory path")
    parser.add_argument("--max_iters", type=int, default=6)
    parser.add_argument("--workdir", default="runs")
    parser.add_argument("--timeout_s", type=float, default=200.0)
    parser.add_argument("--request_iis", action="store_true", default=True)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir, prompts_dir = resolve_default_paths(base_dir, args.data_dir, args.prompts_dir)
    data_path = os.path.join(data_dir, f"{args.scenario}.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
    # quick prompt check
    load_prompt_for_scenario(args.scenario, prompts_dir)

    summary = run_langchain_agent(
        scenario_id=args.scenario,
        data_path=data_path,
        prompts_dir=prompts_dir,
        workdir=args.workdir,
        max_iters=args.max_iters,
        request_iis=args.request_iis,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
