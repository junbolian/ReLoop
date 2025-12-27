import argparse
import glob
import json
import os
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from .env_utils import maybe_reexec_with_gurobi


def resolve_default_paths(base_dir, data_dir_arg, prompts_dir_arg):
    default_data = os.path.join(base_dir, "scenarios", "retailopt_190", "data")
    default_prompts = os.path.join(base_dir, "scenarios", "retailopt_190", "prompts")
    data_dir = data_dir_arg or default_data
    prompts_dir = prompts_dir_arg or default_prompts
    return data_dir, prompts_dir


def main():
    maybe_reexec_with_gurobi()
    from .langchain_agent import run_langchain_agent
    from .prompt_loader import load_prompt_for_scenario

    parser = argparse.ArgumentParser(description="LangChain agent benchmark runner.")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--prompts_dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max_iters", type=int, default=6)
    parser.add_argument("--workdir", default="runs")
    parser.add_argument("--timeout_s", type=float, default=120.0)
    parser.add_argument(
        "--output_csv",
        default=os.path.join("eval", "agent_results.csv"),
        help="Where to write aggregate results.",
    )
    parser.add_argument(
        "--request_iis",
        action="store_true",
        default=True,
        help="Enable IIS-driven feedback.",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir, prompts_dir = resolve_default_paths(base_dir, args.data_dir, args.prompts_dir)

    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if args.limit:
        json_files = json_files[: args.limit]
    scenarios = [os.path.splitext(os.path.basename(f))[0] for f in json_files]

    print(f"Running agent on {len(scenarios)} scenarios.")
    results = []
    start = time.time()

    def _run(scenario):
        data_path = os.path.join(data_dir, f"{scenario}.json")
        # validate prompt early
        load_prompt_for_scenario(scenario, prompts_dir)
        summary = run_langchain_agent(
            scenario_id=scenario,
            data_path=data_path,
            prompts_dir=prompts_dir,
            workdir=args.workdir,
            max_iters=args.max_iters,
            request_iis=args.request_iis,
            timeout_s=args.timeout_s,
        )
        return summary

    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            fut_map = {ex.submit(_run, sc): sc for sc in scenarios}
            for fut in as_completed(fut_map):
                summary = fut.result()
                results.append(summary)
                print(f"[done] {summary.get('scenario_id')} runtime={summary.get('runtime_s'):.2f}s")
    else:
        for sc in scenarios:
            summary = _run(sc)
            results.append(summary)
            print(f"[done] {summary.get('scenario_id')} runtime={summary.get('runtime_s'):.2f}s")

    duration = time.time() - start
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_id",
                "runtime_s",
                "output",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "scenario_id": r.get("scenario_id"),
                    "runtime_s": r.get("runtime_s"),
                    "output": r.get("output"),
                }
            )

    print(f"Finished {len(results)} runs in {duration:.2f}s. Results -> {args.output_csv}")


if __name__ == "__main__":
    main()
