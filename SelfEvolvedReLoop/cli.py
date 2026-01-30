from __future__ import annotations

import argparse
import os
import json
import sys
from typing import Optional

from .controller import Controller
from .state import ExtractionConfig
from .skills.fixes import promote_candidate_to_trusted
from .skills.replay import ReplayValidationSkill
from .llm import _env_api_key
from .eval_min.runner import run_nl4opt_eval


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def cmd_solve(args) -> None:
    use_llm = bool(os.environ.get("OR_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    config = ExtractionConfig(
        use_llm=use_llm,
        prompt_version="v1",
        strict_mode=False,
        auto_apply_trusted=args.auto_apply_trusted,
        suggest_candidate=args.suggest_candidate,
        allow_candidate_auto_apply=args.allow_candidate_auto_apply,
    )
    data_dict = json.loads(args.data_json) if args.data_json else None
    runtime_kwargs = {"budget_override": args.disclosure_budget_tokens, "llm_model": args.llm_model}
    controller = Controller(runtime_kwargs=runtime_kwargs)
    record = controller.run(
        args.text,
        extraction_config=config,
        input_data=data_dict,
        expected_objective=args.expected_obj,
        tolerance=args.tolerance,
    )

    audit_pass = record.audit_result.passed if record.audit_result else False
    objective = record.solve_result.objective if record.solve_result else None
    print(f"run_id: {record.run_id}")
    print(f"archetype: {record.archetype_id}")
    print(f"status: {record.final_status}")
    if record.not_implemented:
        print("not_implemented:", record.not_implemented.model_dump())
    print(f"audit_pass: {audit_pass}")
    print(f"objective: {objective}")
    if record.expected_objective is not None:
        print(f"expected_objective: {record.expected_objective} tolerance: {record.expected_tolerance} is_correct: {record.is_correct}")
    if record.candidate_fix:
        print("candidate_fix:", record.candidate_fix.patch_type, record.candidate_fix.patch_payload)
        if record.sandbox_report:
            print("sandbox_recommendation:", record.sandbox_report.recommendation)
    if record.prompt_injections:
        prefixes = {k: (v.get("sha256", "")[:8] if v else "") for k, v in record.prompt_injections.items()}
        print("Prompt injections:", prefixes)


def cmd_feedback(args) -> None:
    label = args.label.lower()
    if label not in {"correct", "wrong", "skip"}:
        raise SystemExit("label must be correct|wrong|skip")
    fix = promote_candidate_to_trusted(run_id=args.run, label=label, notes=args.notes)
    if fix:
        print(f"Recorded feedback for fix {fix.fix_id} as {label}.")
    else:
        print("No candidate fix found for run; feedback stored as run-level only.")


def cmd_replay(args) -> None:
    skill = ReplayValidationSkill()
    report = skill.run(archetype_id=args.archetype, max_runs=args.max_runs)
    print("archetype:", report.archetype_id)
    print("total_runs:", report.total_runs)
    print("pass_rate:", report.pass_rate)
    print("avg_objective:", report.avg_objective)
    print("avg_runtime:", report.avg_runtime)
    for row in report.details:
        print(row)

def cmd_evaluate(args) -> None:
    if args.llm and not _env_api_key():
        print("OR_LLM_API_KEY (or OPENAI_API_KEY) is required for LLM evaluation")
        sys.exit(2)
    if args.format != "nl4opt":
        raise SystemExit("Unsupported format")
    result = run_nl4opt_eval(
        path=args.dataset,
        llm=bool(args.llm),
        tolerance=args.tolerance,
        max_examples=args.max_examples,
        out_dir=args.out_dir,
    )
    summary = result["summary"]
    out_dir = result["out_dir"]
    print(
        f"n={summary.get('n')} feas={summary.get('feasibility_rate')} opt={summary.get('optimal_rate')} "
        f"correct={summary.get('correctness_rate')} mae={summary.get('mae')} avg_runtime_ms={summary.get('avg_runtime_ms')}"
    )
    print(f"report_dir: {out_dir}")


def cmd_smoke_llm(_args) -> None:
    if not _env_api_key():
        print("OR_LLM_API_KEY (or OPENAI_API_KEY) is required for smoke-llm")
        sys.exit(2)
    scenarios = [
        ("allocation_lp", "allocation problem: minimize cost while meeting calories>=10\n{\"items\":[{\"name\":\"a\",\"cost\":1,\"features\":{\"cal\":5}},{\"name\":\"b\",\"cost\":2,\"features\":{\"cal\":10}}],\"requirements\":{\"cal\":10}}", None),
        ("assignment", "assignment problem\n{\"workers\":[\"w1\",\"w2\"],\"tasks\":[\"t1\",\"t2\"],\"costs\":{\"w1\":{\"t1\":1,\"t2\":5},\"w2\":{\"t1\":4,\"t2\":1}}}", None),
        ("retail_inventory", "retail inventory test", json.dumps({"products": ["p1"], "locations": ["L1"], "periods": ["t1"], "demand": {"p1": [2]}, "production_cap": {"p1": [5]}, "purchase_cost": {"p1": 1}, "hold_cost": {"p1": 0.1}, "lost_penalty": {"p1": 10}, "cold_usage": {"p1": 0}, "cold_capacity": {"L1": 100}})),
    ]
    rows = []
    for arch, text, data_json in scenarios:
        data = json.loads(data_json) if data_json else None
        config = ExtractionConfig(use_llm=True)
        controller = Controller(runtime_kwargs={"budget_override": None, "llm_model": None})
        record = controller.run(text, extraction_config=config, input_data=data)
        rows.append((arch, record.final_status, record.solve_result.objective if record.solve_result else None))
    print("arch | status | objective")
    for r in rows:
        print(f"{r[0]} | {r[1]} | {r[2]}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="SelfEvolvedReLoop", description="OR agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="Solve a scenario text")
    p_solve.add_argument("--text", required=True, help="Scenario description")
    p_solve.add_argument("--auto-apply-trusted", type=parse_bool, default=True)
    p_solve.add_argument("--suggest-candidate", type=parse_bool, default=True)
    p_solve.add_argument("--allow-candidate-auto-apply", type=parse_bool, default=False)
    p_solve.add_argument("--data-json", default=None, help="Optional data dict for retail_inventory archetype")
    p_solve.add_argument("--expected-obj", dest="expected_obj", type=float, default=None)
    p_solve.add_argument("--tolerance", type=float, default=1e-6)
    p_solve.add_argument("--disclosure-budget-tokens", type=int, default=None)
    p_solve.add_argument("--llm-model", default=None)
    p_solve.set_defaults(func=cmd_solve)

    p_fb = sub.add_parser("feedback", help="Record human feedback for a run")
    p_fb.add_argument("--run", required=True, help="run_id")
    p_fb.add_argument("--label", required=True, choices=["correct", "wrong", "skip"])
    p_fb.add_argument("--notes", default=None)
    p_fb.set_defaults(func=cmd_feedback)

    p_rep = sub.add_parser("replay", help="Replay stored runs for regression validation")
    p_rep.add_argument("--archetype", required=True, help="archetype id")
    p_rep.add_argument("--max-runs", type=int, default=20)
    p_rep.set_defaults(func=cmd_replay)

    p_smoke = sub.add_parser("smoke-llm", help="Run small end-to-end LLM smoke tests")
    p_smoke.set_defaults(func=cmd_smoke_llm)

    p_eval = sub.add_parser("evaluate", help="Run offline/online evaluation")
    p_eval.add_argument("--dataset", required=True, help="Path to dataset file")
    p_eval.add_argument("--format", required=True, choices=["nl4opt"])
    p_eval.add_argument("--llm", type=parse_bool, default=True)
    p_eval.add_argument("--tolerance", type=float, default=1e-6)
    p_eval.add_argument("--max-examples", type=int, default=None)
    p_eval.add_argument("--out-dir", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
