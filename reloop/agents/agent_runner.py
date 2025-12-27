import json
import os
import time
from dataclasses import asdict
from typing import Dict, Any, List, Optional

from .llm_client import build_llm_client
from .prompt_loader import load_prompt_for_scenario
from .contract_checker import check_contract
from .verifier import run_and_verify
from .semantic_check import semantic_check
from .iis_analyzer import cluster_iis
from .feedback_builder import build_repair_brief
from .logging_artifacts import (
    append_jsonl,
    ensure_dir,
    save_text,
)
from .agent_types import (
    LLMMessage,
    LLMResponse,
    AgentSummary,
    ExecutionResult,
    SemanticReport,
    ContractReport,
)


def _load_data(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compress_messages(messages: List[LLMMessage], keep_last: int = 3) -> List[LLMMessage]:
    base = []
    if messages:
        base.append(messages[0])  # system
    if len(messages) > 1:
        base.append(messages[1])  # original user
    # keep last few assistant/user exchanges
    tail = messages[2:]
    if keep_last > 0 and tail:
        tail = tail[-keep_last:]
    return base + tail


def run_agent(
    scenario_id: str,
    data_path: str,
    prompts_dir: str,
    max_iters: int = 12,
    strategy: str = "patch",
    workdir: str = "runs",
    semantic_gate: bool = True,
    llm_mode: str = "mock",
    request_iis: bool = True,
    timeout_s: float = 60.0,
    patience: int = 3,
) -> AgentSummary:
    start_time = time.time()
    system_prompt, user_prompt = load_prompt_for_scenario(scenario_id, prompts_dir)
    data = _load_data(data_path)
    cost_per_1k = float(os.environ.get("RELOOP_COST_PER_1K", "0"))

    run_root = os.path.join(workdir, scenario_id, time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_root)
    messages_log = os.path.join(run_root, "messages.jsonl")
    training_trace_path = os.path.join(run_root, "training_trace.jsonl")

    client = build_llm_client(llm_mode)

    messages: List[LLMMessage] = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]

    total_tokens_in = 0
    total_tokens_out = 0
    best_obj = None
    failure_reason: Optional[str] = None
    success = False

    no_progress_streak = 0
    last_missing = None

    for round_id in range(1, max_iters + 1):
        messages = _compress_messages(messages)

        llm_response: LLMResponse = client.complete(messages)
        script_text = llm_response.content
        total_tokens_in += llm_response.tokens_in
        total_tokens_out += llm_response.tokens_out

        append_jsonl(
            messages_log,
            [
                {
                    "round": round_id,
                    "messages": [m.__dict__ for m in messages],
                    "timestamp": time.time(),
                    "tokens_in": llm_response.tokens_in,
                    "tokens_out": llm_response.tokens_out,
                }
            ],
        )

        # Save script
        save_text(os.path.join(run_root, "scripts", f"round_{round_id}.py"), script_text)
        save_text(os.path.join(run_root, "llm_generated.py"), script_text)

        contract_report: ContractReport = check_contract(script_text)
        if not contract_report.ok:
            exec_result = ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                traceback="; ".join(contract_report.reasons),
                duration=0.0,
                license_limited=False,
            )
        else:
            exec_result = run_and_verify(
                script_text,
                data,
                timeout_s=timeout_s,
                request_iis_on_infeasible=request_iis,
            )

        save_text(
            os.path.join(run_root, "exec", f"round_{round_id}_stdout.txt"),
            exec_result.stdout,
        )
        save_text(
            os.path.join(run_root, "exec", f"round_{round_id}_stderr.txt"),
            exec_result.stderr,
        )
        save_text(
            os.path.join(run_root, "exec", f"round_{round_id}_traceback.txt"),
            exec_result.traceback,
        )

        iis_summary = None
        if exec_result.license_limited:
            semantic_report = SemanticReport(
                valid=False, missing_prefixes=[], unexpected_modules=[], suspicious=[]
            )
        else:
            semantic_report = semantic_check(data, exec_result)
            if exec_result.iis_constraints:
                iis_summary = cluster_iis(exec_result.iis_constraints)
                save_text(
                    os.path.join(run_root, "iis", f"round_{round_id}_iis.json"),
                    json.dumps(iis_summary),
                )

        save_text(
            os.path.join(run_root, "semantic", f"round_{round_id}_report.json"),
            json.dumps(asdict(semantic_report)),
        )

        save_text(
            os.path.join(run_root, "exec", f"round_{round_id}_meta.json"),
            json.dumps(
                {
                    "status": exec_result.status_str,
                    "status_code": exec_result.status_code,
                    "objective": exec_result.objective,
                    "feasible": exec_result.feasible,
                    "duration": exec_result.duration,
                    "license_limited": exec_result.license_limited,
                }
            ),
        )

        training_rec = {
            "scenario_id": scenario_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "round": round_id,
            "round_k_input_messages": [m.__dict__ for m in messages],
            "round_k_output_script": script_text,
            "verifier_result": asdict(exec_result),
            "semantic_report": asdict(semantic_report),
            "iis_clusters": iis_summary,
        }
        append_jsonl(training_trace_path, [training_rec])

        best_obj = exec_result.objective if exec_result.objective is not None else best_obj

        semantic_valid = semantic_report.valid
        compilation_ok = exec_result.success and contract_report.ok

        if exec_result.license_limited:
            failure_reason = "license_limited"
            success = False
            break

        if compilation_ok and exec_result.feasible and (semantic_valid or not semantic_gate):
            success = True
            failure_reason = None
            break

        # stopping: check progress
        if semantic_report.missing_prefixes == last_missing:
            no_progress_streak += 1
        else:
            no_progress_streak = 0
        last_missing = list(semantic_report.missing_prefixes)
        if no_progress_streak >= patience:
            failure_reason = "no_improvement"
            break

        repair_brief = build_repair_brief(
            contract_report, exec_result, semantic_report, iis_summary
        )

        # Build next prompt
        if strategy == "patch":
            prompt = (
                "Here is your previous script. Apply the Repair Brief and make minimal changes.\n"
                "Previous script:\n"
                f"{script_text}\n\n{repair_brief}"
            )
        else:
            prompt = (
                "Rewrite the full script from scratch following the original instructions.\n"
                f"{repair_brief}"
            )
        messages.append(LLMMessage(role="assistant", content=script_text))
        messages.append(LLMMessage(role="user", content=prompt))

    end_time = time.time()
    if not success and not failure_reason:
        failure_reason = "max_iters"

    summary = AgentSummary(
        scenario_id=scenario_id,
        mode="agent",
        compilation_ok=compilation_ok if "compilation_ok" in locals() else False,
        feasible=exec_result.feasible if exec_result.feasible is not None else False,
        semantic_valid=semantic_valid if "semantic_valid" in locals() else False,
        status=exec_result.status_str if "exec_result" in locals() else "N/A",
        objective=exec_result.objective if "exec_result" in locals() else None,
        iters=round_id,
        runtime_s=end_time - start_time,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        total_cost_est=cost_per_1k * (total_tokens_in + total_tokens_out) / 1000.0,
        config={
            "strategy": strategy,
            "semantic_gate": semantic_gate,
            "llm_mode": llm_mode,
            "request_iis": request_iis,
            "patience": patience,
        },
        failure_reason=failure_reason,
    )

    save_text(os.path.join(run_root, "summary.json"), json.dumps(asdict(summary)))
    return summary
