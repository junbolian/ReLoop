#!/usr/bin/env python
"""
Dataset ablation runner.

For each record:
1) Generate CoT code.
2) Evaluate baseline execution (CoT only).
3) Run L1, L1+L2 (direction analysis), final L1+L2+L3 (with CPT).
4) Record objective per stage and pass/fail vs ground truth with tolerance.
5) Accumulate token usage and runtime. Concurrency = 20.
6) Save summary CSV and chat logs JSONL.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reloop import (
    CodeGenerator,
    ReLoopVerifier,
    ReLoopPipeline,
    DataExtractor,
    CodeExecutor,
    Severity,
)
# param_utils no longer needed for ablation stages

# Optional: OpenAI client for token accounting
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# --------------------------------------------------------------------------- #
# LLM wrapper with usage tracking
# --------------------------------------------------------------------------- #

class UsageLLM:
    """OpenAI-compatible LLM client with temperature=0, usage & log tracking."""

    # Reasoning models that need max_completion_tokens instead of max_tokens
    REASONING_MODELS = {"deepseek-r1", "deepseek-reasoner", "o1", "o1-preview", "o1-mini", "o3", "o3-mini"}

    def __init__(self, model: str, base_url: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("openai package not available; pip install openai")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://yinli.one/v1")
        kwargs = {"base_url": base_url} if base_url else {}
        self.client = OpenAI(**kwargs)
        self.model = model
        self.is_reasoning = any(model.startswith(r) for r in self.REASONING_MODELS)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.logs: List[Dict[str, Any]] = []

    def generate(self, prompt: str, system: Optional[str] = None, temperature: Optional[float] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.is_reasoning:
            # Reasoning models: use max_completion_tokens (covers thinking + output)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=16384,
            )
        else:
            t = temperature if temperature is not None else 0
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=t,
                seed=0,
                max_tokens=8192,
            )
        usage = resp.usage
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens or 0
            self.total_completion_tokens += usage.completion_tokens or 0
        content = resp.choices[0].message.content
        self.logs.append(
            {
                "messages": messages,
                "response": content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "total_tokens": usage.total_tokens if usage else None,
                },
            }
        )
        return content


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

# Tolerance levels for accuracy evaluation
# Tolerance levels — selected per dataset in main()
TOL_STRICT = 1e-4    # ε=10⁻⁴  (RetailOpt default)
TOL_MEDIUM = 1e-2    # ε=10⁻²  (RetailOpt default)

# Cross-benchmark datasets use stricter tolerance (ε=10⁻⁶)
CROSS_BENCHMARK_DATASETS = {"IndustryOR", "MAMO_ComplexLP", "MAMO_EasyLP"}


@dataclass
class StageResult:
    objective: Optional[float]
    status: str
    passed_strict: bool   # ε=10⁻⁴
    passed_medium: bool   # ε=10⁻²


@dataclass
class RecordResult:
    index: int
    problem_id: Any
    difficulty: str
    ground_truth: Optional[float]
    cot: StageResult
    l1: StageResult
    l2: StageResult
    final: StageResult
    runtime: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    chat_logs: List[Dict[str, Any]]


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def load_dataset(path: str) -> List[Dict[str, Any]]:
    content = Path(path).read_text(encoding="utf-8").strip()
    records: List[Dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line in {"[", "]"}:
            continue
        if line.endswith(","):
            line = line[:-1]
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError:
            continue
    if records:
        return records
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("problems", [])
    except json.JSONDecodeError:
        return []
    return []


def pass_check(pred: Optional[float], truth: Optional[float], tol: float) -> bool:
    if pred is None or truth is None:
        return False
    if truth == 0:
        return abs(pred) < tol
    return abs(pred - truth) / max(abs(truth), 1e-12) < tol


def make_stage(obj: Optional[float], status: str, gt: Optional[float]) -> StageResult:
    """Create StageResult with pass checks at all tolerance levels."""
    return StageResult(
        objective=obj,
        status=status,
        passed_strict=pass_check(obj, gt, TOL_STRICT),
        passed_medium=pass_check(obj, gt, TOL_MEDIUM),
    )


def stage_report_from_layers(verifier: ReLoopVerifier, layers: List[Any],
                             objective: float, solution: Optional[Dict[str, float]]) -> Any:
    return verifier._aggregate(layers, objective, solution, verifier._estimate_complexity("", {}), 0, False)


# --------------------------------------------------------------------------- #
# Per-item worker
# --------------------------------------------------------------------------- #

def run_item(
    idx: int,
    item: Dict[str, Any],
    model: str,
    base_url: Optional[str],
    enable_cpt: bool,
    verbose: bool = False,
    use_cot: bool = True,
    run_verify: bool = True,
) -> RecordResult:
    started = time.time()
    difficulty = item.get("difficulty", "")
    problem_id = item.get("id", idx)
    problem = item.get("en_question", "") or item.get("question", "")
    if "en_answer" in item:
        gt_raw = item.get("en_answer")
    elif "answer" in item:
        gt_raw = item.get("answer")
    else:
        gt_raw = None
    try:
        # Preserve explicit zero; only set None if truly missing/invalid
        ground_truth = float(gt_raw)
    except Exception:
        ground_truth = None

    llm = UsageLLM(model=model, base_url=base_url)
    generator = CodeGenerator(llm_client=llm, use_structured_generation=use_cot)
    extractor = DataExtractor(llm_client=llm)
    executor = CodeExecutor()
    if run_verify:
        pipeline = ReLoopPipeline(
            llm_client=llm,
            enable_cpt=enable_cpt,
            enable_l2_adversarial=True,
            max_repair_iterations=3,
            verbose=False,
        )

    try:
        if verbose:
            print(f"[ablation] idx={idx} start, model={model}")
        # Extract data
        data = extractor.extract(problem)
        if verbose:
            print(f"[ablation] idx={idx} extracted {len(data)} params")

        # Generate code
        code = generator.generate(problem, data)
        if verbose:
            print(f"[ablation] idx={idx} code generated (len={len(code)})")

        # Execute code to get baseline objective (used for all modes)
        exec_result = executor.execute(code, data)
        cot_obj = exec_result.get("objective") if isinstance(exec_result, dict) else None
        cot_status = exec_result.get("status") or "UNKNOWN" if isinstance(exec_result, dict) else "FAILED"

        if run_verify:
            # Single pipeline run with intermediate checkpoints.
            # pipeline.run() records l1/l2 checkpoints automatically:
            #   l1_checkpoint: after L1 verify + regeneration (before L2)
            #   l2_checkpoint: after L2 adversarial loop (before repair)
            #   final: after full repair loop with all diagnostics
            result = pipeline.run(
                problem_description=problem,
                data=data,
                initial_code=code,
            )
            l1_obj = result.l1_checkpoint_obj
            l1_status = result.l1_checkpoint_status
            l2_obj = result.l2_checkpoint_obj
            l2_status = result.l2_checkpoint_status
            final_obj = result.final_report.objective
            final_status = result.final_report.status
            if verbose:
                print(f"[ablation] idx={idx} L1={l1_obj} L2={l2_obj} final={final_obj}")
        else:
            # No verification: all stages same as raw execution
            l1_obj = cot_obj
            l1_status = cot_status
            l2_obj = cot_obj
            l2_status = cot_status
            final_obj = cot_obj
            final_status = cot_status

        runtime = time.time() - started
        prompt_tokens = llm.total_prompt_tokens
        completion_tokens = llm.total_completion_tokens

        return RecordResult(
            index=idx,
            problem_id=problem_id,
            difficulty=difficulty,
            ground_truth=ground_truth,
            cot=make_stage(cot_obj, cot_status, ground_truth),
            l1=make_stage(l1_obj, l1_status, ground_truth),
            l2=make_stage(l2_obj, l2_status, ground_truth),
            final=make_stage(final_obj, final_status, ground_truth),
            runtime=runtime,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            chat_logs=llm.logs,
        )
    except Exception as e:
        if verbose:
            print(f"[ablation] idx={idx} failed: {e}")
        runtime = time.time() - started
        prompt_tokens = llm.total_prompt_tokens
        completion_tokens = llm.total_completion_tokens
        return RecordResult(
            index=idx,
            problem_id=problem_id,
            difficulty=difficulty,
            ground_truth=ground_truth,
            cot=StageResult(None, "FAILED", False, False),
            l1=StageResult(None, "FAILED", False, False),
            l2=StageResult(None, "FAILED", False, False),
            final=StageResult(None, "FAILED", False, False),
            runtime=runtime,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            chat_logs=llm.logs + [{"error": str(e)}],
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Ablation runner for ReLoop.")
    parser.add_argument("-d", "--dataset", required=True, help="Path to dataset json/jsonl.")
    parser.add_argument("-o", "--output-dir", default=None, help="Output base directory (default: experiment_results/<dataset-stem>/<model>).")
    parser.add_argument("-m", "--model", default="gpt-4.1", help="OpenAI model name.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--enable-cpt", action="store_true", help="Enable L3 CPT in final stage.")
    parser.add_argument("--workers", type=int, default=20, help="Concurrency.")
    parser.add_argument("--no-cot", action="store_true", help="Use direct generation (no CoT structured generation).")
    parser.add_argument("--no-verify", action="store_true", help="Skip all verification stages (execute only).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging to stdout.")
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    if not records:
        print(f"[ablation] No records loaded from {args.dataset}")
        sys.exit(1)

    dataset_stem = Path(args.dataset).stem

    # Select tolerance per dataset
    global TOL_STRICT, TOL_MEDIUM
    is_cross = any(cb in dataset_stem for cb in CROSS_BENCHMARK_DATASETS)
    if is_cross:
        TOL_STRICT = 1e-6    # ε=10⁻⁶
        TOL_MEDIUM = 1e-6    # ε=10⁻⁶ (single tier for cross-benchmark)
        print("[ablation] Cross-benchmark dataset detected -> eps=1e-6")
    else:
        print("[ablation] RetailOpt dataset -> eps=1e-4 / eps=1e-2")

    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        mode_suffix = ""
        if args.no_cot:
            mode_suffix += "_direct"
        if args.no_verify:
            mode_suffix += "_noverify"
        base_dir = Path("experiment_results") / dataset_stem / (args.model + mode_suffix)
    base_dir.mkdir(parents=True, exist_ok=True)

    results: List[RecordResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_item, idx, item, args.model, args.base_url, args.enable_cpt, args.verbose,
                      use_cot=not args.no_cot, run_verify=not args.no_verify): idx
            for idx, item in enumerate(records)
        }
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                print(f"[ablation] finished index {res.index}")
            except Exception as e:
                idx = futures[fut]
                print(f"[ablation] failed index {idx}: {e}")

    # Save CSV
    import csv

    csv_path = base_dir / "ablation_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_cross:
            # Cross-benchmark: single tolerance ε=10⁻⁶
            writer.writerow([
                "index", "problem_id", "difficulty", "ground_truth",
                "cot_obj", "cot_pass_1e6",
                "l1_obj", "l1_pass_1e6",
                "l2_obj", "l2_pass_1e6",
                "final_obj", "final_pass_1e6",
                "runtime_s", "prompt_tokens", "completion_tokens", "total_tokens",
            ])
            for r in sorted(results, key=lambda x: x.index):
                writer.writerow([
                    r.index, r.problem_id, r.difficulty, r.ground_truth,
                    r.cot.objective, r.cot.passed_strict,
                    r.l1.objective, r.l1.passed_strict,
                    r.l2.objective, r.l2.passed_strict,
                    r.final.objective, r.final.passed_strict,
                    f"{r.runtime:.3f}", r.prompt_tokens, r.completion_tokens, r.total_tokens,
                ])
        else:
            # RetailOpt: two tiers ε=10⁻⁴ / ε=10⁻²
            writer.writerow([
                "index", "problem_id", "difficulty", "ground_truth",
                "cot_obj", "cot_pass_1e4", "cot_pass_1e2",
                "l1_obj", "l1_pass_1e4", "l1_pass_1e2",
                "l2_obj", "l2_pass_1e4", "l2_pass_1e2",
                "final_obj", "final_pass_1e4", "final_pass_1e2",
                "runtime_s", "prompt_tokens", "completion_tokens", "total_tokens",
            ])
            for r in sorted(results, key=lambda x: x.index):
                writer.writerow([
                    r.index, r.problem_id, r.difficulty, r.ground_truth,
                    r.cot.objective, r.cot.passed_strict, r.cot.passed_medium,
                    r.l1.objective, r.l1.passed_strict, r.l1.passed_medium,
                    r.l2.objective, r.l2.passed_strict, r.l2.passed_medium,
                    r.final.objective, r.final.passed_strict, r.final.passed_medium,
                    f"{r.runtime:.3f}", r.prompt_tokens, r.completion_tokens, r.total_tokens,
                ])

    # Chat logs
    logs_path = base_dir / "chat_logs.jsonl"
    with logs_path.open("w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: x.index):
            f.write(json.dumps({"index": r.index, "problem_id": r.problem_id, "logs": r.chat_logs}) + "\n")

    print(f"[ablation] saved CSV to {csv_path}")
    print(f"[ablation] saved logs to {logs_path}")

    # Summary statistics
    n = len(results)
    if n == 0:
        return
    print(f"\n{'='*70}")
    print(f"  SUMMARY  ({n} problems, model={args.model})")
    print(f"{'='*70}")

    gen_label = "Direct" if args.no_cot else "CoT"
    for stage_name in ["cot", "final"]:
        stages = [getattr(r, stage_name) for r in results]
        gts = [r.ground_truth for r in results]

        exec_count = sum(1 for s in stages if s.objective is not None)
        strict_count = sum(1 for s in stages if s.passed_strict)
        medium_count = sum(1 for s in stages if s.passed_medium)

        label = f"{gen_label} (no verify)" if stage_name == "cot" else f"{gen_label} + ReLoop"
        tol_s = "ε=10⁻⁶" if is_cross else "ε=10⁻⁴"
        tol_m = "ε=10⁻⁶" if is_cross else "ε=10⁻²"
        print(f"\n  {label}:")
        print(f"    Exec%           = {exec_count}/{n} ({100*exec_count/n:.1f}%)")
        print(f"    Acc%({tol_s})    = {strict_count}/{n} ({100*strict_count/n:.1f}%)")
        if not is_cross:
            print(f"    Acc%({tol_m})    = {medium_count}/{n} ({100*medium_count/n:.1f}%)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
