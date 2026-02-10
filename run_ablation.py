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

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
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
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
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
TOL_STRICT = 1e-4    # ε=10⁻⁴ (0.01% relative error)
TOL_MEDIUM = 1e-2    # ε=10⁻² (1% relative error)
TOL_LOOSE  = 0.05    # ε=5%   (5% relative error)


@dataclass
class StageResult:
    objective: Optional[float]
    status: str
    passed_strict: bool   # ε=10⁻⁴
    passed_medium: bool   # ε=10⁻²
    passed_loose: bool    # ε=5%


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


def relative_gap(pred: Optional[float], truth: Optional[float]) -> Optional[float]:
    """Compute relative gap: |pred - truth| / |truth|. Returns None if either is None."""
    if pred is None or truth is None:
        return None
    if truth == 0:
        return abs(pred) if pred != 0 else 0.0
    return abs(pred - truth) / max(abs(truth), 1e-12)


def make_stage(obj: Optional[float], status: str, gt: Optional[float]) -> StageResult:
    """Create StageResult with pass checks at all tolerance levels."""
    return StageResult(
        objective=obj,
        status=status,
        passed_strict=pass_check(obj, gt, TOL_STRICT),
        passed_medium=pass_check(obj, gt, TOL_MEDIUM),
        passed_loose=pass_check(obj, gt, TOL_LOOSE),
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
    generator = CodeGenerator(llm_client=llm, use_structured_generation=True)
    extractor = DataExtractor(llm_client=llm)
    verifier = ReLoopVerifier(llm_client=llm)
    executor = CodeExecutor()
    pipeline = ReLoopPipeline(
        llm_client=llm,
        enable_cpt=enable_cpt,
        enable_l4_adversarial=True,
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

        # Stage 1: L1 execution verification (+ duality check)
        l1_results, l1_base = verifier._layer1(code, data, verbose=False)
        l1_obj = l1_base.get("objective") if isinstance(l1_base, dict) else None
        cot_obj = l1_obj
        cot_status = l1_base.get("status") or "UNKNOWN"
        l1_status = "FAILED" if any(r.severity == Severity.FATAL for r in l1_results) else "PASS"

        # Stage 2: L1+L2 direction analysis (adversarial loop, no CPT)
        l2_obj = None
        l2_status = "FAILED"
        if l1_obj is not None:
            layer_results = list(l1_results)
            report_l1 = verifier._aggregate(
                layer_results, l1_obj, l1_base.get("solution"), verifier._estimate_complexity(code, data), 0, False
            )
            l2_results, l2_exit, code_after_l2 = pipeline._run_l2_adversarial_loop(
                code=code,
                data=data,
                baseline_obj=l1_obj,
                problem_description=problem,
                report=report_l1,
                max_l2_iterations=3,
            )
            # Re-verify (without CPT) to get objective
            report_after_l2 = verifier.verify(
                code_after_l2, data, problem_description=problem, enable_cpt=False, verbose=False
            )
            l2_obj = report_after_l2.objective
            l2_status = report_after_l2.status
            code = code_after_l2  # use for final stage
        if verbose:
            print(f"[ablation] idx={idx} after L2 status={l2_status}")

        # Stage 3: Final (L1+L2+L3 full pipeline with CPT)
        final_result = pipeline.run(
            problem_description=problem,
            data=data,
            initial_code=code,
        )
        final_obj = final_result.final_report.objective
        final_status = final_result.final_report.status
        if verbose:
            print(f"[ablation] idx={idx} final status={final_status}, obj={final_obj}")

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
            cot=StageResult(None, "FAILED", False, False, False),
            l1=StageResult(None, "FAILED", False, False, False),
            l2=StageResult(None, "FAILED", False, False, False),
            final=StageResult(None, "FAILED", False, False, False),
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging to stdout.")
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    if not records:
        print(f"[ablation] No records loaded from {args.dataset}")
        sys.exit(1)

    dataset_stem = Path(args.dataset).stem
    base_dir = Path(args.output_dir) if args.output_dir else Path("experiment_results") / dataset_stem / args.model
    base_dir.mkdir(parents=True, exist_ok=True)

    results: List[RecordResult] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(run_item, idx, item, args.model, args.base_url, args.enable_cpt, args.verbose): idx
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
        writer.writerow([
            "index",
            "problem_id",
            "difficulty",
            "ground_truth",
            "cot_obj",
            "cot_pass_1e4",
            "cot_pass_1e2",
            "cot_pass_5pct",
            "cot_gap",
            "l1_obj",
            "l1_pass_1e4",
            "l1_pass_1e2",
            "l1_pass_5pct",
            "l1_gap",
            "l2_obj",
            "l2_pass_1e4",
            "l2_pass_1e2",
            "l2_pass_5pct",
            "l2_gap",
            "final_obj",
            "final_pass_1e4",
            "final_pass_1e2",
            "final_pass_5pct",
            "final_gap",
            "runtime_s",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ])
        for r in sorted(results, key=lambda x: x.index):
            writer.writerow([
                r.index,
                r.problem_id,
                r.difficulty,
                r.ground_truth,
                r.cot.objective,
                r.cot.passed_strict,
                r.cot.passed_medium,
                r.cot.passed_loose,
                relative_gap(r.cot.objective, r.ground_truth),
                r.l1.objective,
                r.l1.passed_strict,
                r.l1.passed_medium,
                r.l1.passed_loose,
                relative_gap(r.l1.objective, r.ground_truth),
                r.l2.objective,
                r.l2.passed_strict,
                r.l2.passed_medium,
                r.l2.passed_loose,
                relative_gap(r.l2.objective, r.ground_truth),
                r.final.objective,
                r.final.passed_strict,
                r.final.passed_medium,
                r.final.passed_loose,
                relative_gap(r.final.objective, r.ground_truth),
                f"{r.runtime:.3f}",
                r.prompt_tokens,
                r.completion_tokens,
                r.total_tokens,
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

    for stage_name in ["cot", "final"]:
        stages = [getattr(r, stage_name) for r in results]
        gts = [r.ground_truth for r in results]

        exec_count = sum(1 for s in stages if s.objective is not None)
        strict_count = sum(1 for s in stages if s.passed_strict)
        medium_count = sum(1 for s in stages if s.passed_medium)
        loose_count = sum(1 for s in stages if s.passed_loose)

        gaps = [relative_gap(s.objective, gt) for s, gt in zip(stages, gts)
                if s.objective is not None and gt is not None]
        avg_gap = sum(gaps) / len(gaps) if gaps else float("nan")

        label = "CoT (no verify)" if stage_name == "cot" else "CoT + ReLoop"
        print(f"\n  {label}:")
        print(f"    Exec%          = {exec_count}/{n} ({100*exec_count/n:.1f}%)")
        print(f"    Acc%(ε=10⁻⁴)   = {strict_count}/{n} ({100*strict_count/n:.1f}%)")
        print(f"    Acc%(ε=10⁻²)   = {medium_count}/{n} ({100*medium_count/n:.1f}%)")
        print(f"    Acc%(ε=5%)     = {loose_count}/{n} ({100*loose_count/n:.1f}%)")
        print(f"    Avg Gap        = {avg_gap:.4f} ({100*avg_gap:.2f}%)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
