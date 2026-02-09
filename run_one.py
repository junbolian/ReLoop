#!/usr/bin/env python
"""Run ReLoop on the first (or nth) problem of a dataset JSONL/JSON file.

Usage examples:
  python run_one.py                               # default data/RetailOpt-190.jsonl, first problem
  python run_one.py -d data/OptMATH_Bench_166.jsonl -i 2 -m gpt-4.1-mini
  python run_one.py --enable-cpt -v

The script expects OPENAI_API_KEY in the environment. No repository code is modified.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from reloop import run_reloop, DataExtractor, ExperimentRunner


# ---------------------------------------------------------------------------
# LLM adapter expected by ReLoop (generate(prompt, system=None) -> str)
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - import guard
    print("[run_one] Missing dependency 'openai'. Install with 'pip install openai'.")
    print(f"[run_one] Import error: {exc}")
    sys.exit(1)


class OpenAILLM:
    """Minimal OpenAI Chat wrapper matching ReLoop's llm_client interface."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        if not os.environ.get("OPENAI_API_KEY"):
            print("[run_one] Please export OPENAI_API_KEY before running.")
            sys.exit(1)

        # Defaults from env if not provided
        if model is None:
            model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://yinli.one/v1")

        kwargs = {"base_url": base_url} if base_url else {}
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            seed=0,
            max_tokens=8192,
        )
        return resp.choices[0].message.content


class LoggingLLM:
    """
    Wraps another llm_client and records every prompt/response pair.
    """

    def __init__(self, base_llm):
        self.base = base_llm
        self.logs: List[Dict[str, Any]] = []

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        started = datetime.utcnow().isoformat() + "Z"
        response = self.base.generate(prompt, system=system)
        self.logs.append(
            {
                "timestamp": started,
                "system": system,
                "prompt": prompt,
                "response": response,
            }
        )
        return response


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load JSONL or JSON array of problems."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    records: List[Dict[str, Any]] = []

    # Try JSONL first
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

    # Fallback: JSON array or object with "problems" key
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            probs = data.get("problems")
            if isinstance(probs, list):
                return [d for d in probs if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Unrecognized dataset format: {path}")


def split_problem_and_data(problem_text: str) -> (str, Optional[Dict[str, Any]]):
    """Extract structured JSON after a [DATA] marker if present."""
    desc = problem_text.strip()
    data_block = None

    match = re.search(r"\[DATA\]\s*(\{[\s\S]*\})", problem_text)
    if match:
        desc = problem_text[: match.start()].strip()
        try:
            data_block = json.loads(match.group(1))
        except json.JSONDecodeError:
            data_block = None

    return desc, data_block


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_report(report) -> Dict[str, Any]:
    """Convert VerificationReport to a JSON-friendly dict."""
    return {
        "status": report.status,
        "has_solution": report.has_solution,
        "objective": report.objective,
        "solution": report.solution,
        "confidence": report.confidence,
        "complexity": report.complexity.name if report.complexity else None,
        "layer_results": [
            {
                "layer": r.layer,
                "check": r.check,
                "severity": r.severity.value,
                "message": r.message,
                "confidence": r.confidence,
                "details": r.details,
            }
            for r in report.layer_results
        ],
        "recommendations": report.recommendations,
        "execution_time": report.execution_time,
    }


def _save_chat_logs(logs: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "chat_logs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def _materialize_direct_outputs(result, logs: List[Dict[str, Any]], out_dir: str, scenario_id: str):
    """Save report, final code, and chat logs for direct (non-runner) mode."""
    os.makedirs(out_dir, exist_ok=True)

    # Report
    with open(os.path.join(out_dir, f"{scenario_id}_report.json"), "w", encoding="utf-8") as f:
        json.dump(_serialize_report(result.final_report), f, indent=2, ensure_ascii=False)

    # Final code
    with open(os.path.join(out_dir, f"{scenario_id}_final_code.py"), "w", encoding="utf-8") as f:
        f.write(result.final_code)

    # Chat logs
    _save_chat_logs(logs, out_dir)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReLoop on a single dataset entry.")
    parser.add_argument(
        "-d",
        "--dataset",
        default="data/RetailOpt-190.jsonl",
        help="Path to dataset JSONL/JSON file.",
    )
    parser.add_argument(
        "-i", "--index", type=int, default=0, help="0-based index of the problem to run."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4.1",
        help="OpenAI chat model name (e.g., gpt-4.1, gpt-4.1-mini).",
    )
    parser.add_argument(
        "--base-url", default=None, help="Optional OpenAI-compatible base URL (for proxies)."
    )
    parser.add_argument(
        "--enable-cpt",
        action="store_true",
        help="Enable CPT (L5) checks; consumes extra LLM calls.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum repair iterations in the pipeline.",
    )
    parser.add_argument(
        "--use-runner",
        action="store_true",
        help="Use ExperimentRunner (saves record/summary) even though only one item is run.",
    )
    parser.add_argument(
        "--output-dir",
        default="results_single",
        help="Output directory. Runner mode: record_*.json/summary.json. Direct mode: report.json/chat_logs.json/final_code.py.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output from the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset(args.dataset)
    if not dataset:
        print(f"[run_one] No problems found in {args.dataset}")
        sys.exit(1)

    if args.index < 0 or args.index >= len(dataset):
        print(f"[run_one] Index {args.index} out of range (0..{len(dataset)-1}).")
        sys.exit(1)

    item = dataset[args.index]
    raw_problem = item.get("en_question") or item.get("question") or item.get("prompt")
    if not raw_problem:
        print("[run_one] Problem text not found in dataset entry.")
        sys.exit(1)

    problem_description, data = split_problem_and_data(raw_problem)

    # LLM with logging
    llm = LoggingLLM(OpenAILLM(model=args.model, base_url=args.base_url))

    if args.use_runner:
        # Let ExperimentRunner handle data extraction to avoid duplicate LLM calls.
        scenario_id = item.get("scenario_id") or item.get("id") or f"idx_{args.index}"
        print(f"[run_one] Running scenario (runner mode): {scenario_id}")

        # Write single item to a temp JSONL and delegate to ExperimentRunner for full logging.
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as tf:
            json.dump(item, tf)
            tf.write("\n")
            temp_path = tf.name

        runner = ExperimentRunner(
            llm_client=llm,
            output_dir=args.output_dir,
            enable_cpt=args.enable_cpt,
            max_repair_iterations=args.max_iterations,
            verbose=args.verbose,
        )
        summary = runner.run_dataset(temp_path)
        _save_chat_logs(llm.logs, args.output_dir)
        print(f"[run_one] Runner mode complete. Summary: {args.output_dir}/summary.json")
        print(
            f"[run_one] Status counts - VERIFIED: {summary.verified_count}, "
            f"WARNINGS: {summary.warnings_count}, ERRORS: {summary.errors_count}, FAILED: {summary.failed_count}"
        )
        try:
            os.remove(temp_path)
        except OSError:
            pass
    else:
        if data is None:
            extractor = DataExtractor(llm)
            data = extractor.extract(problem_description)
            print(f"[run_one] Extracted {len(data)} parameters via LLM extractor.")
        else:
            print("[run_one] Parsed structured data from [DATA] block.")

        scenario_id = item.get("scenario_id") or item.get("id") or f"idx_{args.index}"
        print(f"[run_one] Running scenario: {scenario_id}")

        result = run_reloop(
            problem_description=problem_description,
            data=data,
            llm_client=llm,
            max_iterations=args.max_iterations,
            enable_cpt=args.enable_cpt,
            verbose=args.verbose,
        )

        print("[run_one] Final status:", result.final_report.status)
        print("[run_one] Objective:", result.final_report.objective)
        print("[run_one] Iterations:", result.iterations)
        if result.final_report.recommendations:
            print("[run_one] Recommendations:")
            for rec in result.final_report.recommendations:
                print("  -", rec)
        _materialize_direct_outputs(result, llm.logs, args.output_dir, scenario_id)


if __name__ == "__main__":
    main()
