"""
ReLoop Test with Detailed Conversation Logging

This script runs ReLoop on a scenario and exports detailed conversation logs to JSON.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

# Add reloop to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reloop import ReLoopConfig, BehavioralVerifier, VerificationReport
from reloop.structured_generation import LLMClient, StructuredGenerator, RETAIL_SCHEMA
from reloop.diagnosis_repair import DiagnosisRepairer
from reloop.prompts import PromptGenerator


@dataclass
class ConversationTurn:
    """Single conversation turn (prompt + response)"""
    turn_id: int
    timestamp: str
    role: str  # "generation" | "repair" | "verification"
    step: str  # "step1" | "step2" | "step3" | "repair" | "verify"
    prompt: str
    response: str
    tokens_estimate: int = 0
    duration_ms: int = 0


@dataclass
class ConversationLog:
    """Full conversation log for a ReLoop run"""
    scenario_id: str
    model: str
    start_time: str
    end_time: str = ""
    total_duration_s: float = 0.0
    iterations: int = 0
    final_status: str = ""
    layers_passed: int = 0
    objective_value: Optional[float] = None
    ground_truth: Optional[float] = None
    turns: List[Dict] = field(default_factory=list)
    verification_reports: List[Dict] = field(default_factory=list)
    final_code: str = ""

    def add_turn(self, turn: ConversationTurn):
        self.turns.append(asdict(turn))

    def add_verification(self, iteration: int, report: VerificationReport):
        # Extract objective from L2 result if available
        objective = None
        layer_results = {}
        for r in report.results:
            layer_results[f"L{r.layer}"] = {"passed": r.passed, "diagnosis": r.diagnosis, "details": r.details}
            if r.layer == 2 and r.details:
                objective = r.details.get("objective")

        self.verification_reports.append({
            "iteration": iteration,
            "passed": report.passed,
            "layers_passed": report.count_layers_passed(),
            "failed_layer": report.failed_layer,
            "diagnosis": report.diagnosis,
            "objective": objective,
            "layer_results": layer_results
        })

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class LoggingLLMClient(LLMClient):
    """LLM Client wrapper that logs all conversations"""

    def __init__(self, base_client: LLMClient, log: ConversationLog):
        self.base_client = base_client
        self.log = log
        self.turn_counter = 0
        self.current_role = "generation"
        self.current_step = "unknown"

    def set_context(self, role: str, step: str):
        self.current_role = role
        self.current_step = step

    def generate(self, prompt: str, temperature: float = 0) -> str:
        self.turn_counter += 1
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        response = self.base_client.generate(prompt, temperature)

        duration_ms = int((time.time() - start_time) * 1000)
        tokens_estimate = (len(prompt) + len(response)) // 4  # rough estimate

        turn = ConversationTurn(
            turn_id=self.turn_counter,
            timestamp=timestamp,
            role=self.current_role,
            step=self.current_step,
            prompt=prompt,
            response=response,
            tokens_estimate=tokens_estimate,
            duration_ms=duration_ms
        )
        self.log.add_turn(turn)

        return response


class LoggingStructuredGenerator(StructuredGenerator):
    """StructuredGenerator that logs conversation context"""

    def __init__(self, llm_client: LoggingLLMClient, dataset_type: str = "retail"):
        super().__init__(llm_client, dataset_type)
        self.logging_llm = llm_client

    def generate(self, problem: str, schema: str, history: List[str] = None) -> str:
        self.logging_llm.set_context("generation", "step1")
        understanding = self._step1(problem, schema)

        self.logging_llm.set_context("generation", "step2")
        math_spec = self._step2(understanding, schema)

        self.logging_llm.set_context("generation", "step3")
        code = self._step3(math_spec, schema, history)
        return code

    def generate_baseline(self, problem: str, schema: str) -> str:
        self.logging_llm.set_context("baseline", "single_shot")
        return super().generate_baseline(problem, schema)


class LoggingDiagnosisRepairer(DiagnosisRepairer):
    """DiagnosisRepairer that logs conversation context"""

    def __init__(self, llm_client: LoggingLLMClient):
        super().__init__(llm_client)
        self.logging_llm = llm_client

    def repair(self, code: str, diagnosis: str, failed_layer: int = 1,
               history: List[str] = None) -> str:
        self.logging_llm.set_context("repair", f"layer{failed_layer}_repair")
        return super().repair(code, diagnosis, failed_layer, history)


def run_baseline_with_logging(
    scenario_id: str,
    problem: str,
    data: Dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    ground_truth: Optional[float] = None,
    baseline_prompt: Optional[str] = None,
    verbose: bool = True
) -> ConversationLog:
    """Run baseline (direct generation, no structured steps or repair) with logging

    If baseline_prompt is provided (from .scenario.txt), use it directly.
    Otherwise, generate prompt using the template (legacy behavior).
    """

    # Initialize log
    log = ConversationLog(
        scenario_id=scenario_id,
        model=model,
        start_time=datetime.now().isoformat(),
        ground_truth=ground_truth
    )

    # Create base LLM client
    from reloop.structured_generation import OpenAIClient
    base_client = OpenAIClient(model=model, base_url=base_url, api_key=api_key)

    # Wrap with logging
    logging_client = LoggingLLMClient(base_client, log)

    # Create components
    generator = LoggingStructuredGenerator(logging_client)
    verifier = BehavioralVerifier(delta=0.2, epsilon=1e-4, timeout=60)

    obj_sense = "minimize"

    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print("BASELINE: Direct Generation (single prompt)")
        print('='*60)

    # Generate code directly (single prompt, no 3-step)
    logging_client.set_context("baseline", "single_shot")

    # Use pre-loaded baseline_prompt from .scenario.txt if available
    if baseline_prompt:
        code = generator._extract_code(logging_client.generate(baseline_prompt))
    else:
        # Legacy: generate prompt from template
        code = generator.generate_baseline(problem, RETAIL_SCHEMA)

    if verbose:
        print("  [Generate] Single-shot direct generation...")

    # Verify
    if verbose:
        print("  [Verify] 6-layer behavioral verification...")
    report = verifier.verify(
        code, data, obj_sense,
        enable_layer6=True,
        verbose=verbose
    )

    log.add_verification(1, report)
    layers_passed = report.count_layers_passed()

    if verbose:
        print(f"  [Result] {layers_passed}/6 layers passed")

    # Finalize log
    log.end_time = datetime.now().isoformat()
    log.total_duration_s = time.time() - start_time
    log.iterations = 1
    log.final_status = "VERIFIED" if report.passed else "NOT_VERIFIED"
    log.layers_passed = layers_passed
    log.objective_value = None
    if log.verification_reports:
        for vr in reversed(log.verification_reports):
            if vr.get("objective") is not None:
                log.objective_value = vr["objective"]
                break
    log.final_code = code

    return log


def run_reloop_with_logging(
    scenario_id: str,
    problem: str,
    data: Dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    ground_truth: Optional[float] = None,
    max_iterations: int = 5,
    enable_layer6: bool = True,
    early_stop: bool = True,
    verbose: bool = True
) -> ConversationLog:
    """Run ReLoop with full conversation logging"""

    # Initialize log
    log = ConversationLog(
        scenario_id=scenario_id,
        model=model,
        start_time=datetime.now().isoformat(),
        ground_truth=ground_truth
    )

    # Create base LLM client (all models use OpenAI-compatible API)
    from reloop.structured_generation import OpenAIClient
    base_client = OpenAIClient(model=model, base_url=base_url, api_key=api_key)

    # Wrap with logging
    logging_client = LoggingLLMClient(base_client, log)

    # Create components
    generator = LoggingStructuredGenerator(logging_client)
    verifier = BehavioralVerifier(delta=0.2, epsilon=1e-4, timeout=60)
    repairer = LoggingDiagnosisRepairer(logging_client)

    schema = RETAIL_SCHEMA
    obj_sense = "minimize"

    # Run pipeline
    history = []
    best_code, best_layers, best_report = None, -1, None
    no_progress_count = 0

    start_time = time.time()

    for k in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Iteration {k + 1}/{max_iterations}")
            print('='*60)

        # Generate or Repair
        if k == 0:
            if verbose:
                print("  [Generate] 3-step structured generation...")
            current_code = generator.generate(problem, schema)
        else:
            if verbose:
                print("  [Repair] Diagnosis-guided repair...")
            current_code = repairer.repair(
                code=best_code,
                diagnosis=history[-1] if history else "",
                failed_layer=best_report.failed_layer if best_report else 1,
                history=history[:-1]
            )

        # Verify
        if verbose:
            print("  [Verify] 6-layer behavioral verification...")
        report = verifier.verify(
            current_code, data, obj_sense,
            enable_layer6=enable_layer6,
            verbose=verbose
        )

        log.add_verification(k + 1, report)
        layers_passed = report.count_layers_passed()

        # Track best
        if layers_passed > best_layers:
            best_code, best_layers, best_report = current_code, layers_passed, report
            no_progress_count = 0
            if verbose:
                print(f"  [Best] New best: {best_layers}/6 layers")
        else:
            no_progress_count += 1
            if verbose:
                print(f"  [No Progress] {no_progress_count} iteration(s) without improvement")

        # Success
        if report.passed:
            if verbose:
                print(f"\n  [SUCCESS] All layers passed!")
            break

        # Early stop (can be disabled with --no-early-stop)
        if early_stop and no_progress_count >= 2 and k >= 2:
            if verbose:
                print(f"  [Early Stop] No progress for {no_progress_count} iterations")
            break

        # Record diagnosis
        if report.diagnosis:
            history.append(report.diagnosis)

    # Finalize log
    log.end_time = datetime.now().isoformat()
    log.total_duration_s = time.time() - start_time
    log.iterations = k + 1
    log.final_status = "VERIFIED" if best_report and best_report.passed else "NOT_VERIFIED"
    log.layers_passed = best_layers
    # Extract objective from verification reports
    log.objective_value = None
    if log.verification_reports:
        for vr in reversed(log.verification_reports):
            if vr.get("objective") is not None:
                log.objective_value = vr["objective"]
                break
    log.final_code = best_code or ""

    return log


def load_scenario(scenario_id: str) -> tuple:
    """Load scenario data and prompts

    Returns:
        data: dict - scenario data from JSON
        problem: str - base prompt for ReLoop agent (.base.txt)
        baseline_prompt: str - full prompt for baseline (.scenario.txt)
    """
    data_path = f"scenarios/data/{scenario_id}.json"
    base_prompt_path = f"scenarios/prompts/{scenario_id}.base.txt"
    scenario_prompt_path = f"scenarios/prompts/{scenario_id}.scenario.txt"

    with open(data_path, 'r') as f:
        data = json.load(f)

    with open(base_prompt_path, 'r') as f:
        problem = f.read()

    # Load scenario prompt for baseline (includes schema and guardrails)
    baseline_prompt = None
    if os.path.exists(scenario_prompt_path):
        with open(scenario_prompt_path, 'r') as f:
            baseline_prompt = f.read()

    return data, problem, baseline_prompt


def get_ground_truth(scenario_id: str, data: Dict) -> Optional[float]:
    """Get ground truth using universal solver"""
    try:
        from solvers.universal_retail_solver import solve_retail_milp
        result = solve_retail_milp(data, verbose=False)
        return result.get('objective')
    except Exception as e:
        print(f"Warning: Could not get ground truth: {e}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run ReLoop with conversation logging')
    parser.add_argument('--scenario', type=str, default='retail_f1_base_v4',
                       help='Scenario ID (e.g., retail_f1_base_v4)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (default: from OPENAI_MODEL env)')
    parser.add_argument('--max-iter', type=int, default=5,
                       help='Maximum iterations')
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline (direct generation) instead of ReLoop')
    parser.add_argument('--compare', action='store_true',
                       help='Run both baseline and ReLoop for comparison')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    args = parser.parse_args()

    # Get API config from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL')
    model = args.model or os.environ.get('OPENAI_MODEL', 'gpt-4o')

    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    print(f"Model: {model}")
    print(f"Base URL: {base_url}")
    print(f"Scenario: {args.scenario}")

    # Load scenario
    data, problem, baseline_prompt = load_scenario(args.scenario)
    print(f"Loaded: {data.get('name', args.scenario)}")
    print(f"  - Periods: {data.get('periods')}")
    print(f"  - Products: {len(data.get('products', []))}")
    print(f"  - Locations: {len(data.get('locations', []))}")

    # Get ground truth
    ground_truth = get_ground_truth(args.scenario, data)
    if ground_truth:
        print(f"  - Ground Truth: {ground_truth:,.2f}")

    def print_summary(log: ConversationLog, title: str = "SUMMARY"):
        """Print summary for a log"""
        print("\n" + "="*60)
        print(title)
        print("="*60)
        print(f"Status: {log.final_status}")
        print(f"Layers Passed: {log.layers_passed}/6")
        print(f"Iterations: {log.iterations}")
        print(f"Duration: {log.total_duration_s:.2f}s")
        print(f"LLM Turns: {len(log.turns)}")
        if log.objective_value is not None:
            print(f"Objective: {log.objective_value:,.2f}")
        if log.ground_truth is not None:
            print(f"Ground Truth: {log.ground_truth:,.2f}")
            if log.objective_value is not None:
                gap = abs(log.objective_value - log.ground_truth) / log.ground_truth * 100
                print(f"Gap: {gap:.2f}%")

    def save_log(log: ConversationLog, suffix: str = ""):
        """Save log to file"""
        output_path = args.output or f"logs/{args.scenario}_{model}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "logs", exist_ok=True)
        log.save(output_path)
        print(f"Log saved to: {output_path}")
        return output_path

    # Run based on mode
    if args.compare:
        # Run both baseline and ReLoop for comparison
        print("\n" + "#"*60)
        print("# COMPARISON MODE: Baseline vs ReLoop")
        print("#"*60)

        # Run baseline first
        baseline_log = run_baseline_with_logging(
            scenario_id=args.scenario,
            problem=problem,
            data=data,
            model=model,
            base_url=base_url,
            api_key=api_key,
            ground_truth=ground_truth,
            baseline_prompt=baseline_prompt,
            verbose=True
        )
        print_summary(baseline_log, "BASELINE SUMMARY")
        save_log(baseline_log, "_baseline")

        # Run ReLoop
        reloop_log = run_reloop_with_logging(
            scenario_id=args.scenario,
            problem=problem,
            data=data,
            model=model,
            base_url=base_url,
            api_key=api_key,
            ground_truth=ground_truth,
            max_iterations=args.max_iter,
            enable_layer6=True,
            early_stop=not args.no_early_stop,
            verbose=True
        )
        print_summary(reloop_log, "RELOOP SUMMARY")
        save_log(reloop_log, "_reloop")

        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"{'Metric':<25} {'Baseline':>15} {'ReLoop':>15} {'Delta':>10}")
        print("-"*60)
        print(f"{'Layers Passed':<25} {baseline_log.layers_passed:>15}/6 {reloop_log.layers_passed:>15}/6 {reloop_log.layers_passed - baseline_log.layers_passed:>+10}")
        print(f"{'LLM Turns':<25} {len(baseline_log.turns):>15} {len(reloop_log.turns):>15} {len(reloop_log.turns) - len(baseline_log.turns):>+10}")
        print(f"{'Duration (s)':<25} {baseline_log.total_duration_s:>15.2f} {reloop_log.total_duration_s:>15.2f} {reloop_log.total_duration_s - baseline_log.total_duration_s:>+10.2f}")

        if baseline_log.objective_value is not None and reloop_log.objective_value is not None:
            print(f"{'Objective':<25} {baseline_log.objective_value:>15,.2f} {reloop_log.objective_value:>15,.2f}")

        # Conclusion
        print("\n" + "-"*60)
        if reloop_log.layers_passed > baseline_log.layers_passed:
            print(f"CONCLUSION: ReLoop improves by {reloop_log.layers_passed - baseline_log.layers_passed} layers")
        elif reloop_log.layers_passed == baseline_log.layers_passed:
            print("CONCLUSION: No difference in layers passed")
        else:
            print(f"CONCLUSION: Baseline performed better (unusual)")

    elif args.baseline:
        # Run baseline only
        log = run_baseline_with_logging(
            scenario_id=args.scenario,
            problem=problem,
            data=data,
            model=model,
            base_url=base_url,
            api_key=api_key,
            ground_truth=ground_truth,
            baseline_prompt=baseline_prompt,
            verbose=True
        )
        print_summary(log, "BASELINE SUMMARY")
        save_log(log, "_baseline")

        # Show conversation summary
        print("\n" + "-"*60)
        print("CONVERSATION TURNS:")
        print("-"*60)
        for turn in log.turns:
            print(f"  [{turn['turn_id']}] {turn['role']}/{turn['step']} - {turn['duration_ms']}ms")

    else:
        # Run ReLoop (default)
        log = run_reloop_with_logging(
            scenario_id=args.scenario,
            problem=problem,
            data=data,
            model=model,
            base_url=base_url,
            api_key=api_key,
            ground_truth=ground_truth,
            max_iterations=args.max_iter,
            enable_layer6=True,
            early_stop=not args.no_early_stop,
            verbose=True
        )
        print_summary(log, "RELOOP SUMMARY")
        save_log(log, "_reloop")

        # Show conversation summary
        print("\n" + "-"*60)
        print("CONVERSATION TURNS:")
        print("-"*60)
        for turn in log.turns:
            print(f"  [{turn['turn_id']}] {turn['role']}/{turn['step']} - {turn['duration_ms']}ms")
