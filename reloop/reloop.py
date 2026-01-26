"""
ReLoop: Reliable LLM-based Optimization Modeling
via Sensitivity-Based Behavioral Verification

Main entry point that orchestrates the three modules:
  Module 1: Structured Generation
  Module 2: Behavioral Verification
  Module 3: Diagnosis-Guided Repair
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time

from .structured_generation import StructuredGenerator, LLMClient
from .behavioral_verification import BehavioralVerifier, VerificationReport
from .diagnosis_repair import DiagnosisRepairer


@dataclass
class ReLoopResult:
    """Result from ReLoop pipeline"""
    code: str
    verified: bool
    iterations: int
    diagnosis_history: List[str] = field(default_factory=list)
    final_report: Optional[VerificationReport] = None
    total_time: float = 0.0
    best_layers_passed: int = 0

    def __str__(self) -> str:
        status = "VERIFIED" if self.verified else "NOT VERIFIED"
        return (
            f"ReLoopResult: {status}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Best layers passed: {self.best_layers_passed}/6\n"
            f"  Time: {self.total_time:.2f}s\n"
            f"  Diagnoses: {len(self.diagnosis_history)}"
        )


@dataclass
class ReLoopConfig:
    """Configuration for ReLoop pipeline"""
    max_iterations: int = 5
    delta: float = 0.2
    epsilon: float = 1e-4
    timeout: int = 60
    enable_layer6: bool = False
    verbose: bool = False


class ReLoop:
    """
    ReLoop: Reliable LLM-based Optimization Modeling

    Key principles:
    - No archetype: Works for any optimization problem
    - L3 is core: Monotonicity check is universal
    - L4 is best-effort: Skip if role cannot be inferred
    - LLM sees schema only: Not actual data values
    - No human feedback: Fully automated
    """

    def __init__(self, llm_client: LLMClient, config: ReLoopConfig = None):
        self.config = config or ReLoopConfig()
        self.generator = StructuredGenerator(llm_client)
        self.verifier = BehavioralVerifier(
            delta=self.config.delta,
            epsilon=self.config.epsilon,
            timeout=self.config.timeout
        )
        self.repairer = DiagnosisRepairer(llm_client)

    def run(self, problem: str, schema: str, data: Dict[str, Any],
            obj_sense: str = "minimize") -> ReLoopResult:
        """Run the complete ReLoop pipeline.

        Strategy:
        - Iteration 1: Generate code using 3-step structured generation
        - Iteration 2+: REPAIR the best code so far using diagnosis
        - ROLLBACK: If repair causes regression, revert and try different approach
        - EARLY STOP: If no progress for 2 consecutive iterations, stop
        """
        start_time = time.time()
        history = []
        best_code, best_layers, best_report = None, -1, None
        current_code = None
        no_progress_count = 0  # Track consecutive iterations without improvement

        if self.config.verbose:
            print("=" * 60)
            print("ReLoop Pipeline Started")
            print("=" * 60)

        for k in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"\n--- Iteration {k + 1}/{self.config.max_iterations} ---")

            # Module 1: Generate or Repair
            if k == 0:
                # First iteration: Structured Generation (3-step)
                if self.config.verbose:
                    print("  [Generate] 3-step structured generation...")
                current_code = self.generator.generate(problem, schema)
            else:
                # Subsequent iterations: REPAIR the best code so far
                if self.config.verbose:
                    print("  [Repair] Diagnosis-guided repair of best code...")
                current_code = self.repairer.repair(
                    code=best_code,
                    diagnosis=history[-1] if history else "",
                    failed_layer=best_report.failed_layer if best_report else 1,
                    history=history[:-1]  # Previous failures (exclude current)
                )

            # Module 2: Behavioral Verification
            report = self.verifier.verify(
                current_code, data, obj_sense,
                enable_layer6=self.config.enable_layer6,
                verbose=self.config.verbose
            )

            layers_passed = report.count_layers_passed()

            # Track best result (only update if strictly better)
            if layers_passed > best_layers:
                best_code, best_layers, best_report = current_code, layers_passed, report
                no_progress_count = 0  # Reset counter on improvement
                if self.config.verbose:
                    print(f"  [Best] New best: {best_layers}/6 layers")
            else:
                no_progress_count += 1
                if self.config.verbose:
                    print(f"  [No Progress] {no_progress_count} iteration(s) without improvement")

            # Success: all layers passed
            if report.passed:
                return ReLoopResult(
                    code=current_code, verified=True, iterations=k + 1,
                    diagnosis_history=history, final_report=report,
                    total_time=time.time() - start_time, best_layers_passed=6
                )

            # Early stopping: no progress for 2 consecutive iterations
            if no_progress_count >= 2 and k >= 2:
                if self.config.verbose:
                    print(f"  [Early Stop] No progress for {no_progress_count} iterations")
                break

            # Module 3: Record diagnosis for next repair iteration
            if report.diagnosis:
                history.append(report.diagnosis)

        return ReLoopResult(
            code=best_code, verified=False, iterations=k + 1,
            diagnosis_history=history, final_report=best_report,
            total_time=time.time() - start_time, best_layers_passed=best_layers
        )

    def run_baseline(self, problem: str, schema: str, data: Dict[str, Any],
                     obj_sense: str = "minimize") -> ReLoopResult:
        """Run baseline (single shot, no verification loop)."""
        start_time = time.time()
        code = self.generator.generate_baseline(problem, schema)
        report = self.verifier.verify(code, data, obj_sense,
                                       enable_layer6=self.config.enable_layer6)
        return ReLoopResult(
            code=code, verified=report.passed, iterations=1,
            diagnosis_history=[report.diagnosis] if report.diagnosis else [],
            final_report=report, total_time=time.time() - start_time,
            best_layers_passed=report.count_layers_passed()
        )


def run_reloop(problem: str, schema: str, data: Dict[str, Any],
               llm_client: LLMClient, obj_sense: str = "minimize",
               max_iterations: int = 5, verbose: bool = False) -> ReLoopResult:
    """Convenience function to run ReLoop."""
    config = ReLoopConfig(max_iterations=max_iterations, verbose=verbose)
    pipeline = ReLoop(llm_client, config)
    return pipeline.run(problem, schema, data, obj_sense)


def verify_code(code: str, data: Dict[str, Any], obj_sense: str = "minimize",
                verbose: bool = False) -> VerificationReport:
    """Convenience function to verify code without generation."""
    verifier = BehavioralVerifier()
    return verifier.verify(code, data, obj_sense, verbose=verbose)
