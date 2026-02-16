"""
ReLoop Main Pipeline

Implements the complete Generate -> Verify -> Repair loop with:
- Chain-of-Thought generation (single API call with step-by-step reasoning)
- L1 FATAL handling with regeneration (not termination)
- L2 Behavioral Testing (CPT + OPT)
- Conservative repair strategy: only WARNING trigger repair, INFO is reference only
- Guaranteed output when possible

Two-Layer Architecture:
- L1: Execution Verification (blocking) + duality check
- L2: Behavioral Testing (CPT + OPT)

Repair Trigger Rules:
- FATAL (L1): Triggers regeneration
- WARNING (L2 CPT/OPT missing): Should fix / reference
- INFO (L1 duality, L2 uncertain): Do NOT trigger repair
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from .generation import CodeGenerator
from .verification import (
    ReLoopVerifier, VerificationReport, Severity, Diagnostic,
    layer_results_to_diagnostics,
)
from .repair import CodeRepairer, RepairResult
from .prompts import (
    format_diagnostic_report, build_repair_prompt, REPAIR_PROMPT_SYSTEM,
    describe_data_schema, format_data_instructions,
)
from .repair_safety import validate_repair_code, SAFETY_RE_REPAIR_PROMPT

logger = logging.getLogger(__name__)


# Status ranking for regression detection
_STATUS_ORDER = {'FAILED': 0, 'ERRORS': 1, 'WARNINGS': 2, 'VERIFIED': 3}


def _is_regression(before: 'VerificationReport', after: 'VerificationReport') -> bool:
    """True if *after* is strictly worse than *before*.

    Regression cases:
    - Crash regression: before had an objective, after crashed (None)
    - Status regression: after's status rank is lower than before's
    - Value regression: same status but objective shifted > 4%
    """
    if before.objective is not None and after.objective is None:
        return True
    if _STATUS_ORDER.get(after.status, -1) < _STATUS_ORDER.get(before.status, -1):
        return True
    # Value regression: same status, objective drifted significantly
    if (before.objective is not None and after.objective is not None
            and before.status == after.status):
        denom = max(abs(before.objective), 1e-12)
        rel_change = abs(after.objective - before.objective) / denom
        if rel_change > 0.04:
            return True
    return False


@dataclass
class PipelineResult:
    """Result of running the ReLoop pipeline."""
    final_code: str
    final_report: VerificationReport
    iterations: int
    history: List[Tuple[str, VerificationReport]]
    total_time: float
    generation_time: float
    verification_time: float
    repair_time: float
    success: bool
    improved: bool
    regeneration_count: int = 0
    # Intermediate ablation checkpoints (recorded automatically)
    l1_checkpoint_obj: Optional[float] = None    # Objective after L1 verify + regeneration
    l1_checkpoint_status: str = ""               # Status after L1
    l2_checkpoint_obj: Optional[float] = None    # Objective after L2 behavioral testing
    l2_checkpoint_status: str = ""               # Status after L2


@dataclass
class RepairContext:
    """
    Context for repair decision.

    Separates issues into:
    - critical_errors: ERROR level, must fix (99%+ confidence)
    - should_fix: WARNING level, should fix (80%+ confidence)
    - for_reference: INFO level, likely normal, do NOT fix
    """
    critical_errors: List[Dict] = field(default_factory=list)
    should_fix: List[Dict] = field(default_factory=list)
    for_reference: List[Dict] = field(default_factory=list)
    should_trigger: bool = False


class ReLoopPipeline:
    """
    Complete ReLoop Pipeline: Generate -> Verify -> Repair loop.

    Two-Layer Architecture:
    - L1: Execution Verification (blocking) + duality check
    - L2: Behavioral Testing (CPT + OPT)

    Key Features:
    - Chain-of-Thought generation: Single API call with step-by-step reasoning
    - L1 FATAL triggers regeneration (not termination)
    - WARNING triggers repair, INFO does NOT
    - Guaranteed output when L1 passes

    Usage:
        pipeline = ReLoopPipeline(llm_client)
        result = pipeline.run(problem_description, data)
    """

    def __init__(
        self,
        llm_client,
        max_repair_iterations: int = 3,
        max_regeneration_attempts: int = 3,
        enable_cpt: bool = True,
        use_structured_generation: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            llm_client: LLM client with generate(prompt, system=None) method
            max_repair_iterations: Max repair attempts for WARNING issues
            max_regeneration_attempts: Max regeneration attempts for L1 FATAL
            enable_cpt: Enable L2 behavioral testing (CPT + OPT)
            use_structured_generation: Use CoT generation pipeline
            verbose: Print progress messages
        """
        self.generator = CodeGenerator(
            llm_client,
            use_structured_generation=use_structured_generation
        )
        self.verifier = ReLoopVerifier(llm_client=llm_client)
        self.repairer = CodeRepairer(llm_client)
        self.llm_client = llm_client
        self.max_repair_iterations = max_repair_iterations
        self.max_regeneration_attempts = max_regeneration_attempts
        self.enable_cpt = enable_cpt
        self.verbose = verbose

    def run(
        self,
        problem_description: str,
        data: Dict[str, Any],
        initial_code: Optional[str] = None
    ) -> PipelineResult:
        """
        Run the complete pipeline.

        Flow:
        1. Generate code (CoT or single-stage)
        2. Verify with L1 + L2 (execution + behavioral testing)
        3. If L1 FATAL: regenerate (up to max_regeneration_attempts)
        4. If WARNING: repair (up to max_repair_iterations)
        5. INFO does NOT trigger repair (likely normal)
        6. Return result (always has output if any L1 passes)

        Args:
            problem_description: Natural language problem description
            data: Problem data dictionary
            initial_code: Optional pre-generated code to verify/repair

        Returns:
            PipelineResult with final code, report, and metrics
        """
        start_time = time.time()
        history = []
        gen_time = 0.0
        verify_time = 0.0
        repair_time = 0.0
        regeneration_count = 0

        # Step 1: Generate initial code (or use provided)
        report = None
        code = None
        if initial_code:
            code = initial_code
            if self.verbose:
                print("[Pipeline] Using provided initial code")
        else:
            try:
                gen_start = time.time()
                code = self.generator.generate(problem_description, data)
                gen_time = time.time() - gen_start

                # If extraction succeeded, use extracted data for execution
                if self.generator.extracted_data is not None:
                    data = self.generator.extracted_data
                    if self.verbose:
                        print(f"[Pipeline] Using extracted data ({len(data)} keys)")

                if self.verbose:
                    print(f"[Pipeline] Generated code in {gen_time:.2f}s")
            except Exception as e:
                if self.verbose:
                    print(f"[Pipeline] Generation failed: {e}")
                code = ""
                # Leave report=None to trigger regeneration loop

        # Step 2: Initial verification (L1 + L2 behavioral)
        if report is None and code:
            verify_start = time.time()
            report = self.verifier.verify(
                code, data,
                problem_description=problem_description,
                enable_cpt=self.enable_cpt,
                verbose=self.verbose
            )
            verify_time += time.time() - verify_start
            history.append((code, report))

            if self.verbose:
                print(f"[Pipeline] Initial verification: {report.status}")

        # Step 3: Handle L1 FATAL with regeneration
        error_message = None
        while (report is None or report.status == 'FAILED') and regeneration_count < self.max_regeneration_attempts:
            regeneration_count += 1
            if self.verbose:
                print(f"[Pipeline] L1 FATAL/Generation failure - Regeneration attempt {regeneration_count}")

            if report is not None:
                error_message = self._get_l1_error(report)
            elif error_message is None:
                error_message = "Initial generation failed"

            gen_start = time.time()
            try:
                code = self.generator.regenerate(
                    problem_description=problem_description,
                    failed_code=code or "",
                    error_message=error_message,
                    data=data
                )
            except Exception as e:
                if self.verbose:
                    print(f"[Pipeline] Regeneration failed: {e}")
                break
            gen_time += time.time() - gen_start

            verify_start = time.time()
            report = self.verifier.verify(
                code, data,
                problem_description=problem_description,
                enable_cpt=self.enable_cpt,
                verbose=self.verbose
            )
            verify_time += time.time() - verify_start
            history.append((code, report))

            if self.verbose:
                print(f"[Pipeline] After regeneration: {report.status}")

        # -- Ablation checkpoint: after L1 verify + regeneration --
        l1_checkpoint_obj = report.objective if report else None
        l1_checkpoint_status = report.status if report else "FAILED"

        # -- Ablation checkpoint: after L2 behavioral testing --
        # L2 runs inside verify() and doesn't change code, so same objective
        l2_checkpoint_obj = report.objective if report else None
        l2_checkpoint_status = report.status if report else "FAILED"

        # Step 4: Handle WARNING with repair (INFO does NOT trigger)
        # Uses unified Diagnostic schema + build_repair_prompt()
        # Save pre-repair state for regression rollback
        pre_repair_code = code
        pre_repair_report = report
        repair_iteration = 0
        all_diagnostics = self._collect_diagnostics(report)

        # Skip repair if already VERIFIED with no actionable issues
        has_actionable = any(
            d.severity in ("ERROR", "WARNING") and d.triggers_repair
            for d in all_diagnostics
        )
        skip_repair = (
            pre_repair_report.status == "VERIFIED"
            and not has_actionable
        )
        if skip_repair and self.verbose:
            print("[Pipeline] VERIFIED with no actionable issues, skipping repair")

        while (not skip_repair
               and any(d.triggers_repair for d in all_diagnostics)
               and repair_iteration < self.max_repair_iterations):
            repair_iteration += 1
            n_actionable = sum(1 for d in all_diagnostics if d.triggers_repair)
            n_info = sum(1 for d in all_diagnostics if not d.triggers_repair)
            if self.verbose:
                print(f"[Pipeline] Repair iteration {repair_iteration}")
                print(f"  Actionable issues: {n_actionable}")
                print(f"  Reference (no action): {n_info}")

            repair_start = time.time()
            try:
                data_schema = describe_data_schema(data)
                repair_prompt = build_repair_prompt(
                    diagnostics=all_diagnostics,
                    code=code,
                    problem_desc=problem_description,
                    data_structure=data_schema,
                    current_obj=report.objective,
                )
                if repair_prompt is None:
                    break

                response = self.llm_client.generate(
                    repair_prompt, system=REPAIR_PROMPT_SYSTEM
                )
                repaired_code = self.repairer._extract_code(response)

                # Safety guardrail: validate repair code before accepting
                repaired_code = self._apply_safety_guardrail(
                    repaired_code, code, data, repair_prompt
                )
            except Exception as e:
                if self.verbose:
                    print(f"[Pipeline] Repair failed: {e}")
                break
            repair_time += time.time() - repair_start

            # Stop if repair produced identical code
            if repaired_code == code:
                if self.verbose:
                    print("[Pipeline] Repair produced no code changes, stopping")
                break

            # Apply repaired code and re-verify
            prev_report = report
            code = repaired_code

            verify_start = time.time()
            report = self.verifier.verify(
                code, data,
                problem_description=problem_description,
                enable_cpt=self.enable_cpt,
                verbose=self.verbose
            )
            verify_time += time.time() - verify_start
            history.append((code, report))

            # Re-collect diagnostics from updated report
            all_diagnostics = self._collect_diagnostics(report)

            if self.verbose:
                print(f"[Pipeline] After repair: {report.status}")

            # Stop if repair didn't help (status AND objective both unchanged)
            if (report.status == prev_report.status
                    and report.objective == prev_report.objective):
                if self.verbose:
                    print("[Pipeline] Repair did not change status or objective, stopping")
                break

            # Regression guard: rollback if repair made things worse than pre-repair baseline
            if _is_regression(pre_repair_report, report):
                if self.verbose:
                    pre_obj = pre_repair_report.objective
                    post_obj = report.objective
                    print(f"[Pipeline] Repair regression detected "
                          f"(obj: {pre_obj} -> {post_obj}, "
                          f"status: {pre_repair_report.status} -> {report.status}), "
                          f"rolling back")
                code = pre_repair_code
                report = pre_repair_report
                break

            # If repair caused L1 FATAL, try to regenerate
            if report.status == 'FAILED' and regeneration_count < self.max_regeneration_attempts:
                regeneration_count += 1
                if self.verbose:
                    print(f"[Pipeline] Repair caused FATAL - Regeneration attempt {regeneration_count}")

                error_message = self._get_l1_error(report)
                gen_start = time.time()
                try:
                    code = self.generator.regenerate(
                        problem_description=problem_description,
                        failed_code=code,
                        error_message=error_message,
                        data=data
                    )
                    gen_time += time.time() - gen_start

                    verify_start = time.time()
                    report = self.verifier.verify(
                        code, data,
                        problem_description=problem_description,
                        enable_cpt=self.enable_cpt,
                        verbose=self.verbose
                    )
                    verify_time += time.time() - verify_start
                    history.append((code, report))
                except Exception:
                    pass

            # Re-collect diagnostics for next iteration
            all_diagnostics = self._collect_diagnostics(report)

        # Determine success and improvement
        success = report.status in ['VERIFIED', 'WARNINGS', 'ERRORS']
        improved = self._is_improved(history[0][1], report) if len(history) > 1 else False

        return PipelineResult(
            final_code=code,
            final_report=report,
            iterations=len(history),
            history=history,
            total_time=time.time() - start_time,
            generation_time=gen_time,
            verification_time=verify_time,
            repair_time=repair_time,
            success=success,
            improved=improved,
            regeneration_count=regeneration_count,
            l1_checkpoint_obj=l1_checkpoint_obj,
            l1_checkpoint_status=l1_checkpoint_status,
            l2_checkpoint_obj=l2_checkpoint_obj,
            l2_checkpoint_status=l2_checkpoint_status,
        )

    def _collect_diagnostics(
        self,
        report: VerificationReport,
    ) -> List[Diagnostic]:
        """
        Collect unified Diagnostic objects from verification report.

        Converts all LayerResult (L1 + L2 behavioral) into Diagnostic schema.
        """
        return layer_results_to_diagnostics(
            report.layer_results,
            baseline_obj=report.objective,
            delta=self.verifier.delta,
        )

    def _get_l1_error(self, report: VerificationReport) -> str:
        """Extract L1 error message from report."""
        for r in report.layer_results:
            if r.layer == "L1" and r.severity == Severity.FATAL:
                return r.message
        return "Unknown execution error"

    def _is_improved(
        self,
        before: VerificationReport,
        after: VerificationReport
    ) -> bool:
        """Check if verification improved."""
        status_order = {
            'FAILED': 0,
            'ERRORS': 1,
            'WARNINGS': 2,
            'VERIFIED': 3
        }
        return status_order.get(after.status, 0) > status_order.get(before.status, 0)

    def _apply_safety_guardrail(
        self,
        repaired_code: str,
        original_code: str,
        data: Dict[str, Any],
        prompt: Optional[str] = None,
    ) -> str:
        """
        Validate repair code and retry once if safety violations found.

        Strategy A: safety violations do NOT consume repair budget.
        Max 1 safety retry; if second attempt also fails, keep original code.
        """
        is_safe, violations = validate_repair_code(
            repaired_code, original_code, data
        )

        if is_safe:
            logger.info("Repair safety check: PASS")
            return repaired_code

        # First violation — log and attempt re-repair
        violation_msg = "\n".join(violations)
        logger.warning(f"Repair safety check: REJECTED\n{violation_msg}")
        if self.verbose:
            print(f"[Pipeline] Repair safety REJECTED: {len(violations)} violation(s)")
            for v in violations:
                print(f"  - {v.split(chr(10))[0]}")

        # Attempt guided re-repair (does not consume repair budget)
        try:
            safety_prompt = SAFETY_RE_REPAIR_PROMPT.format(
                violations=violation_msg
            )
            if prompt:
                re_repair_prompt = safety_prompt + prompt
            else:
                re_repair_prompt = (
                    safety_prompt
                    + f"\n\n## Current Code\n```python\n{original_code}\n```\n\n"
                    + "Return the COMPLETE fixed code in a ```python block."
                )

            response = self.llm_client.generate(
                re_repair_prompt, system=REPAIR_PROMPT_SYSTEM
            )
            retried_code = self.repairer._extract_code(response)

            is_safe_2, violations_2 = validate_repair_code(
                retried_code, original_code, data
            )

            if is_safe_2:
                logger.info("Repair safety re-check: PASS (after guided retry)")
                if self.verbose:
                    print("[Pipeline] Safety re-repair succeeded")
                return retried_code

            violation_msg_2 = "\n".join(violations_2)
            logger.warning(
                f"Repair safety re-check: REJECTED again. "
                f"Keeping original code.\n{violation_msg_2}"
            )
            if self.verbose:
                print("[Pipeline] Safety re-repair also failed, keeping original code")

        except Exception as e:
            logger.warning(f"Safety re-repair failed with exception: {e}")
            if self.verbose:
                print(f"[Pipeline] Safety re-repair exception: {e}")

        return original_code


def run_reloop(
    problem_description: str,
    data: Dict[str, Any],
    llm_client,
    code: Optional[str] = None,
    max_iterations: int = 3,
    max_regenerations: int = 3,
    enable_cpt: bool = True,
    use_structured_generation: bool = True,
    verbose: bool = False
) -> PipelineResult:
    """
    Convenience function to run the ReLoop pipeline.

    Args:
        problem_description: Natural language problem description
        data: Problem data dictionary
        llm_client: LLM client with generate(prompt, system=None) method
        code: Optional pre-generated code
        max_iterations: Maximum repair iterations
        max_regenerations: Maximum regeneration attempts (for L1 FATAL)
        enable_cpt: Enable L2 behavioral testing (CPT + OPT)
        use_structured_generation: Use CoT generation pipeline
        verbose: Print progress messages

    Returns:
        PipelineResult with final code, report, and metrics
    """
    pipeline = ReLoopPipeline(
        llm_client,
        max_repair_iterations=max_iterations,
        max_regeneration_attempts=max_regenerations,
        enable_cpt=enable_cpt,
        use_structured_generation=use_structured_generation,
        verbose=verbose
    )
    return pipeline.run(problem_description, data, initial_code=code)
