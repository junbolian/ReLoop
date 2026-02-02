"""
ReLoop Main Pipeline

Implements the complete Generate → Verify → Repair loop with:
- Chain-of-Thought generation (single API call with step-by-step reasoning)
- L1 FATAL handling with regeneration (not termination)
- Conservative repair strategy: only fix ERROR/WARNING, not INFO
- Guaranteed output when possible

Repair Trigger Rules:
- FATAL (L1): Triggers regeneration
- ERROR (L2 monotonicity, L4 anomaly): Must fix
- WARNING (L5 cpt_missing): Should fix
- INFO (L3, L4 no_effect, L4 sensitivity, L5 uncertain): Do NOT trigger repair
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from .generation import CodeGenerator
from .verification import ReLoopVerifier, VerificationReport, Severity
from .repair import CodeRepairer
from .prompts import format_diagnostic_report


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
    Complete ReLoop Pipeline: Generate → Verify → Repair loop.

    Key Features:
    - Chain-of-Thought generation: Single API call with step-by-step reasoning
    - L1 FATAL triggers regeneration (not termination)
    - ERROR/WARNING triggers repair, INFO does NOT
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
        verbose: bool = False
    ):
        """
        Args:
            llm_client: LLM client with generate(prompt, system=None) method
            max_repair_iterations: Max repair attempts for ERROR/WARNING issues
            max_regeneration_attempts: Max regeneration attempts for L1 FATAL
            enable_cpt: Enable L5 CPT layer
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
        2. Verify with L1-L5
        3. If L1 FATAL: regenerate (up to max_regeneration_attempts)
        4. If ERROR/WARNING: repair (up to max_repair_iterations)
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
        if initial_code:
            code = initial_code
            if self.verbose:
                print("[Pipeline] Using provided initial code")
        else:
            gen_start = time.time()
            code = self.generator.generate(problem_description, data)
            gen_time = time.time() - gen_start
            if self.verbose:
                print(f"[Pipeline] Generated code in {gen_time:.2f}s")

        # Step 2: Initial verification
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
        while report.status == 'FAILED' and regeneration_count < self.max_regeneration_attempts:
            regeneration_count += 1
            if self.verbose:
                print(f"[Pipeline] L1 FATAL - Regeneration attempt {regeneration_count}")

            error_message = self._get_l1_error(report)

            gen_start = time.time()
            try:
                code = self.generator.regenerate(
                    problem_description=problem_description,
                    failed_code=code,
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

        # Step 4: Handle ERROR/WARNING with repair (INFO does NOT trigger)
        repair_iteration = 0
        ctx = self._analyze_verification_results(report)

        while ctx.should_trigger and repair_iteration < self.max_repair_iterations:
            repair_iteration += 1
            if self.verbose:
                print(f"[Pipeline] Repair iteration {repair_iteration}")
                print(f"  Critical errors: {len(ctx.critical_errors)}")
                print(f"  Should fix: {len(ctx.should_fix)}")
                print(f"  For reference (no action): {len(ctx.for_reference)}")

            repair_start = time.time()
            try:
                repaired_code = self.repairer.repair_with_context(
                    code=code,
                    data=data,
                    problem_description=problem_description,
                    critical_errors=ctx.critical_errors,
                    should_fix=ctx.should_fix,
                    for_reference=ctx.for_reference
                )
            except Exception as e:
                if self.verbose:
                    print(f"[Pipeline] Repair failed: {e}")
                break
            repair_time += time.time() - repair_start

            if repaired_code == code:
                if self.verbose:
                    print("[Pipeline] Repair produced no changes, stopping")
                break

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

            if self.verbose:
                print(f"[Pipeline] After repair: {report.status}")

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

            # Re-analyze for next iteration
            ctx = self._analyze_verification_results(report)

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
            regeneration_count=regeneration_count
        )

    def _analyze_verification_results(self, report: VerificationReport) -> RepairContext:
        """
        Analyze verification results to determine repair strategy.

        Classification:
        - critical_errors: ERROR level (L2 monotonicity, L4 anomaly) - Must fix
        - should_fix: WARNING level (L5 cpt_missing) - Should fix
        - for_reference: INFO level (L3, L4 no_effect, etc.) - Do NOT fix

        Only trigger repair if critical_errors or should_fix is non-empty.
        """
        critical_errors = []
        should_fix = []
        for_reference = []

        for r in report.layer_results:
            item = {
                "layer": r.layer,
                "check": r.check,
                "severity": r.severity.value,
                "message": r.message,
                "details": r.details or {},
                "is_likely_normal": (r.details or {}).get("is_likely_normal", False)
            }

            # L1 FATAL - handled separately by regeneration
            if r.layer == "L1" and r.severity == Severity.FATAL:
                item["action"] = "Fix code error or model feasibility"
                critical_errors.append(item)

            # ERROR level - must fix (L2 monotonicity, L4 anomaly)
            elif r.severity == Severity.ERROR:
                if r.layer == "L2":
                    item["action"] = "Fix constraint direction (>= vs <=)"
                elif r.layer == "L4" and r.check == "anomaly":
                    item["action"] = "Fix impossible parameter behavior"
                else:
                    item["action"] = (r.details or {}).get("repair_hint", "Fix this error")
                critical_errors.append(item)

            # WARNING level - should fix (L5 cpt_missing)
            elif r.severity == Severity.WARNING:
                item["action"] = (r.details or {}).get("repair_hint", "Fix this issue")
                should_fix.append(item)

            # INFO level - for reference only, do NOT fix
            elif r.severity == Severity.INFO:
                item["action"] = "No action needed - likely normal"
                for_reference.append(item)

        # Only trigger if there are ERROR or WARNING issues
        should_trigger = len(critical_errors) > 0 or len(should_fix) > 0

        return RepairContext(
            critical_errors=critical_errors,
            should_fix=should_fix,
            for_reference=for_reference,
            should_trigger=should_trigger
        )

    def _get_l1_error(self, report: VerificationReport) -> str:
        """Extract L1 error message from report."""
        for r in report.layer_results:
            if r.layer == "L1" and r.severity == Severity.FATAL:
                return r.message
        return "Unknown execution error"

    def _needs_repair(self, report: VerificationReport) -> bool:
        """Check if report indicates repair is needed."""
        ctx = self._analyze_verification_results(report)
        return ctx.should_trigger

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
        max_iterations: Maximum repair iterations (for ERROR/WARNING issues)
        max_regenerations: Maximum regeneration attempts (for L1 FATAL)
        enable_cpt: Enable L5 CPT layer
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
