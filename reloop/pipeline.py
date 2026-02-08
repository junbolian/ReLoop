"""
ReLoop Main Pipeline

Implements the complete Generate → Verify → Repair loop with:
- Chain-of-Thought generation (single API call with step-by-step reasoning)
- L1 FATAL handling with regeneration (not termination)
- L2 Anomaly Detection (bidirectional perturbation)
- L4 Adversarial Direction Analysis (LLM-based with Accept/Reject)
- Conservative repair strategy: only fix ERROR/WARNING, not INFO
- Guaranteed output when possible

Repair Trigger Rules:
- FATAL (L1): Triggers regeneration
- ERROR (L2 anomaly): Must fix (physically impossible behavior)
- L4 Issues: Review & Accept/Reject with LLM
- WARNING (L5 cpt_missing): Should fix
- INFO (L2 no_effect, L3, etc.): Do NOT trigger repair
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from .generation import CodeGenerator
from .verification import ReLoopVerifier, VerificationReport, Severity
from .repair import CodeRepairer, RepairResult
from .prompts import format_diagnostic_report
from .l4_adversarial import (
    L4AdversarialVerifier,
    L4VerifyResult,
    L4RepairDecision,
    L4_REPAIR_PROMPT,
    should_exit_l4_loop
)


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
        max_l4_rejections: int = 2,
        enable_cpt: bool = True,
        enable_l4_adversarial: bool = True,
        use_structured_generation: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            llm_client: LLM client with generate(prompt, system=None) method
            max_repair_iterations: Max repair attempts for ERROR/WARNING issues
            max_regeneration_attempts: Max regeneration attempts for L1 FATAL
            max_l4_rejections: Max rejections per param before L4 downgrades to INFO
            enable_cpt: Enable L5 CPT layer
            enable_l4_adversarial: Enable L4 Adversarial Direction Analysis
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
        self.enable_l4_adversarial = enable_l4_adversarial
        self.verbose = verbose

        # L4 Adversarial Verifier
        if enable_l4_adversarial and llm_client:
            self.l4_verifier = L4AdversarialVerifier(
                llm_client=llm_client,
                max_rejections=max_l4_rejections
            )
        else:
            self.l4_verifier = None

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
                if self.verbose:
                    print(f"[Pipeline] Generated code in {gen_time:.2f}s")
            except Exception as e:
                if self.verbose:
                    print(f"[Pipeline] Generation failed: {e}")
                code = ""
                # Leave report=None to trigger regeneration loop

        # Step 2: Initial verification (only if we have code and generation succeeded)
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

        # Step 4: Run L4 Adversarial Loop (if enabled)
        l4_time = 0.0
        if self.enable_l4_adversarial and self.l4_verifier and report.status != 'FAILED':
            if self.verbose:
                print("[Pipeline] Running L4 Adversarial Direction Analysis")

            l4_start = time.time()
            baseline_obj = report.objective or 0.0

            l4_results, l4_exit_reason, l4_code = self._run_l4_adversarial_loop(
                code=code,
                data=data,
                baseline_obj=baseline_obj,
                problem_description=problem_description,
                report=report,
                max_l4_iterations=3
            )
            l4_time = time.time() - l4_start

            if self.verbose:
                print(f"[Pipeline] L4 exit reason: {l4_exit_reason}")

            # If L4 produced fixed code, update and re-verify
            if l4_code != code and l4_exit_reason in ("accepted_fixed", "max_iterations"):
                code = l4_code
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
                    print(f"[Pipeline] After L4 fix: {report.status}")

        # Step 5: Handle ERROR/WARNING with repair (INFO does NOT trigger)
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

            # Stop if code unchanged OR verification result unchanged OR objective unchanged
            if repaired_code == code:
                if self.verbose:
                    print("[Pipeline] Repair produced no code changes, stopping")
                break
            if history and history[-1][1].status == report.status:
                if self.verbose:
                    print("[Pipeline] Repair did not change verification status, stopping")
                break
            if history and history[-1][1].objective == report.objective:
                if self.verbose:
                    print("[Pipeline] Repair did not change objective, stopping")
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
        - critical_errors: ERROR level (L2 anomaly) - Must fix
        - should_fix: WARNING level (L5 cpt_missing) - Should fix
        - for_reference: INFO level (L2 no_effect, L3, etc.) - Do NOT fix

        Note: L4 issues are handled separately by adversarial mechanism.
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

            # L2 ERROR - anomaly (both directions improve)
            elif r.layer == "L2" and r.severity == Severity.ERROR:
                if r.check == "anomaly":
                    item["action"] = "Fix structural error - both increasing and decreasing improve objective"
                else:
                    item["action"] = (r.details or {}).get("repair_hint", "Fix this error")
                critical_errors.append(item)

            # L4 - handled by adversarial mechanism, skip here
            elif r.layer == "L4":
                # L4 issues go to for_reference; actual L4 repair handled separately
                item["action"] = "L4 issues handled by adversarial mechanism"
                for_reference.append(item)

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

    # =========================================================================
    # L4 Adversarial Loop
    # =========================================================================

    def _run_l4_adversarial_loop(
        self,
        code: str,
        data: Dict[str, Any],
        baseline_obj: float,
        problem_description: str,
        report: VerificationReport,
        max_l4_iterations: int = 3
    ) -> Tuple[List[L4VerifyResult], str, str]:
        """
        Run L4 Adversarial Direction Analysis loop.

        Args:
            code: Current code
            data: Problem data
            baseline_obj: Baseline objective value
            problem_description: Problem description
            report: Current verification report
            max_l4_iterations: Max iterations for L4 loop

        Returns:
            Tuple[l4_results, exit_reason, final_code]

        Exit reasons:
            - "all_pass": All L4 checks passed
            - "all_rejected_others_pass": All L4 rejected, other layers PASS
            - "max_rejections": Max rejections reached, downgraded to INFO
            - "accepted_fixed": Some accepted and code was fixed
            - "max_iterations": Reached max L4 iterations
            - "no_violations": No violations found
            - "disabled": L4 disabled or no LLM client
        """
        if not self.l4_verifier or not self.enable_l4_adversarial:
            return [], "disabled", code

        # Reset L4 verifier state
        self.l4_verifier.reset()

        # Get parameters already flagged by L2 (exclude from L4)
        l2_error_params = self._get_l2_error_params(report)

        current_code = code
        l4_results = []
        # After first iteration, constrain future analyses to only the originally
        # flagged parameters (or their rejected subset) to avoid surfacing new issues.
        allowed_params: Optional[List[str]] = None

        for l4_iter in range(max_l4_iterations):
            if self.verbose:
                print(f"  [L4] Adversarial loop iteration {l4_iter + 1}")

            # Step 1: L4 Verify
            l4_results = self.l4_verifier.verify(
                code=current_code,
                data=data,
                baseline_obj=baseline_obj,
                problem_description=problem_description,
                params=allowed_params,
                exclude_params=l2_error_params,
                executor=self.verifier.executor
            )

            # Step 2: Check if all PASS (no violations)
            violations = [
                r for r in l4_results
                if r.is_violation and r.confidence >= self.l4_verifier.confidence_threshold
            ]

            # Freeze the parameter set after first pass to prevent new params
            # from appearing in subsequent iterations.
            if allowed_params is None:
                allowed_params = [v.param for v in violations]
            else:
                # Keep only violations within the locked set
                violations = [v for v in violations if v.param in allowed_params]

            if not violations:
                if self.verbose:
                    print("  [L4] No violations found - PASS")
                return l4_results, "all_pass", current_code

            if self.verbose:
                print(f"  [L4] Found {len(violations)} violations")

            # Step 3: Get repair decisions (Accept/Reject)
            decisions, fixed_code = self._get_l4_repair_decisions(
                code=current_code,
                problem_description=problem_description,
                data=data,
                l4_results=l4_results
            )

            if not decisions:
                if self.verbose:
                    print("  [L4] No decisions received, treating as all rejected")
                # Check if other layers pass
                if self._check_other_layers_pass(report):
                    return l4_results, "all_rejected_others_pass", current_code
                continue

            # Step 4: Process decisions
            decision_result = self.l4_verifier.process_repair_decisions(
                decisions=decisions,
                verify_results=l4_results
            )

            accepted = decision_result["accepted"]
            rejected = decision_result["rejected"]
            # Limit any further L4_VERIFY runs to only the still-rejected params
            # so new parameters/issues cannot appear in later iterations.
            allowed_params = [d.param for d in rejected]

            if self.verbose:
                print(f"  [L4] Accepted: {len(accepted)}, Rejected: {len(rejected)}")

            # Step 5: Handle accepted (apply fix)
            if accepted and fixed_code and fixed_code != current_code:
                # Verify fixed code executes
                try:
                    fix_result = self.verifier.executor.execute(fixed_code, data)
                    if fix_result.get("status") == "OPTIMAL":
                        current_code = fixed_code
                        if self.verbose:
                            print("  [L4] Applied fix, continuing to verify")
                        continue
                except Exception:
                    pass

            # Step 6: Handle all rejected
            if rejected and not accepted:
                if decision_result["should_reverify"]:
                    if self.verbose:
                        print("  [L4] All rejected, re-verifying with context")
                    # Next round only reconsider rejected params
                    allowed_params = [d.param for d in rejected if d.param in (allowed_params or [])]
                    continue

                # Check if other layers pass
                if self._check_other_layers_pass(report):
                    if self.verbose:
                        print("  [L4] All rejected + others PASS")
                    return l4_results, "all_rejected_others_pass", current_code

            # Step 7: Check max rejections
            l4_status = self.l4_verifier.get_final_status(l4_results)
            if l4_status == "INFO":
                if self.verbose:
                    print("  [L4] Max rejections reached, downgraded to INFO")
                return l4_results, "max_rejections", current_code

        if self.verbose:
            print("  [L4] Max iterations reached")
        return l4_results, "max_iterations", current_code

    def _get_l4_repair_decisions(
        self,
        code: str,
        problem_description: str,
        data: Dict[str, Any],
        l4_results: List[L4VerifyResult]
    ) -> Tuple[List[L4RepairDecision], Optional[str]]:
        """
        Call repair LLM to get Accept/Reject decisions for L4 diagnostics.

        Returns:
            Tuple[decisions, fixed_code]
        """
        # Format L4 diagnostics
        diagnostics = self.l4_verifier.format_diagnostics_for_repair(l4_results)

        if diagnostics == "No direction violations detected.":
            return [], None

        # Build repair prompt
        from .prompts import describe_data_schema
        data_schema = describe_data_schema(data)

        prompt = L4_REPAIR_PROMPT.format(
            problem_description=problem_description,
            code=code,
            l4_diagnostics=diagnostics
        )

        try:
            response = self.llm_client.generate(prompt)
            decisions, fixed_code = self.l4_verifier.parse_repair_response(response)
            return decisions, fixed_code
        except Exception as e:
            if self.verbose:
                print(f"  [L4] Failed to get repair decisions: {e}")
            return [], None

    def _get_l2_error_params(self, report: VerificationReport) -> List[str]:
        """Get parameters already flagged as ERROR by L2."""
        params = []
        for r in report.layer_results:
            if r.layer == "L2" and r.severity == Severity.ERROR:
                if r.details and "param" in r.details:
                    params.append(r.details["param"])
        return params

    def _check_other_layers_pass(self, report: VerificationReport) -> bool:
        """Check if L2, L3, L5 all PASS or INFO (no ERROR/WARNING)."""
        for r in report.layer_results:
            if r.layer == "L4":
                continue  # Skip L4
            if r.severity in [Severity.ERROR, Severity.WARNING]:
                return False
        return True

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
