"""
ReLoop Main Pipeline

Implements the complete Generate -> Verify -> Repair loop with:
- Chain-of-Thought generation (single API call with step-by-step reasoning)
- L1 FATAL handling with regeneration (not termination)
- L2 Direction Consistency Analysis (LLM-based with Accept/Reject)
- L3 Constraint Presence Testing (optional)
- L4 Specification Compliance Checking (optional, LLM-based white-box review)
- Conservative repair strategy: only fix ERROR/WARNING, not INFO
- Guaranteed output when possible

Four-Layer Architecture:
- L1: Execution Verification (blocking) + duality check
- L2: Direction Consistency Analysis (adversarial LLM debate)
- L3: Constraint Presence Testing (CPT)
- L4: Specification Compliance Checking (white-box code review)

Repair Trigger Rules:
- FATAL (L1): Triggers regeneration
- ERROR (L2 direction accepted, L4 spec violation): Must fix
- WARNING (L3 cpt_missing, L4 spec uncertain): Should fix / reference
- INFO (L1 duality, L2 rejected): Do NOT trigger repair
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

from .generation import CodeGenerator
from .verification import (
    ReLoopVerifier, VerificationReport, Severity, Diagnostic,
    layer_results_to_diagnostics, l2_verify_results_to_diagnostics,
    l4_verify_results_to_diagnostics,
)
from .repair import CodeRepairer, RepairResult
from .prompts import (
    format_diagnostic_report, build_repair_prompt, REPAIR_PROMPT_SYSTEM,
    describe_data_schema,
)
from .repair_safety import validate_repair_code, SAFETY_RE_REPAIR_PROMPT
from .specification import run_l4

logger = logging.getLogger(__name__)
from .l2_direction import (
    L2DirectionVerifier,
    L2VerifyResult,
    L2RepairDecision,
    L2_REPAIR_PROMPT,
    should_exit_l2_loop,
    # Backward compatibility aliases
    L4AdversarialVerifier,
    L4VerifyResult,
    L4RepairDecision,
    L4_REPAIR_PROMPT,
    should_exit_l4_loop,
)
from .perturbation import detect_perturbation_mode


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
    Complete ReLoop Pipeline: Generate -> Verify -> Repair loop.

    Four-Layer Architecture:
    - L1: Execution Verification (blocking) + duality check
    - L2: Direction Consistency Analysis (adversarial LLM debate)
    - L3: Constraint Presence Testing (CPT, optional)
    - L4: Specification Compliance Checking (white-box code review, optional)

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
        enable_l4_specification: bool = False,
        use_structured_generation: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            llm_client: LLM client with generate(prompt, system=None) method
            max_repair_iterations: Max repair attempts for ERROR/WARNING issues
            max_regeneration_attempts: Max regeneration attempts for L1 FATAL
            max_l4_rejections: Max rejections per param before L2 downgrades to INFO
            enable_cpt: Enable L3 CPT layer
            enable_l4_adversarial: Enable L2 Direction Consistency Analysis
            enable_l4_specification: Enable L4 Specification Compliance Checking
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
        self.enable_l2_direction = enable_l4_adversarial
        self.enable_l4_specification = enable_l4_specification
        self.verbose = verbose

        # L2 Direction Consistency Verifier (adversarial)
        if self.enable_l2_direction and llm_client:
            self.l2_verifier = L2DirectionVerifier(
                llm_client=llm_client,
                max_rejections=max_l4_rejections
            )
        else:
            self.l2_verifier = None

        # Backward compatibility alias
        self.l4_verifier = self.l2_verifier
        self.enable_l4_adversarial = self.enable_l2_direction

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
        2. Verify with L1 (execution + duality)
        3. If L1 FATAL: regenerate (up to max_regeneration_attempts)
        4. Run L2 Direction Consistency Analysis (adversarial loop)
        5. If ERROR/WARNING: repair (up to max_repair_iterations)
        6. INFO does NOT trigger repair (likely normal)
        7. Return result (always has output if any L1 passes)

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

        # Step 4: Run L2 Direction Consistency Analysis (if enabled)
        l2_time = 0.0
        last_l2_results: List[L2VerifyResult] = []
        if self.enable_l2_direction and self.l2_verifier and report.status != 'FAILED':
            if self.verbose:
                print("[Pipeline] Running L2 Direction Consistency Analysis")

            l2_start = time.time()
            baseline_obj = report.objective or 0.0

            l2_results, l2_exit_reason, l2_code = self._run_l2_adversarial_loop(
                code=code,
                data=data,
                baseline_obj=baseline_obj,
                problem_description=problem_description,
                report=report,
                max_l2_iterations=3
            )
            l2_time = time.time() - l2_start
            last_l2_results = l2_results

            if self.verbose:
                print(f"[Pipeline] L2 exit reason: {l2_exit_reason}")

            # If L2 produced fixed code, update and re-verify
            if l2_code != code and l2_exit_reason in ("accepted_fixed", "max_iterations"):
                code = l2_code
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
                    print(f"[Pipeline] After L2 fix: {report.status}")

        # Step 4.5: Run L4 Specification Compliance Check (if enabled)
        # Collect L1-L3 diagnostics to provide as context to L4
        l4_diagnostics: List[Diagnostic] = []
        if self.enable_l4_specification and report.status != 'FAILED':
            if self.verbose:
                print("[Pipeline] Running L4 Specification Compliance Check")

            # Gather L1-L3 diagnostics for cross-layer context
            pre_l4_diagnostics = self._collect_diagnostics(
                report, l2_results=last_l2_results
            )
            l1_diags = [d for d in pre_l4_diagnostics if d.layer == "L1"]
            l2_diags = [d for d in pre_l4_diagnostics if d.layer == "L2"]
            l3_diags = [d for d in pre_l4_diagnostics if d.layer == "L3"]

            l4_result = run_l4(
                code=code,
                problem_desc=problem_description,
                llm_fn=self._make_l4_llm_fn(),
                z_base=report.objective,
                l1_diagnostics=l1_diags,
                l2_diagnostics=l2_diags,
                l3_diagnostics=l3_diags,
            )
            l4_diagnostics = l4_result['diagnostics']

            if self.verbose:
                s = l4_result['summary']
                print(f"  [L4] Specs: {s['total']}, "
                      f"PASS={s['pass']}, FAIL={s['fail']}, "
                      f"UNCERTAIN={s['uncertain']}")

            if l4_result['summary']['fail'] > 0:
                logger.warning(
                    f"L4 Specification Check: {l4_result['summary']['fail']} "
                    f"violations found out of {l4_result['summary']['total']} specs"
                )

        # Step 5: Handle ERROR/WARNING with repair (INFO does NOT trigger)
        # Uses unified Diagnostic schema + build_repair_prompt()
        repair_iteration = 0
        all_diagnostics = self._collect_diagnostics(report, l2_results=last_l2_results)
        all_diagnostics.extend(l4_diagnostics)

        while (any(d.triggers_repair for d in all_diagnostics)
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

            # Re-collect diagnostics from updated report (L1/L2/L3 only).
            # L4 is NOT re-run (too expensive); if L4 was the only source of
            # triggers_repair=True, the loop will now exit correctly.
            all_diagnostics = self._collect_diagnostics(report, l2_results=last_l2_results)

            if self.verbose:
                print(f"[Pipeline] After repair: {report.status}")

            # Stop if repair didn't help (status AND objective both unchanged)
            if (report.status == prev_report.status
                    and report.objective == prev_report.objective):
                if self.verbose:
                    print("[Pipeline] Repair did not change status or objective, stopping")
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
            all_diagnostics = self._collect_diagnostics(report, l2_results=[])

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
        - critical_errors: ERROR level - Must fix
        - should_fix: WARNING level (L3 cpt_missing) - Should fix
        - for_reference: INFO level (L1 duality, etc.) - Do NOT fix

        Note: L2 issues are handled separately by adversarial mechanism.
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

            # L2 - handled by adversarial mechanism, skip here
            elif r.layer == "L2":
                item["action"] = "L2 issues handled by adversarial mechanism"
                for_reference.append(item)

            # WARNING level - should fix (L3 cpt_missing)
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

    def _collect_diagnostics(
        self,
        report: VerificationReport,
        l2_results: Optional[List[L2VerifyResult]] = None,
    ) -> List[Diagnostic]:
        """
        Collect unified Diagnostic objects from verification report + L2 results.

        Converts all LayerResult and L2VerifyResult into Diagnostic schema.
        """
        # Convert L1, L3 layer results
        diagnostics = layer_results_to_diagnostics(
            report.layer_results,
            baseline_obj=report.objective,
            delta=self.verifier.delta,
        )

        # Convert L2 results (if any)
        if l2_results and self.l2_verifier:
            l2_diags = l2_verify_results_to_diagnostics(
                l2_results,
                rejection_history=self.l2_verifier.rejection_history,
            )
            diagnostics.extend(l2_diags)

        return diagnostics

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
    # L2 Direction Consistency Analysis (Adversarial Loop)
    # =========================================================================

    def _run_l2_adversarial_loop(
        self,
        code: str,
        data: Dict[str, Any],
        baseline_obj: float,
        problem_description: str,
        report: VerificationReport,
        max_l2_iterations: int = 3
    ) -> Tuple[List[L2VerifyResult], str, str]:
        """
        Run L2 Direction Consistency Analysis loop.

        Args:
            code: Current code
            data: Problem data
            baseline_obj: Baseline objective value
            problem_description: Problem description
            report: Current verification report
            max_l2_iterations: Max iterations for L2 loop

        Returns:
            Tuple[l2_results, exit_reason, final_code]

        Exit reasons:
            - "all_pass": All L2 checks passed
            - "all_rejected_others_pass": All L2 rejected, other layers PASS
            - "max_rejections": Max rejections reached, downgraded to INFO
            - "accepted_fixed": Some accepted and code was fixed
            - "max_iterations": Reached max L2 iterations
            - "no_violations": No violations found
            - "disabled": L2 disabled or no LLM client
        """
        if not self.l2_verifier or not self.enable_l2_direction:
            return [], "disabled", code

        # Reset L2 verifier state
        self.l2_verifier.reset()

        # Detect perturbation mode for L2
        mode = detect_perturbation_mode(code, data)

        current_code = code
        l2_results = []
        # After first iteration, constrain future analyses to only the originally
        # flagged parameters (or their rejected subset) to avoid surfacing new issues.
        allowed_params: Optional[List[str]] = None

        for l2_iter in range(max_l2_iterations):
            if self.verbose:
                print(f"  [L2] Adversarial loop iteration {l2_iter + 1}")

            # Step 1: L2 Verify
            l2_results = self.l2_verifier.verify(
                code=current_code,
                data=data,
                baseline_obj=baseline_obj,
                problem_description=problem_description,
                params=allowed_params,
                executor=self.verifier.executor,
                mode=mode
            )

            # Step 2: Check if all PASS (no violations)
            violations = [
                r for r in l2_results
                if r.is_violation and r.confidence >= self.l2_verifier.confidence_threshold
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
                    print("  [L2] No violations found - PASS")
                return l2_results, "all_pass", current_code

            if self.verbose:
                print(f"  [L2] Found {len(violations)} violations")

            # Step 3: Get repair decisions (Accept/Reject)
            decisions, fixed_code = self._get_l2_repair_decisions(
                code=current_code,
                problem_description=problem_description,
                data=data,
                l2_results=l2_results
            )

            if not decisions:
                if self.verbose:
                    print("  [L2] No decisions received, treating as all rejected")
                # Check if other layers pass
                if self._check_other_layers_pass(report):
                    return l2_results, "all_rejected_others_pass", current_code
                continue

            # Step 4: Process decisions
            decision_result = self.l2_verifier.process_repair_decisions(
                decisions=decisions,
                verify_results=l2_results
            )

            accepted = decision_result["accepted"]
            rejected = decision_result["rejected"]
            # Limit any further L2 runs to only the still-rejected params
            allowed_params = [d.param for d in rejected]

            if self.verbose:
                print(f"  [L2] Accepted: {len(accepted)}, Rejected: {len(rejected)}")

            # Step 5: Handle accepted (apply fix)
            if accepted and fixed_code and fixed_code != current_code:
                # Safety guardrail: validate L2 repair code
                fixed_code = self._apply_safety_guardrail(
                    fixed_code, current_code, data, prompt=None
                )
                if fixed_code == current_code:
                    if self.verbose:
                        print("  [L2] Fix rejected by safety guardrail")
                    continue

                # Verify fixed code executes
                try:
                    fix_result = self.verifier.executor.execute(fixed_code, data)
                    if fix_result.get("status") == "OPTIMAL":
                        current_code = fixed_code
                        if self.verbose:
                            print("  [L2] Applied fix, continuing to verify")
                        continue
                except Exception:
                    pass

            # Step 6: Handle all rejected
            if rejected and not accepted:
                if decision_result["should_reverify"]:
                    if self.verbose:
                        print("  [L2] All rejected, re-verifying with context")
                    allowed_params = [d.param for d in rejected if d.param in (allowed_params or [])]
                    continue

                # Check if other layers pass
                if self._check_other_layers_pass(report):
                    if self.verbose:
                        print("  [L2] All rejected + others PASS")
                    return l2_results, "all_rejected_others_pass", current_code

            # Step 7: Check max rejections
            l2_status = self.l2_verifier.get_final_status(l2_results)
            if l2_status == "INFO":
                if self.verbose:
                    print("  [L2] Max rejections reached, downgraded to INFO")
                return l2_results, "max_rejections", current_code

        if self.verbose:
            print("  [L2] Max iterations reached")
        return l2_results, "max_iterations", current_code

    # Backward compatibility alias
    def _run_l4_adversarial_loop(self, code, data, baseline_obj, problem_description,
                                  report, max_l4_iterations=3):
        return self._run_l2_adversarial_loop(
            code, data, baseline_obj, problem_description,
            report, max_l2_iterations=max_l4_iterations
        )

    def _get_l2_repair_decisions(
        self,
        code: str,
        problem_description: str,
        data: Dict[str, Any],
        l2_results: List[L2VerifyResult]
    ) -> Tuple[List[L2RepairDecision], Optional[str]]:
        """
        Call repair LLM to get Accept/Reject decisions for L2 diagnostics.

        Returns:
            Tuple[decisions, fixed_code]
        """
        # Format L2 diagnostics
        diagnostics = self.l2_verifier.format_diagnostics_for_repair(l2_results)

        if diagnostics == "No direction violations detected.":
            return [], None

        # Build repair prompt
        prompt = L2_REPAIR_PROMPT.format(
            problem_description=problem_description,
            code=code,
            l4_diagnostics=diagnostics
        )

        try:
            response = self.llm_client.generate(prompt)
            decisions, fixed_code = self.l2_verifier.parse_repair_response(response)
            return decisions, fixed_code
        except Exception as e:
            if self.verbose:
                print(f"  [L2] Failed to get repair decisions: {e}")
            return [], None

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

        Args:
            repaired_code: Code from repair LLM
            original_code: Code before repair
            data: External data dict
            prompt: Original repair prompt (for re-repair context)

        Returns:
            Safe repaired code, or original_code if safety check fails twice
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
            # Re-send original prompt with safety feedback prepended
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

            # Second violation — give up
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

    def _make_l4_llm_fn(self):
        """Create LLM function adapter for L4 specification checking.

        L4 expects llm_fn(system_prompt, user_prompt) -> str.
        This wraps the project's llm_client.generate(prompt, system=...) interface.
        """
        def llm_fn(system_prompt: str, user_prompt: str) -> str:
            return self.llm_client.generate(user_prompt, system=system_prompt)
        return llm_fn

    def _check_other_layers_pass(self, report: VerificationReport) -> bool:
        """Check if L1, L3 all PASS or INFO (no ERROR/WARNING)."""
        for r in report.layer_results:
            if r.layer == "L2":
                continue  # Skip L2 (handled separately)
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
    enable_l4_specification: bool = False,
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
        enable_cpt: Enable L3 CPT layer
        enable_l4_specification: Enable L4 Specification Compliance Checking
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
        enable_l4_specification=enable_l4_specification,
        use_structured_generation=use_structured_generation,
        verbose=verbose
    )
    return pipeline.run(problem_description, data, initial_code=code)
