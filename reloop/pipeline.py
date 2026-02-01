"""ReLoop Main Pipeline"""

import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .generation import CodeGenerator
from .verification import ReLoopVerifier, VerificationReport
from .repair import CodeRepairer


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


class ReLoopPipeline:
    """
    Complete ReLoop Pipeline: Generate -> Verify -> Repair loop.

    Usage:
        pipeline = ReLoopPipeline(llm_client)
        result = pipeline.run(problem_description, data)
    """

    def __init__(
        self,
        llm_client,
        max_repair_iterations: int = 3,
        enable_cpt: bool = True,
        verbose: bool = False
    ):
        self.generator = CodeGenerator(llm_client)
        self.verifier = ReLoopVerifier(llm_client=llm_client)
        self.repairer = CodeRepairer(llm_client)
        self.max_repair_iterations = max_repair_iterations
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

        # Step 1: Generate code (or use provided)
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

        # Step 3: Repair loop
        iteration = 0
        while self._needs_repair(report) and iteration < self.max_repair_iterations:
            iteration += 1
            if self.verbose:
                print(f"[Pipeline] Repair iteration {iteration}")

            # Repair
            repair_start = time.time()
            repaired_code = self.repairer.repair(
                code, data, report, problem_description
            )
            repair_time += time.time() - repair_start

            # Check if repair produced different code
            if repaired_code == code:
                if self.verbose:
                    print("[Pipeline] Repair produced no changes, stopping")
                break

            code = repaired_code

            # Re-verify
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

        # Determine success and improvement
        success = report.status in ['VERIFIED', 'WARNINGS']
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
            improved=improved
        )

    def _needs_repair(self, report: VerificationReport) -> bool:
        """Check if report indicates repair is needed."""
        return report.status in ['WARNINGS', 'ERRORS']

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
    enable_cpt: bool = True,
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
        enable_cpt: Enable L5 CPT layer
        verbose: Print progress messages

    Returns:
        PipelineResult with final code, report, and metrics
    """
    pipeline = ReLoopPipeline(
        llm_client,
        max_repair_iterations=max_iterations,
        enable_cpt=enable_cpt,
        verbose=verbose
    )
    return pipeline.run(problem_description, data, initial_code=code)
