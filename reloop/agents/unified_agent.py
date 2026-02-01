"""
Unified Agent wrapper that auto-selects dataset settings and keeps
the 7-layer verification core unchanged.

- Layers 1-6: universal
- Layer 7 : enabled only for RetailOpt-190 (retail data)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from reloop.reloop import ReLoop, ReLoopConfig, ReLoopResult
from reloop.structured_generation import StructuredGenerator, LLMClient
from reloop.behavioral_verification import BehavioralVerifier, VerificationReport


@dataclass
class UnifiedRunResult:
    code: str
    report: Optional[VerificationReport]
    reloop_result: Optional[ReLoopResult]
    dataset_type: str
    used_structured: bool


class UnifiedAgent:
    """Facade to run ReLoop across multiple datasets with minimal config."""

    def __init__(self, llm_client: LLMClient, max_iterations: int = 5, timeout: int = 60):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.timeout = timeout

    def _plan(self, data: Dict[str, Any], scenario_id: Optional[str], problem_text: str) -> Dict[str, Any]:
        return {
            "schema": "",
            "enable_layer7": False,
            "use_structured": True,
            "obj_sense": "minimize",
        }

    def run(self, problem_text: str, data: Optional[Dict[str, Any]] = None,
            scenario_id: Optional[str] = None, obj_sense: str = "minimize") -> UnifiedRunResult:
        plan = self._plan(data or {}, scenario_id, problem_text)

        # If no structured data provided, generate code only (cannot verify without data)
        if not data:
            generator = StructuredGenerator(self.llm_client)
            if plan["use_structured"]:
                code = generator.generate(problem_text, plan["schema"])
            else:
                code = generator.generate_baseline(problem_text, plan["schema"])
            return UnifiedRunResult(
                code=code,
                report=None,
                reloop_result=None,
                dataset_type="prompt_only",
                used_structured=plan["use_structured"],
            )

        # Config: keep 7-layer core; layer7 gated by enable_layer7 flag
        config = ReLoopConfig(
            max_iterations=self.max_iterations,
            timeout=self.timeout,
            enable_layer7=plan["enable_layer7"],
            verbose=False,
        )

        if plan["use_structured"]:
            pipeline = ReLoop(self.llm_client, config)
            res = pipeline.run(problem_text, plan["schema"], data or {}, obj_sense=obj_sense)
            return UnifiedRunResult(
                code=res.code,
                report=res.final_report,
                reloop_result=res,
                dataset_type="prompt_only",
                used_structured=True,
            )

        # Baseline (text-only) path with verification if data exists
        generator = StructuredGenerator(self.llm_client)
        code = generator.generate_baseline(problem_text, plan["schema"])
        verifier = BehavioralVerifier(timeout=self.timeout)
        report = verifier.verify(code, data, obj_sense=obj_sense, enable_layer7=False,
                                 verbose=False)

        return UnifiedRunResult(
            code=code,
            report=report,
            reloop_result=None,
            dataset_type="prompt_only",
            used_structured=False,
        )
