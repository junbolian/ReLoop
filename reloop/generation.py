"""
ReLoop Code Generation Module

Generation Approaches:
  1. Chain-of-Thought (CoT): Single API call with step-by-step reasoning (RECOMMENDED)
  2. Single-Stage: Direct problem → code generation
  3. Legacy 3-Stage: Separate API calls (deprecated)

This is a universal architecture that works for all optimization domains.
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .prompts import (
    CHAIN_OF_THOUGHT_PROMPT, CHAIN_OF_THOUGHT_SYSTEM,
    UNDERSTAND_PROMPT, UNDERSTAND_SYSTEM,
    FORMALIZE_PROMPT, FORMALIZE_SYSTEM,
    SYNTHESIZE_PROMPT, SYNTHESIZE_SYSTEM,
    REGENERATE_PROMPT, REGENERATE_SYSTEM,
    describe_data_schema
)


@dataclass
class GenerationResult:
    """Result of the three-stage generation process."""
    code: str
    understanding: Optional[str] = None  # Stage 1 output (U)
    mathematical_model: Optional[str] = None  # Stage 2 output (M)
    stages_completed: int = 0
    stage_logs: Optional[Dict[str, Any]] = None  # Log of each stage's input/output


class CodeGenerator:
    """
    Three-stage optimization code generator.

    Pipeline: x → Understand → U → Formalize → M → Synthesize → Ck

    Features:
    - Multi-turn dialogue for complex problems
    - Schema-based data description (structure only, not values)
    - Universal architecture for all optimization domains
    - Fallback to single-stage generation if stages fail
    """

    def __init__(self, llm_client, use_structured_generation: bool = True):
        """
        Args:
            llm_client: LLM client with generate(prompt, system=None) method
            use_structured_generation: If True, use 3-stage pipeline; if False, use single-stage
        """
        self.llm_client = llm_client
        self.use_structured_generation = use_structured_generation

    def generate(
        self,
        problem_description: str,
        data: Dict[str, Any],
        max_retries: int = 1  # kept for backward compatibility, ignored
    ) -> str:
        """
        Generate Gurobi code for the given problem.

        Args:
            problem_description: Natural language problem description
            data: Problem data dictionary
            max_retries: Maximum retry attempts

        Returns:
            Generated Python code string
        """
        # Always do a single CoT generation attempt. We intentionally ignore
        # max_retries here to avoid repeated CoT prompts.
        return self._generate_chain_of_thought(problem_description, data)

    def _generate_chain_of_thought(
        self,
        problem_description: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Chain-of-Thought generation: Single API call with step-by-step reasoning.

        This is the RECOMMENDED approach. The LLM reasons through:
        1. Understanding the problem
        2. Formulating the mathematical model
        3. Generating the code

        All in ONE conversation turn, preserving context.
        """
        data_structure = self._describe_data_schema(data)
        prompt = CHAIN_OF_THOUGHT_PROMPT.format(
            problem_description=problem_description,
            data_structure=data_structure
        )

        response = self.llm_client.generate(prompt, system=CHAIN_OF_THOUGHT_SYSTEM)
        code = self._extract_code(response)
        # Even if validation would fail, return the extracted code for L1 to attempt/repair.
        if not code or not code.strip():
            raise ValueError("Chain-of-Thought generation produced no code block")
        return code

    def generate_structured(
        self,
        problem_description: str,
        data: Dict[str, Any],
        max_retries: int = 3,
        verbose: bool = False
    ) -> GenerationResult:
        """
        Execute the full three-stage generation pipeline.

        Stage 1: Understand - Analyze problem → structured understanding U
        Stage 2: Formalize  - U → mathematical specification M = (I, P, V, C, f)
        Stage 3: Synthesize - M → executable Gurobi code

        Args:
            problem_description: Natural language problem description
            data: Problem data dictionary
            max_retries: Maximum retry attempts per stage
            verbose: If True, log each stage's input/output

        Returns:
            GenerationResult with code and intermediate outputs
        """
        data_structure = self._describe_data_schema(data)
        stage_logs = {"stages": []}

        for attempt in range(max_retries):
            try:
                # Stage 1: Understand
                if verbose:
                    print("\n[Stage 1: Understand]")
                understanding = self._stage1_understand(problem_description)
                if not understanding:
                    continue
                stage_logs["stages"].append({
                    "stage": 1,
                    "name": "Understand",
                    "input": {"problem_description": problem_description[:500] + "..."},
                    "output": understanding[:1000] + "..." if len(understanding) > 1000 else understanding
                })
                if verbose:
                    print(f"  Output (U): {understanding[:500]}...")

                # Stage 2: Formalize
                if verbose:
                    print("\n[Stage 2: Formalize]")
                mathematical_model = self._stage2_formalize(
                    problem_description, understanding, data_structure
                )
                if not mathematical_model:
                    continue
                stage_logs["stages"].append({
                    "stage": 2,
                    "name": "Formalize",
                    "input": {
                        "problem": problem_description[:200] + "...",
                        "understanding": understanding[:200] + "...",
                        "data_structure": data_structure
                    },
                    "output": mathematical_model[:1000] + "..." if len(mathematical_model) > 1000 else mathematical_model
                })
                if verbose:
                    print(f"  Output (M): {mathematical_model[:500]}...")

                # Stage 3: Synthesize
                if verbose:
                    print("\n[Stage 3: Synthesize]")
                code = self._stage3_synthesize(
                    problem_description, mathematical_model, data_structure
                )
                stage_logs["stages"].append({
                    "stage": 3,
                    "name": "Synthesize",
                    "input": {
                        "problem": problem_description[:200] + "...",
                        "mathematical_model": mathematical_model[:200] + "...",
                        "data_structure": data_structure
                    },
                    "output": code[:500] + "..." if len(code) > 500 else code
                })
                if verbose:
                    print(f"  Output (Code): {code[:300]}...")

                if self._validate_code(code):
                    return GenerationResult(
                        code=code,
                        understanding=understanding,
                        mathematical_model=mathematical_model,
                        stages_completed=3,
                        stage_logs=stage_logs
                    )

            except Exception:
                continue

        raise ValueError("Chain-of-Thought generation failed")

    def _generate_single_stage(
        self,
        problem_description: str,
        data: Dict[str, Any],
        max_retries: int = 1
    ) -> str:
        """
        Legacy placeholder: reuse single-attempt CoT generation.
        """
        return self._generate_chain_of_thought(problem_description, data)

    def _stage1_understand(self, problem_description: str) -> Optional[str]:
        """
        Stage 1: Understand

        Analyze the natural language description to produce structured understanding U.
        Identifies: objective, decisions, constraints, parameters.
        """
        prompt = UNDERSTAND_PROMPT.format(problem_description=problem_description)

        try:
            response = self.llm_client.generate(prompt, system=UNDERSTAND_SYSTEM)
            # Extract YAML content
            yaml_match = re.search(r'```yaml\s*([\s\S]*?)\s*```', response)
            if yaml_match:
                return yaml_match.group(1).strip()
            # Return raw response if no YAML block
            return response.strip()
        except Exception:
            return None

    def _stage2_formalize(
        self,
        problem_description: str,
        understanding: str,
        data_structure: str
    ) -> Optional[str]:
        """
        Stage 2: Formalize

        Transform understanding U into mathematical specification M = (I, P, V, C, f):
        - I: Index sets
        - P: Parameters
        - V: Decision variables
        - C: Constraints
        - f: Objective function
        """
        prompt = FORMALIZE_PROMPT.format(
            problem_description=problem_description,
            understanding=understanding,
            data_structure=data_structure
        )

        try:
            response = self.llm_client.generate(prompt, system=FORMALIZE_SYSTEM)
            # Extract code block if present
            code_match = re.search(r'```\s*([\s\S]*?)\s*```', response)
            if code_match:
                return code_match.group(1).strip()
            return response.strip()
        except Exception:
            return None

    def _stage3_synthesize(
        self,
        problem_description: str,
        mathematical_model: str,
        data_structure: str
    ) -> str:
        """
        Stage 3: Synthesize

        Generate executable Gurobi code from mathematical specification M.
        """
        prompt = SYNTHESIZE_PROMPT.format(
            problem_description=problem_description,
            mathematical_model=mathematical_model,
            data_structure=data_structure
        )

        response = self.llm_client.generate(prompt, system=SYNTHESIZE_SYSTEM)
        return self._extract_code(response)

    def regenerate(
        self,
        problem_description: str,
        failed_code: str,
        error_message: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Regenerate code after L1 FATAL error.

        This is called when initial code fails to execute (syntax error,
        runtime error, or solver failure). It provides error context to
        help the LLM generate corrected code.

        Args:
            problem_description: Original problem description
            failed_code: The code that failed
            error_message: Error message from execution
            data: Problem data dictionary

        Returns:
            New generated code
        """
        data_structure = self._describe_data_schema(data)

        prompt = REGENERATE_PROMPT.format(
            problem_description=problem_description,
            failed_code=failed_code,
            error_message=error_message,
            data_structure=data_structure
        )

        response = self.llm_client.generate(prompt, system=REGENERATE_SYSTEM)
        return self._extract_code(response)

    def _describe_data_schema(self, data: Dict, prefix: str = "", depth: int = 0) -> str:
        """
        Create a schema description of the data dictionary.
        Delegates to the common describe_data_schema function in prompts.py.
        """
        return describe_data_schema(data, prefix, depth)

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find python code block
        match = re.search(r'```python\s*([\s\S]*?)\s*```', response)
        if match:
            return match.group(1).strip()

        # Try generic code block
        match = re.search(r'```\s*([\s\S]*?)\s*```', response)
        if match:
            return match.group(1).strip()

        # Return as-is
        return response.strip()

    def _validate_code(self, code: str) -> bool:
        """Validate that code contains required Gurobi elements."""
        required_patterns = [
            r'import\s+gurobipy',
            r'\.addVar|\.addVars',
            r'\.setObjective',
            r'\.optimize\s*\(\s*\)'
        ]
        return all(re.search(p, code) for p in required_patterns)
