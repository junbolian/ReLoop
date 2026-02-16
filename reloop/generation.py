"""
ReLoop Code Generation Module

Generation Approaches:
  1. Chain-of-Thought (CoT): Single API call with step-by-step reasoning (RECOMMENDED)
  2. Single-Stage: Direct problem → code generation
  3. Legacy 3-Stage: Separate API calls (deprecated)

This is a universal architecture that works for all optimization domains.
"""

import json
import logging
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .prompts import (
    CHAIN_OF_THOUGHT_PROMPT, CHAIN_OF_THOUGHT_SYSTEM,
    CHAIN_OF_THOUGHT_WITH_DATA_PROMPT, CHAIN_OF_THOUGHT_WITH_DATA_SYSTEM,
    DATA_EXTRACTION_PROMPT, DATA_EXTRACTION_SYSTEM,
    CODE_GENERATION_PROMPT, CODE_GENERATION_SYSTEM,
    DIRECT_GENERATION_PROMPT, DIRECT_GENERATION_SYSTEM,
    UNDERSTAND_PROMPT, UNDERSTAND_SYSTEM,
    FORMALIZE_PROMPT, FORMALIZE_SYSTEM,
    SYNTHESIZE_PROMPT, SYNTHESIZE_SYSTEM,
    REGENERATE_PROMPT, REGENERATE_SYSTEM,
    describe_data_schema, format_data_instructions
)

logger = logging.getLogger(__name__)


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
        self.extracted_data = None  # Set when extraction succeeds

    def generate(
        self,
        problem_description: str,
        data: Dict[str, Any] = None,
        max_retries: int = 1  # kept for backward compatibility, ignored
    ) -> str:
        """
        Generate Gurobi code for the given problem.

        When extraction succeeds, self.extracted_data is set to the extracted
        dict.  The caller (pipeline) should use self.extracted_data for
        execution when it is not None.

        Args:
            problem_description: Natural language problem description
            data: Problem data dictionary
            max_retries: Maximum retry attempts

        Returns:
            Generated Python code string
        """
        self.extracted_data = None  # Reset
        if self.use_structured_generation:
            return self._generate_chain_of_thought(problem_description, data)
        else:
            return self._generate_direct(problem_description, data)

    def _generate_direct(
        self,
        problem_description: str,
        data: Dict[str, Any] = None
    ) -> str:
        """
        Direct generation: Single API call without chain-of-thought reasoning.
        Generates self-contained code (no data extraction needed).
        """
        prompt = DIRECT_GENERATION_PROMPT.format(
            problem_description=problem_description,
        )

        response = self.llm_client.generate(prompt, system=DIRECT_GENERATION_SYSTEM)
        code = self._extract_code(response)
        if not code or not code.strip():
            raise ValueError("Direct generation produced no code block")
        return code

    def _generate_chain_of_thought(
        self,
        problem_description: str,
        data: Dict[str, Any] = None
    ) -> str:
        """
        Chain-of-Thought generation with extraction + fallback.

        Strategy:
        1. Try extraction: LLM extracts structured data → generate code using data[...]
        2. Fallback: Generate self-contained code (data embedded via json.loads)

        Extraction advantages:
        - Data passed accurately as pre-loaded dict (no LLM copy errors)
        - Perturbation testing works (code reads from external data dict)

        Fallback ensures execution rate is preserved when extraction fails.
        """
        # Try extraction + data-reference code generation
        extracted_data, code = self._try_extraction_generation(
            problem_description, data
        )
        if code is not None:
            self.extracted_data = extracted_data
            return code

        # Fallback: self-contained generation (current behavior)
        logger.info("Extraction failed, falling back to self-contained generation")
        prompt = CHAIN_OF_THOUGHT_PROMPT.format(
            problem_description=problem_description,
        )

        response = self.llm_client.generate(prompt, system=CHAIN_OF_THOUGHT_SYSTEM)
        code = self._extract_code(response)
        if not code or not code.strip():
            raise ValueError("Chain-of-Thought generation produced no code block")
        return code

    def _try_extraction_generation(
        self,
        problem_description: str,
        data: Dict[str, Any] = None
    ) -> tuple:
        """
        Try extraction-based code generation.

        Returns:
            (extracted_data, code) on success
            (None, None) on failure (caller should fallback)
        """
        try:
            # Step 1: Extract data from problem description
            extracted_data = self._extract_data(problem_description)
            if extracted_data is None:
                return None, None

            # Step 2: Generate code that uses data[...] (not self-contained)
            data_structure = describe_data_schema(extracted_data)
            prompt = CHAIN_OF_THOUGHT_WITH_DATA_PROMPT.format(
                problem_description=problem_description,
                data_structure=data_structure,
            )

            response = self.llm_client.generate(
                prompt, system=CHAIN_OF_THOUGHT_WITH_DATA_SYSTEM
            )
            code = self._extract_code(response)
            if not code or not code.strip():
                return None, None

            # Verify the code actually uses data[] and doesn't redefine data
            if 'json.loads' in code:
                logger.info("Extraction-mode code still uses json.loads, discarding")
                return None, None

            logger.info("Extraction + data-reference generation succeeded")
            return extracted_data, code

        except Exception as e:
            logger.info(f"Extraction generation failed: {e}")
            return None, None

    def _extract_data(self, problem_description: str) -> Optional[Dict]:
        """
        LLM-based data extraction from problem description.

        Returns parsed dict on success, None on failure.
        """
        prompt = DATA_EXTRACTION_PROMPT.format(
            problem_description=problem_description,
        )

        try:
            response = self.llm_client.generate(prompt, system=DATA_EXTRACTION_SYSTEM)

            # Extract JSON block
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Try raw JSON
                json_str = response.strip()

            data = json.loads(json_str)
            if not isinstance(data, dict):
                return None

            logger.info(f"Data extraction succeeded: {len(data)} top-level keys")
            return data

        except (json.JSONDecodeError, Exception) as e:
            logger.info(f"Data extraction failed: {e}")
            return None

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
        Legacy placeholder: reuse direct generation.
        """
        return self._generate_direct(problem_description, data)

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
        data: Dict[str, Any] = None
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
            data: Problem data dictionary (empty dict for self-contained code)

        Returns:
            New generated code
        """
        data_structure = self._describe_data_schema(data) if data else ""
        data_instructions = format_data_instructions(data_structure)

        prompt = REGENERATE_PROMPT.format(
            problem_description=problem_description,
            failed_code=failed_code,
            error_message=error_message,
            data_instructions=data_instructions
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
