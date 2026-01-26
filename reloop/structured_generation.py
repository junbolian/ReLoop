"""
Module 1: Structured Generation

3-step generation process:
  Step 1: Problem Understanding → {objective, decisions, constraints, key_relationships}
  Step 2: Mathematical Specification → {sets, params, vars, constraints, objective}
  Step 3: Code Generation → executable GurobiPy code

Key Design: LLM only sees schema, NOT actual data values.

Supports: RetailOpt-190, MAMO, NL4OPT datasets
"""

import json
from typing import List, Optional
from abc import ABC, abstractmethod

from .prompts import (
    STEP1_PROMPT, STEP2_PROMPT, STEP3_PROMPT,
    BASELINE_PROMPT, MAMO_BASELINE_PROMPT, NL4OPT_BASELINE_PROMPT,
    RETAIL_SCHEMA_TEMPLATE, MAMO_SCHEMA_TEMPLATE, NL4OPT_SCHEMA_TEMPLATE,
    PromptGenerator
)


class LLMClient(ABC):
    """Abstract interface for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0) -> str:
        pass


class OpenAIClient(LLMClient):
    """OpenAI-compatible client"""

    def __init__(self, model: str = "gpt-4o", base_url: str = None, api_key: str = None):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic API client for Claude models"""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", base_url: str = None, api_key: str = None):
        import anthropic
        # Use custom base_url if provided (strip /v1 suffix if present)
        if base_url:
            base_url = base_url.rstrip('/')
            if base_url.endswith('/v1'):
                base_url = base_url[:-3]
        self.client = anthropic.Anthropic(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.content[0].text


class StructuredGenerator:
    """
    Module 1: Structured Generation (3-step process)

    Supports multiple dataset types:
    - retail: RetailOpt-190 inventory optimization
    - mamo: MAMO mathematical optimization
    - nl4opt: NL4OPT competition format
    """

    def __init__(self, llm_client: LLMClient, dataset_type: str = "retail"):
        self.llm = llm_client
        self.dataset_type = dataset_type

    def generate(self, problem: str, schema: str, history: List[str] = None) -> str:
        """Run complete 3-step generation."""
        understanding = self._step1(problem, schema)
        math_spec = self._step2(understanding, schema)
        # Pass original problem to Step 3 to preserve key equations
        code = self._step3(math_spec, schema, history, problem)
        return code

    def generate_baseline(self, problem: str, schema: str) -> str:
        """Generate code directly without structured steps (for comparison)."""
        prompt = PromptGenerator.baseline(problem, schema, self.dataset_type)
        return self._extract_code(self.llm.generate(prompt))

    def _step1(self, problem: str, schema: str) -> str:
        """
        Step 1: Problem Understanding

        Output: JSON with objective, decisions, constraints, key_relationships
        """
        prompt = PromptGenerator.step1(problem, schema)
        return self._extract_json(self.llm.generate(prompt))

    def _step2(self, understanding: str, schema: str) -> str:
        """
        Step 2: Mathematical Specification

        Output: JSON with sets, parameters, variables, constraints, objective
        """
        prompt = PromptGenerator.step2(understanding, schema)
        return self._extract_json(self.llm.generate(prompt))

    def _step3(self, math_spec: str, data_access: str, history: List[str] = None,
                original_problem: str = None) -> str:
        """
        Step 3: Code Generation

        Output: Executable GurobiPy code
        """
        # Build history section for repair hints
        history_section = ""
        if history:
            history_section = "\n\n## Previous Failures (FIX THESE)\n" + "\n".join(f"- {h}" for h in history)

        # Include original problem for key equations (e.g., shelf_life FIFO)
        problem_context = ""
        if original_problem:
            problem_context = "\n\n## Original Problem Context (Reference for Key Equations)\n" + original_problem

        prompt = PromptGenerator.step3(math_spec, data_access + problem_context + history_section)
        return self._extract_code(self.llm.generate(prompt))

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract from markdown code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Try to find code starting with import
        lines = response.strip().split("\n")
        code_lines, in_code = [], False
        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                in_code = True
            if in_code:
                code_lines.append(line)
        return "\n".join(code_lines) if code_lines else response.strip()

    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response."""
        # Try to extract from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        # Try to find JSON object
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                json.loads(json_str)  # Validate
                return json_str
        except json.JSONDecodeError:
            pass
        return response.strip()


# ==============================================================================
# SCHEMA TEMPLATES
# ==============================================================================

RETAIL_SCHEMA = RETAIL_SCHEMA_TEMPLATE

MAMO_SCHEMA = MAMO_SCHEMA_TEMPLATE

NL4OPT_SCHEMA = NL4OPT_SCHEMA_TEMPLATE


def get_schema_for_dataset(dataset_type: str = "retail") -> str:
    """Get appropriate schema template for dataset type."""
    if dataset_type == "retail":
        return RETAIL_SCHEMA
    elif dataset_type == "mamo":
        return MAMO_SCHEMA
    elif dataset_type == "nl4opt":
        return NL4OPT_SCHEMA
    return "Standard LP/MILP data structure"


def detect_dataset_type(data: dict) -> str:
    """Auto-detect dataset type from data structure."""
    data_str = str(data).lower()

    # RetailOpt-190 signatures
    retail_signatures = ["shelf_life", "demand_curve", "cold_capacity",
                         "demand_share", "waste", "lost_sales"]
    retail_score = sum(1 for sig in retail_signatures if sig in data_str)
    if retail_score >= 3:
        return "retail"

    # MAMO signatures
    if "problem_type" in data or "problem_description" in data:
        return "mamo"

    # NL4OPT signatures
    if "obj_declaration" in data or "const_declarations" in data or "problem_text" in data:
        return "nl4opt"

    return "generic"
