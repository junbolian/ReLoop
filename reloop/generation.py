"""ReLoop Code Generation Module"""

import re
from typing import Dict, Any
from .prompts import CODE_GENERATION_PROMPT, CODE_GENERATION_SYSTEM


class CodeGenerator:
    """Generate optimization code from problem description."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate(
        self,
        problem_description: str,
        data: Dict[str, Any],
        max_retries: int = 3
    ) -> str:
        """Generate Gurobi code for the given problem."""
        data_structure = self._describe_data(data)
        prompt = CODE_GENERATION_PROMPT.format(
            problem_description=problem_description,
            data_structure=data_structure
        )

        for _ in range(max_retries):
            try:
                response = self.llm_client.generate(prompt, system=CODE_GENERATION_SYSTEM)
                code = self._extract_code(response)
                if self._validate_code(code):
                    return code
            except Exception:
                continue

        raise ValueError("Failed to generate valid code after retries")

    def _describe_data(self, data: Dict, prefix: str = "", depth: int = 0) -> str:
        """Create a structured description of the data dictionary."""
        if depth > 3:
            return ""

        lines = []
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, (int, float)):
                lines.append(f"- {path}: {type(value).__name__} = {value}")
            elif isinstance(value, list):
                sample = value[:3] if len(value) > 3 else value
                lines.append(f"- {path}: list[{len(value)}], sample={sample}")
            elif isinstance(value, dict):
                keys_sample = list(value.keys())[:5]
                lines.append(f"- {path}: dict, keys={keys_sample}")

        return "\n".join(lines)

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code block
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
