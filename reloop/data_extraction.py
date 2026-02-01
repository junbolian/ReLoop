"""
ReLoop Data Extraction Module

Extract structured parameter data from natural language problem descriptions.
"""

import json
import re
from typing import Dict, Any, Optional


DATA_EXTRACTION_PROMPT = '''Extract ALL numerical parameters from this optimization problem.

## Problem
{problem}

## Task
Identify every numerical value mentioned in the problem and give it a meaningful parameter name.

## Output Format
Return ONLY a JSON object with parameter names as keys and values as numbers or lists:
```json
{{
  "demand": [100, 150, 200],
  "capacity": 500,
  "cost_per_unit": 10,
  "holding_cost": 2.5
}}
```

## Rules
1. Use snake_case for parameter names
2. Use descriptive names (e.g., "labor_hours" not "h")
3. Group related values into lists when appropriate
4. Include ALL numbers mentioned in the problem
5. If a parameter is a matrix/table, use nested structure

Return ONLY the JSON, no explanation.
'''


class DataExtractor:
    """Extract structured parameters from problem descriptions."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract(self, problem_description: str) -> Dict[str, Any]:
        """
        Extract parameter data from problem description.

        Args:
            problem_description: Natural language problem description (en_question)

        Returns:
            Parameter dictionary, e.g., {"demand": [100, 150], "capacity": 500}
        """
        prompt = DATA_EXTRACTION_PROMPT.format(problem=problem_description)

        try:
            response = self.llm_client.generate(prompt)
            data = self._parse_json(response)

            if data and self._validate_data(data):
                return data
        except Exception as e:
            print(f"Data extraction error: {e}")

        # Fallback: extract numbers with regex
        return self._extract_numbers_regex(problem_description)

    def _parse_json(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response."""
        # Try to find JSON code block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except Exception:
                pass

        # Try to find {} block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass

        return None

    def _validate_data(self, data: Dict) -> bool:
        """Validate extracted data is valid."""
        if not isinstance(data, dict):
            return False
        if len(data) == 0:
            return False

        # Must have at least one numeric parameter
        has_numeric = False
        for value in data.values():
            if isinstance(value, (int, float)):
                has_numeric = True
            elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                has_numeric = True

        return has_numeric

    def _extract_numbers_regex(self, text: str) -> Dict[str, Any]:
        """Extract numbers with regex (fallback method)."""
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)

        data = {}
        for i, num in enumerate(numbers[:20]):  # Max 20 parameters
            try:
                value = float(num)
                if value == int(value):
                    value = int(value)
                data[f"param_{i+1}"] = value
            except Exception:
                pass

        return data


def extract_data_from_question(problem: str, llm_client) -> Dict[str, Any]:
    """Convenience function to extract data from problem description."""
    extractor = DataExtractor(llm_client)
    return extractor.extract(problem)
