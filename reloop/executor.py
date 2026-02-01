"""
ReLoop Code Executor

Functions:
- Syntax checking
- Subprocess isolated execution
- Output parsing
"""

import subprocess
import sys
import json
import base64
import re
from typing import Dict, Any, Tuple, Optional


class CodeExecutor:
    """Execute optimization code in isolated subprocess."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check Python syntax.

        Returns: (passed, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def execute(self, code: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code in subprocess.

        Returns: {
            'exit_code': int,
            'status': str | None,      # 'OPTIMAL', 'INFEASIBLE', etc.
            'objective': float | None,
            'dual_objective': float | None,
            'solution': dict | None,
            'stdout': str,
            'stderr': str,
            'error': str | None
        }
        """
        # Encode data as base64 for passing
        data_b64 = base64.b64encode(json.dumps(data).encode()).decode()

        wrapper = f'''
import sys, json, base64
data = json.loads(base64.b64decode("{data_b64}").decode())
{code}
'''

        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return self._parse_output(result)
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "status": "TIMEOUT",
                "objective": None,
                "error": "Solver timeout"
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "status": "ERROR",
                "objective": None,
                "error": str(e)
            }

    def _parse_output(self, result) -> Dict[str, Any]:
        """Parse subprocess output."""
        output = {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "status": None,
            "objective": None,
            "dual_objective": None,
            "solution": None,
            "error": result.stderr if result.returncode != 0 else None
        }

        # Parse status: X
        status_match = re.search(r"status:\s*(\d+)", result.stdout)
        if status_match:
            code = int(status_match.group(1))
            output["status"] = {
                2: "OPTIMAL",
                3: "INFEASIBLE",
                5: "UNBOUNDED",
                9: "TIME_LIMIT"
            }.get(code, f"CODE_{code}")

        # Parse objective: X
        obj_match = re.search(
            r"objective:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            result.stdout
        )
        if obj_match:
            output["objective"] = float(obj_match.group(1))

        # Parse dual_objective: X
        dual_match = re.search(
            r"dual_objective:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            result.stdout
        )
        if dual_match:
            output["dual_objective"] = float(dual_match.group(1))

        # Parse solution: {...}
        sol_match = re.search(r"solution:\s*(\{[^}]+\})", result.stdout)
        if sol_match:
            try:
                output["solution"] = json.loads(sol_match.group(1))
            except:
                pass

        return output
