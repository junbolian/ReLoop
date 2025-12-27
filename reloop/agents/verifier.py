from typing import Dict, Any, Tuple

from .script_executor import execute_script
from .agent_types import ExecutionResult


def run_and_verify(
    script_text: str,
    data: Dict[str, Any],
    timeout_s: float = 60.0,
    request_iis_on_infeasible: bool = True,
) -> ExecutionResult:
    res = execute_script(
        script_text, data, timeout_s=timeout_s, request_iis=request_iis_on_infeasible
    )
    return res
