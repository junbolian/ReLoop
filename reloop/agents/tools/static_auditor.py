from __future__ import annotations

import re
from typing import List

from ..schemas import StaticAuditReport


FORBIDDEN_PATTERNS = [
    r"\bopen\(",
    r"json\.load",
    r"Path\(.*read_text",
    r"\.read_text\(",
]

PREFIXES = [
    "demand_route",
    "sales_conservation",
    "availability",
    "expire_clear",
    "leadtime",
    "returns",
    "init",
    "fresh_inflow",
    "aging",
    "storage_cap",
    "prod_cap",
    "labor_cap",
    "moq_lb",
    "moq_ub",
    "pack",
    "budget",
    "wastecap",
]

VAR_NAMES = ["I", "y", "W", "Q", "L", "d", "S", "X", "z", "n"]


def _check_forbidden(code: str, failures: List[str]) -> None:
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            failures.append(f"Forbidden I/O detected: {pattern}")


def _check_solver_params(code: str, failures: List[str]) -> None:
    if "OutputFlag" not in code or re.search(r"OutputFlag\s*=\s*0", code) is None:
        failures.append("Missing OutputFlag=0 setting.")
    if "Threads" not in code or re.search(r"Threads\s*=\s*1", code) is None:
        failures.append("Missing Threads=1 setting.")
    if "Seed" not in code or re.search(r"Seed\s*=\s*0", code) is None:
        failures.append("Missing Seed=0 setting.")


def _check_print_contract(code: str, failures: List[str]) -> None:
    status_pattern = re.compile(r"status\s*[:=]", re.IGNORECASE)
    if not status_pattern.search(code):
        failures.append("Script must print solver status.")
    if "GRB.OPTIMAL" not in code and "GRB.optimal" not in code:
        failures.append("Script must branch on GRB.OPTIMAL when printing objective.")
    if "ObjBound" not in code and "MIPGap" not in code:
        failures.append("Script must expose ObjBound or MIPGap when not optimal.")


def _check_imports(code: str, failures: List[str], warnings: List[str]) -> None:
    if "import gurobipy as gp" not in code:
        failures.append("Missing required import: import gurobipy as gp")
    if "from gurobipy import GRB" not in code:
        warnings.append("Missing 'from gurobipy import GRB' import.")


def _check_naming_contract(code: str, warnings: List[str]) -> None:
    if not any(f"{name} =" in code for name in VAR_NAMES):
        warnings.append("No variable dictionaries from the naming contract found.")
    if not any(prefix in code for prefix in PREFIXES):
        warnings.append("No constraint name prefixes detected; ensure name= uses the contract prefixes.")


def audit_script(code: str) -> StaticAuditReport:
    failures: List[str] = []
    warnings: List[str] = []
    _check_forbidden(code, failures)
    _check_imports(code, failures, warnings)
    _check_solver_params(code, failures)
    _check_print_contract(code, failures)
    _check_naming_contract(code, warnings)
    passed = not failures
    return StaticAuditReport(passed=passed, failures=failures, warnings=warnings)
