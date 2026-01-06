from __future__ import annotations

import re
from typing import List

from ..schemas import StaticAuditReport


# ==============================================================================
# STATIC AUDITOR (RELAXED VERSION)
# 
# Changes from original:
#   - Solver params (OutputFlag, Threads, Seed) -> WARNING instead of FAILURE
#   - Print contract (status, GRB.OPTIMAL) -> WARNING instead of FAILURE
#   - Only FORBIDDEN I/O patterns cause actual failures
#
# This allows functionally correct code to proceed to probe/run even if it
# doesn't follow all formatting conventions.
# ==============================================================================


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
    """Check for forbidden I/O operations - these are HARD failures."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            failures.append(f"Forbidden I/O detected: {pattern}")


def _check_solver_params(code: str, warnings: List[str]) -> None:
    """Check solver params - now warnings only, not failures."""
    if "OutputFlag" not in code or re.search(r"OutputFlag\s*[,=]\s*0", code) is None:
        warnings.append("Missing OutputFlag=0 setting (recommended).")
    if "Threads" not in code or re.search(r"Threads\s*[,=]\s*1", code) is None:
        warnings.append("Missing Threads=1 setting (recommended).")
    if "Seed" not in code or re.search(r"Seed\s*[,=]\s*0", code) is None:
        warnings.append("Missing Seed=0 setting (recommended).")


def _check_print_contract(code: str, warnings: List[str]) -> None:
    """Check print contract - now warnings only, not failures."""
    status_pattern = re.compile(r"status\s*[:=]", re.IGNORECASE)
    if not status_pattern.search(code):
        warnings.append("Script should print solver status (recommended).")
    if "GRB.OPTIMAL" not in code and "GRB.optimal" not in code and "m.status" not in code:
        warnings.append("Script should check solver status before printing objective (recommended).")


def _check_imports(code: str, failures: List[str], warnings: List[str]) -> None:
    """Check required imports."""
    if "gurobipy" not in code:
        failures.append("Missing gurobipy import")
    if "from gurobipy import GRB" not in code:
        warnings.append("Missing 'from gurobipy import GRB' import (recommended).")


def _check_naming_contract(code: str, warnings: List[str]) -> None:
    """Check naming conventions - warnings only."""
    if not any(f"{name} =" in code or f"{name}[" in code for name in VAR_NAMES):
        warnings.append("No variable dictionaries from the naming contract found.")
    if not any(prefix in code for prefix in PREFIXES):
        warnings.append("No constraint name prefixes detected.")


def audit_script(code: str) -> StaticAuditReport:
    """
    Audit a generated script for compliance with code standards.
    
    HARD FAILURES (block execution):
    - Forbidden I/O operations (open, json.load, etc.)
    - Missing gurobipy import
    
    SOFT WARNINGS (allow execution):
    - Missing solver params (OutputFlag, Threads, Seed)
    - Missing print contract
    - Missing naming conventions
    
    This relaxed version allows functionally correct code to proceed
    to semantic probes and execution even if formatting is imperfect.
    """
    failures: List[str] = []
    warnings: List[str] = []
    
    # Hard failures - these block execution
    _check_forbidden(code, failures)
    _check_imports(code, failures, warnings)
    
    # Soft warnings - these don't block execution
    _check_solver_params(code, warnings)
    _check_print_contract(code, warnings)
    _check_naming_contract(code, warnings)
    
    passed = not failures
    
    # Combine failures and warnings for issues list (for logging)
    all_issues = failures + [f"[WARNING] {w}" for w in warnings]
    
    return StaticAuditReport(passed=passed, issues=all_issues)