"""
Static Error Pattern Table

Provides repair hints based on verification layer and diagnosis.
"""

from typing import List

ERROR_PATTERNS = {
    "L1_keyerror": {
        "repair_hints": [
            "Use data.get('key', default) instead of data['key'] for optional fields",
            "Check if the key exists before accessing: if 'key' in data",
            "Provide default values: data.get('initial_inventory', {})"
        ]
    },
    "L1_typeerror": {
        "repair_hints": [
            "COMMON GUROBI ERRORS:",
            "",
            "1. 'unhashable type: list' - using list as dict key:",
            "   FIX: edges = [tuple(e) for e in data.get('network', {}).get('edges', [])]",
            "",
            "2. 'unsupported operand type(s) for *: float and GenExprMax' - gp.max_() returns expression, not value:",
            "   WRONG: cost * gp.max_(0, I[p,l,t] - threshold)",
            "   FIX: Use auxiliary variable: W[p,l,t] >= I[p,l,t] - threshold; W[p,l,t] >= 0",
            "",
            "3. '>' not supported between Var and int - comparing Gurobi Var directly:",
            "   WRONG: if I[p,l,t] > 0: ...",
            "   FIX: Use indicator constraints or Big-M formulation",
            "",
            "4. 'Var object has no attribute' - accessing Var value before optimization:",
            "   WRONG: I[p,l,t].X in constraint building",
            "   FIX: Use variable directly in expressions, not .X attribute"
        ]
    },
    "L1_indexerror": {
        "repair_hints": [
            "Check array bounds: period t uses index [t-1] for 0-indexed lists",
            "Verify loop ranges match data dimensions",
            "Use len() to check array sizes before indexing"
        ]
    },
    "L1_syntax": {
        "repair_hints": [
            "Check for missing colons, parentheses, or indentation",
            "Verify f-string syntax",
            "Check for unmatched brackets"
        ]
    },
    "L2_infeasible": {
        "repair_hints": [
            "Add lost_sales variable L[p,l,t] as slack in demand constraint",
            "Check RHS values for consistency across constraints",
            "Verify demand <= total_supply is achievable"
        ]
    },
    "L2_unbounded": {
        "repair_hints": [
            "Add lower/upper bounds to all decision variables",
            "Check for missing capacity constraints",
            "Verify objective direction (minimize vs maximize)"
        ]
    },
    "L3_no_effect": {
        "repair_hints": [
            "The parameter is NOT connected to any decision variable in constraints",
            "ADD a constraint that uses this parameter - common patterns:",
            "  - Capacity: sum(var[...]) <= data['param'][key]",
            "  - Requirement: sum(var[...]) >= data['param'][key]",
            "  - Balance: var_in - var_out == data['param'][key]",
            "Check loop indices: data['param'][p] must index correctly with var[p,t]"
        ]
    },
    "L4_wrong_direction": {
        "repair_hints": [
            "Review constraint: should it be ≤ or ≥?",
            "Check coefficient signs in constraint",
            "Verify objective is minimize vs maximize"
        ]
    },
    "L5_boundary_no_effect": {
        "repair_hints": [
            "Check if constraint is redundant",
            "Review Big-M formulation if used",
            "Ensure constraint binds on decision variables"
        ]
    },
    "L6_initialization": {
        "repair_hints": [
            "Add: I[p,l,1,a] = 0 for all a < shelf_life[p]",
            "Initialize all inventory at t=1 to zero"
        ]
    },
    "L6_lost_sales": {
        "repair_hints": [
            "Add L[p,l,t] >= 0 variable for lost sales",
            "Modify demand constraint: sales + L = demand"
        ]
    },
    "L6_substitution": {
        "repair_hints": [
            "Create S[p_from, p_to, l, t] substitution variable",
            "Add demand_route: S_out[p,l,t] <= demand[p,l,t]"
        ]
    },
    "L6_holding_cost": {
        "repair_hints": [
            "Change holding cost from I to (I - y) for end-of-period",
            "Ensure holding cost is on END-of-period inventory"
        ]
    }
}


def get_repair_hints(layer: int, diagnosis: str) -> List[str]:
    """Get repair hints based on failed layer and diagnosis."""
    diag_upper = diagnosis.upper()

    if layer == 1:
        # Runtime errors
        if "KEYERROR" in diag_upper:
            return ERROR_PATTERNS["L1_keyerror"]["repair_hints"]
        elif "TYPEERROR" in diag_upper:
            return ERROR_PATTERNS["L1_typeerror"]["repair_hints"]
        elif "INDEXERROR" in diag_upper:
            return ERROR_PATTERNS["L1_indexerror"]["repair_hints"]
        elif "SYNTAXERROR" in diag_upper:
            return ERROR_PATTERNS["L1_syntax"]["repair_hints"]
        else:
            # Generic runtime error hints
            return [
                "Check for missing imports or undefined variables",
                "Verify all data fields exist before accessing",
                "Use try/except or data.get() for optional data"
            ]

    if layer == 2:
        if "INFEASIBLE" in diagnosis.upper():
            return ERROR_PATTERNS["L2_infeasible"]["repair_hints"]
        elif "UNBOUNDED" in diagnosis.upper():
            return ERROR_PATTERNS["L2_unbounded"]["repair_hints"]
    elif layer == 3:
        return ERROR_PATTERNS["L3_no_effect"]["repair_hints"]
    elif layer == 4:
        return ERROR_PATTERNS["L4_wrong_direction"]["repair_hints"]
    elif layer == 5:
        return ERROR_PATTERNS["L5_boundary_no_effect"]["repair_hints"]
    elif layer == 6:
        diag = diagnosis.lower()
        if "initialization" in diag:
            return ERROR_PATTERNS["L6_initialization"]["repair_hints"]
        elif "lost_sales" in diag:
            return ERROR_PATTERNS["L6_lost_sales"]["repair_hints"]
        elif "substitution" in diag:
            return ERROR_PATTERNS["L6_substitution"]["repair_hints"]
        elif "holding" in diag:
            return ERROR_PATTERNS["L6_holding_cost"]["repair_hints"]
    return []


def format_repair_guidance(layer: int, diagnosis: str) -> str:
    """Format repair guidance for LLM repair prompt."""
    hints = get_repair_hints(layer, diagnosis)
    if not hints:
        return ""
    lines = ["\n## Repair Guidance:\n"]
    for i, hint in enumerate(hints, 1):
        lines.append(f"{i}. {hint}")
    return "\n".join(lines)
