"""
Parameter Utilities for ReLoop

Handles:
- Extracting numeric parameters from data
- Perturbing parameters for sensitivity testing
- Inferring parameter roles from names (keyword matching)
"""

import copy
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class ParameterRole(Enum):
    """Parameter roles for sensitivity analysis"""
    REQUIREMENT = "requirement"
    CAPACITY = "capacity"
    COST = "cost"
    REVENUE = "revenue"
    UNKNOWN = "unknown"


ROLE_KEYWORDS = {
    ParameterRole.REQUIREMENT: [
        "demand", "need", "order", "request", "requirement", "target",
        "rhs", "quota", "goal", "customer", "demand_curve"
    ],
    ParameterRole.CAPACITY: [
        "capacity", "supply", "limit", "available", "max", "bound", "cap",
        "production_cap", "cold_capacity", "labor_cap", "storage", "budget"
    ],
    ParameterRole.COST: [
        "cost", "price", "penalty", "expense", "fee", "rate", "weight",
        "purchasing", "holding", "waste", "lost_sales", "inventory"
    ],
    ParameterRole.REVENUE: [
        "revenue", "profit", "income", "benefit", "reward", "selling_price"
    ]
}

EXPECTED_DIRECTION_MINIMIZE = {
    ParameterRole.REQUIREMENT: "increase",
    ParameterRole.CAPACITY: "decrease",
    ParameterRole.COST: "increase",
    ParameterRole.REVENUE: "decrease"
}


def extract_numeric_params(data: Dict[str, Any], prefix: str = "") -> List[str]:
    """Recursively extract all numeric parameter paths from data."""
    params = []
    for key, value in data.items():
        if key in ("name", "products", "locations", "network"):
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            params.append(path)
        elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
            params.append(path)
        elif isinstance(value, dict):
            if all(isinstance(v, (int, float, list)) for v in value.values()):
                params.append(path)
            else:
                params.extend(extract_numeric_params(value, path))
    return params


def infer_param_role(param_path: str) -> ParameterRole:
    """Infer parameter role from its name using keyword matching."""
    name = param_path.split(".")[-1].lower()
    full_path_lower = param_path.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in name or kw in full_path_lower:
                return role
    return ParameterRole.UNKNOWN


def get_expected_direction(role: ParameterRole, obj_sense: str = "minimize") -> Optional[str]:
    """Get expected direction of objective change when parameter increases."""
    if role == ParameterRole.UNKNOWN:
        return None
    direction = EXPECTED_DIRECTION_MINIMIZE.get(role)
    if obj_sense == "maximize" and role == ParameterRole.COST:
        direction = "decrease"
    return direction


def perturb_param(data: Dict[str, Any], param_path: str, factor: float) -> Dict[str, Any]:
    """Create a copy of data with one parameter multiplied by factor."""
    data_copy = copy.deepcopy(data)
    keys = param_path.split(".")
    obj = data_copy
    for key in keys[:-1]:
        if key not in obj:
            return data_copy
        obj = obj[key]
    final_key = keys[-1]
    if final_key not in obj:
        return data_copy
    value = obj[final_key]

    def apply_factor(v, factor):
        """Apply factor while preserving int type for int inputs."""
        if isinstance(v, int):
            return max(1, int(round(v * factor)))  # Preserve int, min 1
        elif isinstance(v, float):
            return v * factor
        return v

    if isinstance(value, (int, float)):
        obj[final_key] = apply_factor(value, factor)
    elif isinstance(value, list):
        obj[final_key] = [apply_factor(v, factor) for v in value]
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (int, float)):
                value[k] = apply_factor(v, factor)
            elif isinstance(v, list):
                value[k] = [apply_factor(x, factor) for x in v]
    return data_copy


def set_param(data: Dict[str, Any], param_path: str, new_value: Any) -> Dict[str, Any]:
    """Create a copy of data with one parameter set to a specific value."""
    data_copy = copy.deepcopy(data)
    keys = param_path.split(".")
    obj = data_copy
    for key in keys[:-1]:
        if key not in obj:
            return data_copy
        obj = obj[key]
    final_key = keys[-1]
    if final_key not in obj:
        return data_copy
    value = obj[final_key]
    if isinstance(value, (int, float)):
        obj[final_key] = new_value
    elif isinstance(value, list):
        obj[final_key] = [new_value] * len(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (int, float)):
                value[k] = new_value
            elif isinstance(v, list):
                value[k] = [new_value] * len(v)
    return data_copy


def get_param_value(data: Dict[str, Any], param_path: str) -> Any:
    """Get the value of a parameter by path."""
    keys = param_path.split(".")
    obj = data
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return None
    return obj


def is_effectively_zero(value: Any, threshold: float = 1e-6) -> bool:
    """Check if a parameter value is effectively zero (shouldn't affect objective)."""
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return abs(value) < threshold
    if isinstance(value, list):
        return all(abs(v) < threshold for v in value if isinstance(v, (int, float)))
    if isinstance(value, dict):
        return all(is_effectively_zero(v, threshold) for v in value.values())
    return False


def is_big_m_value(value: Any, threshold: float = 90000) -> bool:
    """Check if a parameter value is a 'big M' (very large, won't bind)."""
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value >= threshold
    if isinstance(value, list):
        return all(v >= threshold for v in value if isinstance(v, (int, float)))
    if isinstance(value, dict):
        return all(is_big_m_value(v, threshold) for v in value.values())
    return False


def should_skip_param(data: Dict[str, Any], param_path: str) -> Tuple[bool, str]:
    """
    Determine if a parameter should be skipped in monotonicity testing.

    Returns: (should_skip, reason)

    Skip conditions:
    1. Value is zero or near-zero (e.g., lead_time=0, return_rate=0)
    2. Value is Big M (very large capacity that won't bind)
    3. Parameter is explicitly disabled in the scenario
    """
    value = get_param_value(data, param_path)

    if value is None:
        return True, "not found"

    if is_effectively_zero(value):
        return True, "value is zero"

    # Check for Big M values (won't bind)
    if is_big_m_value(value):
        return True, "Big M (won't bind)"

    # Check for special cases
    param_name = param_path.split(".")[-1].lower()

    # lead_time = 0 means immediate availability, no constraint
    if "lead_time" in param_name and is_effectively_zero(value):
        return True, "lead_time=0 (immediate)"

    # return_rate = 0 means no returns, no constraint
    if "return_rate" in param_name and is_effectively_zero(value):
        return True, "return_rate=0 (no returns)"

    return False, ""


def filter_testable_params(data: Dict[str, Any], params: List[str]) -> List[Tuple[str, str]]:
    """
    Filter parameters to only those that should be tested.

    Returns: List of (param_path, skip_reason) - skip_reason is empty if testable
    """
    results = []
    for param in params:
        should_skip, reason = should_skip_param(data, param)
        results.append((param, reason if should_skip else ""))
    return results
