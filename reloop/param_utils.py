"""
ReLoop Parameter Utilities

Functions:
- Extract numeric parameters
- Parameter perturbation
- Role inference
- Skip determination
"""

import copy
from typing import Dict, Any, List, Tuple, Optional
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
        # Core
        "demand", "need", "order", "request", "requirement",
        "target", "quota", "goal",
        # Diet/Nutrition (MAMO)
        "protein", "carbohydrate", "carbs", "calorie", "calories",
        "fiber", "nutrient", "minimum", "at_least", "min_",
        # Labor/Production (IndustryOR, NL4OPT)
        "hours_needed", "labor_required", "units_required",
        "workers_needed", "trips", "deliveries",
        # Transport (MAMO warehouse)
        "required", "units_needed", "destination"
    ],
    ParameterRole.CAPACITY: [
        # Core
        "capacity", "supply", "limit", "available", "max",
        "bound", "cap", "budget", "resource",
        # Inventory/Storage
        "stock", "inventory", "storage", "warehouse",
        # Labor/Time (NL4OPT, IndustryOR)
        "hours_available", "labor_cap", "time_limit",
        "workers", "shifts", "overtime",
        # Physical constraints
        "weight", "volume", "space", "area",
        "at_most", "maximum", "upper"
    ],
    ParameterRole.COST: [
        # Core
        "cost", "price", "penalty", "expense", "fee",
        "rate", "holding", "waste", "transport",
        # Specific costs
        "shipping", "purchasing", "production_cost",
        "labor_cost", "material", "raw_material",
        "wage", "salary", "overtime_cost",
        # Loss
        "loss", "spoilage", "damage"
    ],
    ParameterRole.REVENUE: [
        # Core
        "revenue", "profit", "income", "benefit",
        "reward", "selling_price", "value",
        # Sales
        "sales_price", "unit_price", "margin",
        "return", "gain", "earning"
    ]
}

EXPECTED_DIRECTION_MINIMIZE = {
    ParameterRole.REQUIREMENT: "increase",
    ParameterRole.CAPACITY: "decrease",
    ParameterRole.COST: "increase",
    ParameterRole.REVENUE: "decrease"
}


def extract_numeric_params(data: Dict, prefix: str = "") -> List[str]:
    """
    Recursively extract all numeric parameter paths.

    Returns: ["param1", "nested.param2", ...]
    """
    params = []
    skip_keys = {"name", "products", "locations", "network", "nodes", "edges"}

    for key, value in data.items():
        if key in skip_keys:
            continue

        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            params.append(path)
        elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
            params.append(path)
        elif isinstance(value, dict):
            # Check if it's a numeric dictionary
            if all(isinstance(v, (int, float, list)) for v in value.values()):
                params.append(path)
            else:
                params.extend(extract_numeric_params(value, path))

    return params


def get_param_value(data: Dict, param_path: str) -> Any:
    """Get parameter value by path."""
    keys = param_path.split(".")
    obj = data
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
        else:
            return None
    return obj


def infer_param_role(param_path: str) -> ParameterRole:
    """Infer parameter role from name."""
    name = param_path.split(".")[-1].lower()
    full_path = param_path.lower()

    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in name or kw in full_path:
                return role

    return ParameterRole.UNKNOWN


def get_expected_direction(role: ParameterRole, obj_sense: str) -> Optional[str]:
    """Get expected direction of objective change when parameter increases."""
    if role == ParameterRole.UNKNOWN:
        return None

    direction = EXPECTED_DIRECTION_MINIMIZE.get(role)

    # Reverse direction for maximization
    if obj_sense == "maximize":
        if direction == "increase":
            return "decrease"
        elif direction == "decrease":
            return "increase"

    return direction


def perturb_param(data: Dict, param_path: str, factor: float) -> Dict:
    """
    Create a perturbed copy of data.

    factor > 1: increase
    factor < 1: decrease
    """
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

    def apply_factor(v, f):
        """Apply factor while preserving type."""
        if isinstance(v, int):
            return max(1, int(round(v * f)))  # Keep int, minimum 1
        elif isinstance(v, float):
            return v * f
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


def set_param(data: Dict, param_path: str, new_value: Any) -> Dict:
    """Create a copy with parameter set to specific value."""
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
        for k in value:
            if isinstance(value[k], (int, float)):
                value[k] = new_value
            elif isinstance(value[k], list):
                value[k] = [new_value] * len(value[k])

    return data_copy


def should_skip_param(data: Dict, param_path: str) -> Tuple[bool, str]:
    """
    Determine if parameter should be skipped in testing.

    Skip conditions:
    - Value is None
    - Value is zero or near-zero
    - Value is Big-M (>=90000)
    """
    value = get_param_value(data, param_path)

    if value is None:
        return True, "not found"

    # Zero value
    if isinstance(value, (int, float)) and abs(value) < 1e-6:
        return True, "zero value"

    # Big-M
    if isinstance(value, (int, float)) and value >= 90000:
        return True, "Big-M value"

    # List all zeros
    if isinstance(value, list):
        if all(abs(v) < 1e-6 for v in value if isinstance(v, (int, float))):
            return True, "all zeros"

    # Dict all zeros
    if isinstance(value, dict):
        if all(abs(v) < 1e-6 for v in value.values() if isinstance(v, (int, float))):
            return True, "all zeros"

    return False, ""
