"""
ReLoop Source-Code Level Perturbation

AST-based perturbation for LLM-generated code that hardcodes numerical values
instead of reading from the data dict.

Two perturbation modes:
- data_dict:   Perturb data dict values (existing behavior, e.g. RetailOpt-190)
- source_code: Perturb hardcoded constants via AST rewriting (IndustryOR/MAMO)

Auto-detection + fallback: try data_dict first, fall back to source_code
if perturbation has no effect on the objective.
"""

import ast
import copy
import re
from typing import List, Dict, Any, Optional

from .param_utils import perturb_param, get_param_value, should_skip_param


# ============================================================================
# Exclusion rules
# ============================================================================

# Variable names that should NOT be perturbed (control / solver params, not
# optimization parameters).
EXCLUDED_NAMES = {
    # Loop / control variables
    'iter', 'max_iter', 'n_iter', 'num_iterations', 'max_iterations',
    'i', 'j', 'k', 't', 'n', 'm', 'idx',
    # Numerical precision
    'tolerance', 'tol', 'epsilon', 'eps', 'threshold',
    # Gurobi-related
    'TimeLimit', 'MIPGap', 'Threads', 'Seed', 'OutputFlag',
    'time_limit', 'mip_gap',
    # Generic non-parameters
    'status', 'num_vars', 'num_constraints', 'count',
    'M', 'BigM', 'big_m',  # Big-M constants
}

MIN_PERTURBABLE_VALUE = 1e-4   # Values smaller than this are likely precision params
MAX_PERTURBABLE_VALUE = 1e10   # Values larger than this are likely Big-M


# ============================================================================
# 1A: Extract perturbable parameters from source code
# ============================================================================

def is_gurobi_param_setting(node: ast.AST) -> bool:
    """Check if node is a Gurobi parameter setting like m.Params.xxx = yyy."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        # m.Params.OutputFlag = 0
        if isinstance(target, ast.Attribute):
            val = target
            while isinstance(val, ast.Attribute):
                if val.attr in ('Params', 'params'):
                    return True
                val = val.value
    return False


def should_perturb(name: str, value: float) -> bool:
    """Decide whether a variable should be perturbed."""
    # Exclude known non-parameter variable names
    name_lower = name.lower()
    for excluded in EXCLUDED_NAMES:
        if excluded.lower() == name_lower:
            return False

    # Exclude zero values (multiplying by factor still yields 0)
    if value == 0:
        return False

    # Exclude values too small or too large
    abs_val = abs(value)
    if abs_val < MIN_PERTURBABLE_VALUE or abs_val > MAX_PERTURBABLE_VALUE:
        return False

    # Exclude small integers (0, 1, 2) that are likely indices, not parameters.
    # Larger integers like capacity = 500 are kept.
    if isinstance(value, int) and abs(value) <= 2:
        return False

    return True


def extract_perturbable_params(code: str) -> List[Dict[str, Any]]:
    """
    Extract all perturbable numeric parameters from source code via AST.

    Handles assignment patterns:
    1. Simple:  capacity = 500
    2. List:    costs = [10, 20, 15]
    3. Dict:    demand = {"A": 100, "B": 200}
    4. Nested:  data = {"costs": {"A": 10}, "demand": [100]}

    Returns list of dicts with keys: name, value, location, access_path.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    params: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        # Skip Gurobi parameter settings
        if is_gurobi_param_setting(node):
            continue

        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        if not isinstance(node.targets[0], ast.Name):
            continue

        var_name = node.targets[0].id
        value_node = node.value

        # Case 1: Simple numeric assignment  capacity = 500
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, (int, float)):
            if should_perturb(var_name, value_node.value):
                params.append({
                    'name': var_name,
                    'value': value_node.value,
                    'location': (value_node.lineno, value_node.col_offset),
                    'access_path': var_name,
                })

        # Case 2: List assignment  costs = [10, 20, 15]
        elif isinstance(value_node, ast.List):
            for idx, elt in enumerate(value_node.elts):
                if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                    if should_perturb(f"{var_name}[{idx}]", elt.value):
                        params.append({
                            'name': f'{var_name}[{idx}]',
                            'value': elt.value,
                            'location': (elt.lineno, elt.col_offset),
                            'access_path': f'{var_name}.{idx}',
                        })

        # Case 3: Dict assignment  demand = {"A": 100, "B": 200}
        elif isinstance(value_node, ast.Dict):
            _extract_dict_params(var_name, value_node, params)

    return params


def _extract_dict_params(prefix: str, dict_node: ast.Dict, params: list):
    """Recursively extract numeric parameters from a dict AST node."""
    for key_node, val_node in zip(dict_node.keys, dict_node.values):
        # Get string representation of key
        if isinstance(key_node, ast.Constant):
            key_str = str(key_node.value)
        else:
            continue

        full_name = f'{prefix}["{key_str}"]'
        access_path = f'{prefix}.{key_str}'

        if isinstance(val_node, ast.Constant) and isinstance(val_node.value, (int, float)):
            if should_perturb(full_name, val_node.value):
                params.append({
                    'name': full_name,
                    'value': val_node.value,
                    'location': (val_node.lineno, val_node.col_offset),
                    'access_path': access_path,
                })
        elif isinstance(val_node, ast.Dict):
            _extract_dict_params(f'{prefix}["{key_str}"]', val_node, params)
        elif isinstance(val_node, ast.List):
            for idx, elt in enumerate(val_node.elts):
                if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                    if should_perturb(f'{full_name}[{idx}]', elt.value):
                        params.append({
                            'name': f'{full_name}[{idx}]',
                            'value': elt.value,
                            'location': (elt.lineno, elt.col_offset),
                            'access_path': f'{access_path}.{idx}',
                        })


# ============================================================================
# 1B: Generate perturbed code
# ============================================================================

def perturb_code(code: str, access_path: str, factor: float) -> str:
    """
    Multiply the numeric constant at *access_path* by *factor* inside *code*.

    Uses text-based replacement (no ast.unparse) to preserve formatting and
    work on Python 3.8+.

    Args:
        code: Original source code string.
        access_path: From extract_perturbable_params (e.g. "capacity", "costs.0",
                     "demand.A").
        factor: Multiplicative factor (e.g. 1.2 for +20%).

    Returns:
        Modified code string. Returns original code if the path is not found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Find the target AST node
    target_node = _find_target_node(tree, access_path)
    if target_node is None:
        return code

    # Apply text-based replacement at the node's source location
    return _replace_node_value(code, target_node, factor)


def _find_target_node(tree: ast.AST, access_path: str) -> Optional[ast.AST]:
    """Find the AST Constant node matching *access_path*."""
    for node in ast.walk(tree):
        if is_gurobi_param_setting(node):
            continue
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue

        var_name = node.targets[0].id
        value_node = node.value

        # Simple assignment: access_path == var_name
        if access_path == var_name:
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, (int, float)):
                return value_node

        # Dotted path
        if '.' in access_path:
            parts = access_path.split('.')
            if parts[0] != var_name:
                continue

            # List element: "costs.0"
            if isinstance(value_node, ast.List):
                try:
                    idx = int(parts[1])
                    elt = value_node.elts[idx]
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                        return elt
                except (ValueError, IndexError):
                    pass

            # Dict: "demand.A" or "data.costs.A"
            if isinstance(value_node, ast.Dict):
                found = _find_dict_node(value_node, parts[1:])
                if found is not None:
                    return found

    return None


def _find_dict_node(dict_node: ast.Dict, path_parts: list) -> Optional[ast.AST]:
    """Recursively find a Constant node inside a dict AST."""
    if not path_parts:
        return None

    target_key = path_parts[0]

    for key_node, val_node in zip(dict_node.keys, dict_node.values):
        if isinstance(key_node, ast.Constant) and str(key_node.value) == target_key:
            if len(path_parts) == 1:
                if isinstance(val_node, ast.Constant) and isinstance(val_node.value, (int, float)):
                    return val_node
            else:
                remaining = path_parts[1:]
                if isinstance(val_node, ast.Dict):
                    return _find_dict_node(val_node, remaining)
                elif isinstance(val_node, ast.List):
                    try:
                        idx = int(remaining[0])
                        elt = val_node.elts[idx]
                        if isinstance(elt, ast.Constant) and len(remaining) == 1:
                            return elt
                    except (ValueError, IndexError):
                        pass
    return None


def _replace_node_value(code: str, node: ast.AST, factor: float) -> str:
    """Replace a numeric literal in *code* at the position of *node*."""
    lines = code.split('\n')
    # AST line numbers are 1-based
    line_idx = node.lineno - 1
    col_start = node.col_offset

    if line_idx >= len(lines):
        return code

    line = lines[line_idx]

    # Determine end column: use end_col_offset if available (Python 3.8+)
    end_col = getattr(node, 'end_col_offset', None)
    if end_col is None:
        # Fallback: find the end of the numeric literal manually
        end_col = col_start
        while end_col < len(line) and (line[end_col].isdigit() or line[end_col] in '.eE+-'):
            end_col += 1
    # Verify multi-line doesn't happen (it shouldn't for numeric literals)
    end_line = getattr(node, 'end_lineno', node.lineno)
    if end_line != node.lineno:
        return code

    old_literal = line[col_start:end_col]
    new_value = _apply_factor(node.value, factor)

    # Format the new value
    if isinstance(new_value, int):
        new_literal = str(new_value)
    else:
        new_literal = repr(new_value)

    # Replace in the line
    new_line = line[:col_start] + new_literal + line[end_col:]
    lines[line_idx] = new_line
    return '\n'.join(lines)


def _apply_factor(value, factor: float):
    """Apply multiplicative factor, preserving int type when possible."""
    new_val = value * factor
    if isinstance(value, int):
        new_val = int(round(new_val))
    return new_val


# ============================================================================
# 1C: Auto-detect perturbation mode + fallback
# ============================================================================

def detect_perturbation_mode(code: str, data: Optional[dict] = None) -> str:
    """
    Detect which perturbation strategy to use.

    Returns:
        "data_dict"   - Code reads params from data dict (RetailOpt-190 style)
        "source_code" - Code hardcodes numeric values (IndustryOR/MAMO style)
        "hybrid"      - Both patterns present
    """
    if data is None:
        return "source_code"

    # Check if code accesses the data dict
    data_access_patterns = [
        'data[',
        'data.get(',
        "data['",
        'data["',
    ]
    has_data_access = any(p in code for p in data_access_patterns)

    # Check if code has hardcoded numeric values
    hardcode_params = extract_perturbable_params(code)
    # A few numeric assignments may just be control params; require >5 to
    # consider the code "hardcoded".
    has_hardcode = len(hardcode_params) > 5

    if has_data_access and not has_hardcode:
        return "data_dict"
    elif has_hardcode and not has_data_access:
        return "source_code"
    else:
        return "hybrid"


def _match_param(params: List[Dict], param_name: str) -> Optional[Dict]:
    """Fuzzy-match *param_name* against a list of extracted code parameters."""
    # Exact match
    for p in params:
        if p['name'] == param_name or p['access_path'] == param_name:
            return p

    # Normalised match
    param_name_lower = param_name.lower().replace('_', '').replace('-', '')
    for p in params:
        p_lower = p['name'].lower().replace('_', '').replace('-', '')
        if param_name_lower in p_lower or p_lower in param_name_lower:
            return p

    return None


def _apply_data_perturbation(data: dict, param_name: str, factor: float) -> bool:
    """
    Apply perturbation inside the data dict using existing param_utils.

    Returns True if the parameter was found and modified.
    """
    val = get_param_value(data, param_name)
    if val is None:
        return False

    # perturb_param returns a new dict; we need to mutate data in-place.
    perturbed = perturb_param(data, param_name, factor)
    new_val = get_param_value(perturbed, param_name)
    if new_val is None:
        return False

    # Copy the perturbed value back into data.
    keys = param_name.split(".")
    obj = data
    for key in keys[:-1]:
        obj = obj[key]
    obj[keys[-1]] = new_val
    return True


def run_perturbation(
    code: str,
    data: Optional[dict],
    param_name: str,
    factor: float,
    executor_fn,
    mode: str = "auto",
    baseline_obj: Optional[float] = None,
) -> Optional[float]:
    """
    Unified perturbation interface with auto-fallback.

    1. If mode allows data_dict: try data-level perturbation first.
    2. If objective is unchanged (or mode is source_code): try AST perturbation.

    Args:
        code:          LLM-generated source code.
        data:          Data dict (may be None for pure source_code mode).
        param_name:    Parameter to perturb.
        factor:        Multiplicative factor (e.g. 1.2 = +20%).
        executor_fn:   Callable(code, data) -> Optional[float]  returning objective.
        mode:          "auto", "data_dict", "source_code".
        baseline_obj:  Pre-computed baseline objective (avoids redundant execution).

    Returns:
        Perturbed objective value, or None on failure.
    """
    if mode == "auto":
        mode = detect_perturbation_mode(code, data)

    # Compute baseline if not provided
    if baseline_obj is None:
        baseline_obj = executor_fn(code, data)
    if baseline_obj is None:
        return None

    # Strategy 1: data-dict perturbation
    if mode in ("data_dict", "hybrid") and data is not None:
        data_perturbed = copy.deepcopy(data)
        if _apply_data_perturbation(data_perturbed, param_name, factor):
            # Strip json.loads override so the perturbed data dict is actually used
            exec_code = strip_data_override(code) if has_data_override(code) else code
            z_perturbed = executor_fn(exec_code, data_perturbed)
            if z_perturbed is not None and abs(z_perturbed - baseline_obj) > 1e-10:
                return z_perturbed

    # Strategy 2: source-code perturbation (fallback)
    if mode in ("source_code", "hybrid"):
        params = extract_perturbable_params(code)
        matched = _match_param(params, param_name)
        if matched:
            code_perturbed = perturb_code(code, matched['access_path'], factor)
            if code_perturbed != code:
                z_perturbed = executor_fn(code_perturbed, data)
                return z_perturbed

    return None


# ============================================================================
# Strip json.loads data override for perturbation
# ============================================================================

def has_data_override(code: str) -> bool:
    """Check if code overrides the `data` variable via json.loads."""
    return bool(re.search(r'\bdata\s*=\s*json\.loads\(', code))


def strip_data_override(code: str) -> str:
    """Strip json.loads data override so externally-injected data dict is used.

    LLM-generated code often embeds data as:
        data_json = \"\"\"{...}\"\"\"
        data = json.loads(data_json)

    This overwrites the external `data` variable, making data-dict perturbation
    useless.  Stripping these lines lets the executor-injected (perturbed)
    `data` flow through to the code.

    Returns the modified code, or the original if no override was found.
    """
    # Pattern 1: data = json.loads(VARNAME) — indirect via a string variable
    m = re.search(r'^data\s*=\s*json\.loads\((\w+)\)', code, re.MULTILINE)
    if m:
        source_var = m.group(1)
        # Remove the json.loads assignment line
        code = code[:m.start()] + code[m.end():]
        # Remove the source variable's triple-quoted string assignment
        # Handles both """ and ''' delimiters
        for delim in ['"""', "'''"]:
            esc = re.escape(delim)
            pattern = re.compile(
                rf'^{re.escape(source_var)}\s*=\s*{esc}.*?{esc}\s*$',
                re.MULTILINE | re.DOTALL,
            )
            code = pattern.sub('', code, count=1)
        return code

    # Pattern 2: data = json.loads("""...""") — inline triple-quoted string
    for delim in ['"""', "'''"]:
        esc = re.escape(delim)
        pattern = re.compile(
            rf'^data\s*=\s*json\.loads\(\s*{esc}.*?{esc}\s*\)\s*$',
            re.MULTILINE | re.DOTALL,
        )
        code = pattern.sub('', code, count=1)

    # Pattern 3: data = json.loads('...') or data = json.loads("...") — single-line
    code = re.sub(
        r'^data\s*=\s*json\.loads\(.+\)\s*$', '', code, count=1, flags=re.MULTILINE
    )

    return code


# ============================================================================
# Source-code parameter list for L2
# ============================================================================

def get_source_code_param_names(code: str, max_params: int = 30) -> List[str]:
    """
    Return parameter names extracted from source code, capped at *max_params*.

    Each name is an access_path suitable for passing to perturb_code().
    """
    params = extract_perturbable_params(code)
    return [p['access_path'] for p in params[:max_params]]
