from typing import Dict, Any, List

from .agent_types import ExecutionResult, SemanticReport


def _has_prefix(items: List[str], prefix: str) -> bool:
    return any(it.startswith(prefix) for it in items)


def semantic_check(data: Dict[str, Any], exec_result: ExecutionResult) -> SemanticReport:
    if not exec_result.success:
        return SemanticReport(
            valid=False,
            missing_prefixes=[],
            unexpected_modules=[],
            suspicious=["execution_failed"],
        )

    missing: List[str] = []
    unexpected: List[str] = []
    suspicious: List[str] = []

    constr_names = exec_result.constr_names or []
    var_names = exec_result.var_names or []

    # Shelf life handling
    shelf_life = data.get("shelf_life", {})
    if any(v > 1 for v in shelf_life.values()):
        if not _has_prefix(constr_names, "aging") and not _has_prefix(constr_names, "expire_clear"):
            missing.append("aging/expire_clear")

    # Substitution
    network = data.get("network", {})
    sub_edges = network.get("sub_edges", [])
    if sub_edges:
        if not _has_prefix(constr_names, "demand_route") and not _has_prefix(var_names, "S_"):
            missing.append("demand_route")
        if not _has_prefix(constr_names, "sales_conservation"):
            missing.append("sales_conservation")

    # Production capacity
    prod_cap = data.get("production_cap", {})
    if prod_cap:
        if not _has_prefix(constr_names, "prod_cap"):
            evidence = any(name.startswith("Q_") or name.startswith("X_") for name in var_names)
            if not evidence:
                missing.append("prod_cap")

    # Storage capacity
    if data.get("cold_capacity") and data.get("cold_usage"):
        if not _has_prefix(constr_names, "storage_cap"):
            if not any(name.startswith("I_") for name in var_names):
                missing.append("storage_cap")

    # Transshipment presence/absence
    # NOTE:
    # - Do NOT treat "X_" (purchase/order) as transshipment evidence.
    # - Only treat explicit transshipment constructs as evidence:
    #   TR_ variables and/or constraints explicitly named with "transshipment".
    trans_edges = network.get("trans_edges", [])
    has_trans_vars = _has_prefix(var_names, "TR_") or _has_prefix(constr_names, "transshipment")

    if trans_edges and not has_trans_vars:
        missing.append("transshipment")
    if not trans_edges and has_trans_vars:
        unexpected.append("transshipment")

    # MOQ / pack / budget / wastecap gating
    constraints = data.get("constraints", {})
    moq = constraints.get("moq", 0)
    pack_size = constraints.get("pack_size", 1)
    budget = constraints.get("budget_per_period")
    wastecap = constraints.get("waste_limit_pct")

    if moq and not _has_prefix(constr_names, "moq_lb") and not _has_prefix(constr_names, "moq_ub"):
        missing.append("moq")
    if pack_size and pack_size > 1 and not _has_prefix(constr_names, "pack"):
        missing.append("pack")
    if budget is not None and not _has_prefix(constr_names, "budget"):
        missing.append("budget")
    if wastecap is not None and not _has_prefix(constr_names, "wastecap"):
        missing.append("wastecap")

    if moq == 0 and _has_prefix(constr_names, "moq"):
        unexpected.append("moq")
    if (not pack_size or pack_size == 1) and _has_prefix(constr_names, "pack"):
        unexpected.append("pack")
    if budget is None and _has_prefix(constr_names, "budget"):
        unexpected.append("budget")
    if wastecap is None and _has_prefix(constr_names, "wastecap"):
        unexpected.append("wastecap")

    valid = len(missing) == 0 and len(unexpected) == 0
    return SemanticReport(
        valid=valid,
        missing_prefixes=missing,
        unexpected_modules=unexpected,
        suspicious=suspicious,
    )
