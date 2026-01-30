from __future__ import annotations

from typing import List, Any, Dict

from pydantic import BaseModel

from ..state import AuditResult, AuditViolation, ModelHandle


class AuditInput(BaseModel):
    archetype_id: str
    canonical_schema: Any
    handle: ModelHandle


class AuditOutput(BaseModel):
    audit: AuditResult


class AuditSkill:
    """
    Purpose:
        Validate feasibility of solved solutions across archetypes.
    Determinism:
        Pure arithmetic over solver outputs; deterministic.
    Failure modes:
        Missing solution values -> marks audit as failed with symptom tags.
    Logging:
        Returns per-constraint slacks and aggregate audit_score.
    """

    def run(self, inp: AuditInput) -> AuditOutput:
        arch = inp.archetype_id
        if arch == "diet_lp":
            arch = "allocation_lp"
        if arch == "allocation_lp":
            return AuditOutput(audit=self._audit_allocation(inp.canonical_schema, inp.handle))
        if arch == "transport_flow":
            return AuditOutput(audit=self._audit_transport(inp.canonical_schema, inp.handle))
        if arch == "assignment":
            return AuditOutput(audit=self._audit_assignment(inp.canonical_schema, inp.handle))
        if arch == "facility_location":
            return AuditOutput(audit=self._audit_facility(inp.canonical_schema, inp.handle))
        if arch == "retail_inventory":
            return AuditOutput(audit=self._audit_retail(inp.canonical_schema, inp.handle))
        return AuditOutput(audit=AuditResult(passed=False, symptom_tags=["AUDIT_VIOLATION"]))

    def _audit_allocation(self, schema, handle: ModelHandle) -> AuditResult:
        variables = handle.variables
        audit_score = 1.0
        violations: List[AuditViolation] = []
        for feat in schema.feature_order:
            lhs = 0.0
            for item in schema.items:
                var = variables.get(item.name)
                val = getattr(var, "X", None)
                if val is None:
                    return AuditResult(passed=False, audit_score=0.0, symptom_tags=["AUDIT_VIOLATION"], violations=[])
                lhs += item.features.get(feat, 0.0) * float(val)
            rhs = schema.requirements.get(feat, 0.0)
            slack = lhs - rhs
            if slack < -1e-6:
                audit_score -= 0.1
                violations.append(AuditViolation(name=feat, lhs=lhs, rhs=rhs, slack=slack))
        passed = len(violations) == 0
        tags = [] if passed else ["AUDIT_VIOLATION"]
        return AuditResult(passed=passed, violations=violations, audit_score=max(0.0, audit_score), symptom_tags=tags)

    def _audit_transport(self, schema: Dict[str, Any], handle: ModelHandle) -> AuditResult:
        vars_dict: Dict[Any, Any] = handle.variables
        violations: List[AuditViolation] = []
        audit_score = 1.0
        def _get(i, j):
            return getattr(vars_dict.get((i, j)), "X", None) or getattr(vars_dict.get(f"{i}__{j}"), "X", None) or 0.0
        # supply
        for i in schema["sources"]:
            lhs = sum(_get(i, j) for j in schema["sinks"])
            rhs = schema["supply"][i]
            slack = rhs - lhs
            if slack < -1e-6:
                violations.append(AuditViolation(name=f"supply_{i}", lhs=lhs, rhs=rhs, slack=slack))
                audit_score -= 0.1
        # demand
        for j in schema["sinks"]:
            lhs = sum(_get(i, j) for i in schema["sources"])
            rhs = schema["demand"][j]
            slack = lhs - rhs
            if slack < -1e-6:
                violations.append(AuditViolation(name=f"demand_{j}", lhs=lhs, rhs=rhs, slack=slack))
                audit_score -= 0.1
        passed = len(violations) == 0
        return AuditResult(passed=passed, violations=violations, audit_score=audit_score, symptom_tags=[] if passed else ["AUDIT_VIOLATION"])

    def _audit_assignment(self, schema: Dict[str, Any], handle: ModelHandle) -> AuditResult:
        vars_dict = handle.variables
        violations: List[AuditViolation] = []
        audit_score = 1.0

        def _get(w, t):
            return getattr(vars_dict.get((w, t)), "X", None) or getattr(vars_dict.get(f"{w}__{t}"), "X", None) or 0.0

        for w in schema["workers"]:
            lhs = sum(_get(w, t) for t in schema["tasks"])
            slack = lhs - 1.0
            if abs(slack) > 1e-6:
                violations.append(AuditViolation(name=f"worker_{w}", lhs=lhs, rhs=1.0, slack=slack))
                audit_score -= 0.1
        for t in schema["tasks"]:
            lhs = sum(_get(w, t) for w in schema["workers"])
            slack = lhs - 1.0
            if abs(slack) > 1e-6:
                violations.append(AuditViolation(name=f"task_{t}", lhs=lhs, rhs=1.0, slack=slack))
                audit_score -= 0.1
        passed = len(violations) == 0
        return AuditResult(passed=passed, violations=violations, audit_score=audit_score, symptom_tags=[] if passed else ["AUDIT_VIOLATION"])

    def _audit_facility(self, schema: Dict[str, Any], handle: ModelHandle) -> AuditResult:
        vars_dict = handle.variables
        open_vars = vars_dict.get("open", {})
        ship_vars = vars_dict.get("ship", {})
        violations: List[AuditViolation] = []
        audit_score = 1.0
        for c in schema["customers"]:
            lhs = sum(getattr(ship_vars.get((f["name"], c)), "X", 0.0) for f in schema["facilities"])
            rhs = schema["demand"][c]
            slack = lhs - rhs
            if abs(slack) > 1e-6:
                violations.append(AuditViolation(name=f"demand_{c}", lhs=lhs, rhs=rhs, slack=slack))
                audit_score -= 0.1
        for f in schema["facilities"]:
            for c in schema["customers"]:
                ship_val = getattr(ship_vars.get((f["name"], c)), "X", 0.0)
                open_val = getattr(open_vars.get(f["name"]), "X", 0.0)
                if ship_val - open_val > 1e-6:
                    violations.append(AuditViolation(name=f"link_{f['name']}_{c}", lhs=ship_val, rhs=open_val, slack=ship_val - open_val))
                    audit_score -= 0.1
            if f["capacity"] != float("inf"):
                lhs = sum(getattr(ship_vars.get((f["name"], c)), "X", 0.0) for c in schema["customers"])
                rhs = f["capacity"]
                if lhs - rhs > 1e-6:
                    violations.append(AuditViolation(name=f"cap_{f['name']}", lhs=lhs, rhs=rhs, slack=lhs - rhs))
                    audit_score -= 0.1
        passed = len(violations) == 0
        return AuditResult(passed=passed, violations=violations, audit_score=audit_score, symptom_tags=[] if passed else ["AUDIT_VIOLATION"])

    def _audit_retail(self, schema: Any, handle: ModelHandle) -> AuditResult:
        vars_dict = handle.variables
        sales = vars_dict.get("sales", {})
        lost = vars_dict.get("lost", {})
        prod = vars_dict.get("prod", {})
        inv = vars_dict.get("inv", {})
        if not sales:
            return AuditResult(passed=False, symptom_tags=["AUDIT_VIOLATION"], audit_score=0.0)
        violations: List[AuditViolation] = []
        audit_score = 1.0
        # Demand balance: sales+lost == demand within data stored in metadata
        data = handle.metadata.get("retail_data", {})
        products = data.get("products", [])
        locations = data.get("locations", [])
        periods = data.get("periods", [])
        demand = data.get("demand", {})
        demand_share = data.get("demand_share", {l: 1.0 for l in locations})
        for p in products:
            for l in locations:
                for t_idx, t in enumerate(periods):
                    demand_loc = demand[p][t_idx] * demand_share.get(l, 1.0)
                    lhs = getattr(sales.get((p, l, t)), "X", 0.0) + getattr(lost.get((p, l, t)), "X", 0.0)
                    slack = lhs - demand_loc
                    if abs(slack) > 1e-6:
                        violations.append(AuditViolation(name=f"demand_{p}_{l}_{t}", lhs=lhs, rhs=demand_loc, slack=slack))
                        audit_score -= 0.1
                # inventory balance
                prev_inv = 0.0
                for t_idx, t in enumerate(periods):
                    inv_t = getattr(inv.get((p, l, t)), "X", 0.0)
                    prod_t = getattr(prod.get((p, l, t)), "X", 0.0)
                    sales_t = getattr(sales.get((p, l, t)), "X", 0.0)
                    lhs = inv_t
                    rhs = prev_inv + prod_t - sales_t
                    slack = lhs - rhs
                    if abs(slack) > 1e-6:
                        violations.append(AuditViolation(name=f"balance_{p}_{l}_{t}", lhs=lhs, rhs=rhs, slack=slack))
                        audit_score -= 0.1
                    prev_inv = inv_t
        passed = len(violations) == 0
        return AuditResult(passed=passed, violations=violations, audit_score=audit_score, symptom_tags=[] if passed else ["AUDIT_VIOLATION"])


__all__ = ["AuditSkill", "AuditInput", "AuditOutput"]
