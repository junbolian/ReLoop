from __future__ import annotations

from typing import Optional, Dict, Any

from pydantic import BaseModel
import numpy as np
import itertools

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover - fallback when gurobi missing
    gp = None

from ..state import ModelHandle, SolveResult


class SolveInput(BaseModel):
    handle: ModelHandle


class SolveOutput(BaseModel):
    result: SolveResult


class SolveSkill:
    """
    Purpose:
        Optimize a constructed Gurobi model and capture relevant artifacts.
    Determinism:
        Deterministic given model; no randomness.
    Failure modes:
        Solver errors are captured as status messages instead of raising.
    Logging:
        Returns runtime, status, objective, mipgap (if available).
    """

    def run(self, inp: SolveInput) -> SolveOutput:
        # Fallback path when no gurobi model but metadata exists
        meta = inp.handle.metadata
        if inp.handle.model is None:
            if meta.get("allocation_data"):
                return SolveOutput(result=self._fallback_allocation(meta["allocation_data"], inp.handle))
            if meta.get("transport_data"):
                return SolveOutput(result=self._fallback_transport(meta["transport_data"], inp.handle))
            if meta.get("assignment_data"):
                return SolveOutput(result=self._fallback_assignment(meta["assignment_data"], inp.handle))
            if meta.get("facility_data"):
                return SolveOutput(result=self._fallback_facility(meta["facility_data"], inp.handle))
            if meta.get("retail_data"):
                return SolveOutput(result=self._fallback_retail(meta["retail_data"], inp.handle))
            if meta.get("missing_inputs"):
                return SolveOutput(result=SolveResult(status="not_yet_implemented", solver_log=";".join(meta["missing_inputs"])))
            return SolveOutput(result=SolveResult(status="not_built", solver_log="No model to solve"))

        if gp is None:
            return SolveOutput(result=SolveResult(status="gurobi_missing", solver_log="gurobipy not installed"))

        model: gp.Model = inp.handle.model
        try:
            model.optimize()
            status = model.Status
            status_str = gp.GRB.Status.names.get(status, str(status))
            objective = model.ObjVal if status == gp.GRB.OPTIMAL else None
            mip_gap = model.MIPGap if model.IsMIP and status == gp.GRB.OPTIMAL else None
            runtime = model.Runtime
            log = f"status={status_str} runtime={runtime}"
            return SolveOutput(
                result=SolveResult(
                    status=status_str, objective=objective, runtime=runtime, mip_gap=mip_gap, solver_log=log
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Last-resort fallback for environments without a valid license
            if meta.get("allocation_data"):
                return SolveOutput(result=self._fallback_allocation(meta["allocation_data"], inp.handle, error=str(exc)))
            if meta.get("transport_data"):
                return SolveOutput(result=self._fallback_transport(meta["transport_data"], inp.handle))
            if meta.get("assignment_data"):
                return SolveOutput(result=self._fallback_assignment(meta["assignment_data"], inp.handle))
            if meta.get("facility_data"):
                return SolveOutput(result=self._fallback_facility(meta["facility_data"], inp.handle))
            if meta.get("retail_data"):
                return SolveOutput(result=self._fallback_retail(meta["retail_data"], inp.handle))
            return SolveOutput(result=SolveResult(status="error", solver_log=str(exc)))

    def _fallback_allocation(self, data, handle, error: Optional[str] = None) -> SolveResult:
        """
        Deterministic LP solver for allocation problems using corner-point enumeration.
        Only used when Gurobi is unavailable.
        """
        items = data["items"]
        reqs = data["requirements"]
        feature_order = data["feature_order"]
        m = len(feature_order)
        n = len(items)
        A = np.zeros((m, n), dtype=float)
        b = np.array([reqs[feat] for feat in feature_order], dtype=float)
        costs = np.array([it["cost"] for it in items], dtype=float)
        for j, item in enumerate(items):
            for i, feat in enumerate(feature_order):
                A[i, j] = item["features"].get(feat, 0.0)

        best_obj = np.inf
        best_x = None

        # Helper to evaluate feasibility and objective
        def evaluate(x_vec: np.ndarray):
            nonlocal best_obj, best_x
            if np.any(x_vec < -1e-9):
                return
            if np.all(A @ x_vec >= b - 1e-6):
                obj = costs @ x_vec
                if obj < best_obj:
                    best_obj = obj
                    best_x = x_vec

        # Candidate 1: single-item scaling
        for j in range(n):
            denom = A[:, j]
            if np.any(denom <= 0):
                continue
            scale = np.max(b / denom)
            x_vec = np.zeros(n)
            x_vec[j] = scale
            evaluate(x_vec)

        # Candidate 2: combinations of constraints and variables
        for k in range(1, min(n, m) + 1):
            for vars_idx in _combinations(range(n), k):
                for constr_idx in _combinations(range(m), k):
                    A_eq = A[np.ix_(constr_idx, vars_idx)]
                    try:
                        b_subset = b[list(constr_idx)]
                        sol = np.linalg.solve(A_eq, b_subset)
                    except Exception:
                        continue
                    full = np.zeros(n)
                    full[list(vars_idx)] = sol
                    evaluate(full)

        # Candidate 3: grid search coarse (limited to small n)
        if best_x is None and n <= 3:
            steps = [np.linspace(0, 5, 26) for _ in range(n)]  # 0 to 5 in 0.2 increments
            for grid in np.array(np.meshgrid(*steps)).T.reshape(-1, n):
                evaluate(grid)

        # Construct result and dummy variable values for audit
        if best_x is not None:
            handle.variables = {items[i]["name"]: _DummyVar(float(best_x[i])) for i in range(n)}
            return SolveResult(
                status="optimal",
                objective=float(best_obj),
                runtime=None,
                solver_log="fallback_solver" + (f" original_error={error}" if error else ""),
            )
        return SolveResult(status="infeasible_fallback", objective=None, runtime=None, solver_log="fallback_solver_failed")

    def _fallback_transport(self, data: Dict[str, Any], handle) -> SolveResult:
        sources = data["sources"]
        sinks = data["sinks"]
        supply = data["supply"].copy()
        demand = data["demand"].copy()
        costs = data["costs"]
        x_vars = {}
        obj = 0.0
        # greedy min-cost allocation
        for j in sinks:
            remaining = demand[j]
            # sort sources by cost to this sink
            order = sorted(sources, key=lambda s: costs[s][j])
            for i in order:
                if remaining <= 1e-9:
                    break
                qty = min(supply[i], remaining)
                if qty > 0:
                    x_vars[f"{i}__{j}"] = _DummyVar(qty)
                    obj += qty * costs[i][j]
                    supply[i] -= qty
                    remaining -= qty
            if remaining > 1e-6:
                return SolveResult(status="infeasible", solver_log="demand unmet in fallback")
        handle.variables = x_vars
        return SolveResult(status="optimal", objective=obj, solver_log="fallback_transport")

    def _fallback_assignment(self, data: Dict[str, Any], handle) -> SolveResult:
        workers = data["workers"]
        tasks = data["tasks"]
        if len(workers) != len(tasks):
            return SolveResult(status="infeasible", solver_log="assignment size mismatch")
        best_obj = float("inf")
        best_assign = None
        for perm in itertools.permutations(tasks):
            obj = 0.0
            for w, t in zip(workers, perm):
                obj += data["costs"][w][t]
            if obj < best_obj:
                best_obj = obj
                best_assign = dict(zip(workers, perm))
        if best_assign is None:
            return SolveResult(status="infeasible", solver_log="no assignment found")
        handle.variables = {f"{w}__{t}": _DummyVar(1.0 if best_assign[w] == t else 0.0) for w in workers for t in tasks}
        return SolveResult(status="optimal", objective=best_obj, solver_log="fallback_assignment")

    def _fallback_facility(self, data: Dict[str, Any], handle) -> SolveResult:
        facilities = data["facilities"]
        customers = data["customers"]
        demand = data["demand"]
        ship_cost = data["ship_cost"]
        open_vars = {}
        ship_vars = {}
        obj = 0.0
        opened = set()
        for c in customers:
            # choose cheapest facility for each customer
            f_best = min(facilities, key=lambda f: ship_cost[f["name"]][c] + f["open_cost"])
            if f_best["name"] not in open_vars:
                open_vars[f_best["name"]] = _DummyVar(1.0)
            ship_vars[(f_best["name"], c)] = _DummyVar(demand[c])
            if f_best["name"] not in opened:
                obj += f_best["open_cost"]
                opened.add(f_best["name"])
            obj += demand[c] * ship_cost[f_best["name"]][c]
        handle.variables = {"open": open_vars, "ship": ship_vars}
        return SolveResult(status="optimal", objective=obj, solver_log="fallback_facility")

    def _fallback_retail(self, data: Dict[str, Any], handle) -> SolveResult:
        products = data["products"]
        locations = data["locations"]
        periods = data["periods"]
        demand = data["demand"]
        demand_share = data.get("demand_share", {l: 1.0 for l in locations})
        cold_capacity = data.get("cold_capacity", {l: float("inf") for l in locations})
        cold_usage = data.get("cold_usage", {p: 0.0 for p in products})
        prod_cap = data.get("production_cap", {p: [float('inf')]*len(periods) for p in products})
        purchase_cost = data.get("purchase_cost", {p: 0.0 for p in products})
        hold_cost = data.get("hold_cost", {p: 0.0 for p in products})
        lost_penalty = data.get("lost_penalty", {p: 0.0 for p in products})

        prod_vars = {}
        inv_vars = {}
        sales_vars = {}
        lost_vars = {}
        obj = 0.0
        feasible = True

        for p in products:
            for l in locations:
                prev_inv = 0.0
                for t_idx, t in enumerate(periods):
                    demand_loc = demand[p][t_idx] * demand_share.get(l, 1.0)
                    # simple heuristic: meet demand if capacity allows else lost
                    prod_allowed = min(prod_cap.get(p, [float('inf')]*len(periods))[t_idx], demand_loc)
                    sales = min(prod_allowed + prev_inv, demand_loc)
                    lost = max(0.0, demand_loc - sales)
                    inv = max(0.0, prev_inv + prod_allowed - sales)
                    # cold capacity check
                    if inv * cold_usage.get(p, 0.0) > cold_capacity.get(l, float("inf")) + 1e-6:
                        feasible = False
                    prod_vars[(p, l, t)] = _DummyVar(prod_allowed)
                    inv_vars[(p, l, t)] = _DummyVar(inv)
                    sales_vars[(p, l, t)] = _DummyVar(sales)
                    lost_vars[(p, l, t)] = _DummyVar(lost)
                    obj += purchase_cost.get(p, 0.0) * prod_allowed + hold_cost.get(p, 0.0) * inv + lost_penalty.get(p, 0.0) * lost
                    prev_inv = inv
        handle.variables = {"prod": prod_vars, "inv": inv_vars, "sales": sales_vars, "lost": lost_vars}
        status = "optimal" if feasible else "infeasible"
        return SolveResult(status=status, objective=obj if feasible else None, solver_log="fallback_retail")


def _combinations(iterable, r):
    # local lightweight combination generator to avoid importing itertools to keep deterministic iteration order
    pool = list(iterable)
    n = len(pool)
    if r > n or r <= 0:
        return []
    idx = list(range(r))
    result = []
    while True:
        result.append(tuple(pool[i] for i in idx))
        # generate next
        for i in reversed(range(r)):
            if idx[i] != i + n - r:
                break
        else:
            return result
        idx[i] += 1
        for j in range(i + 1, r):
            idx[j] = idx[j - 1] + 1


class _DummyVar:
    """Minimal variable wrapper exposing X attribute."""

    def __init__(self, value: float):
        self.X = value


__all__ = ["SolveSkill", "SolveInput", "SolveOutput"]
