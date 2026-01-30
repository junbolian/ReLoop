from __future__ import annotations

from typing import Dict, Any

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover - allow fallback when gurobi missing
    gp = None
from pydantic import BaseModel

from ..state import ModelHandle
from ..model_ir import IRPlan, compile_ir_to_gurobi


class ModelBuilderInput(BaseModel):
    archetype_id: str
    canonical_schema: Any
    model_plan: Any = None


class ModelBuilderOutput(BaseModel):
    handle: ModelHandle
    model_plan: Dict


class ModelBuilderSkill:
    """
    Purpose:
        Translate canonical schemas into concrete Gurobi models.
    Determinism:
        Fully deterministic; no randomness.
    Failure modes:
        Infeasible model definitions raise and bubble up.
    Logging:
        Returns a structured model_plan summarizing variables and constraints.
    """

    def run(self, inp: ModelBuilderInput) -> ModelBuilderOutput:
        arch = inp.archetype_id
        if arch == "diet_lp":
            arch = "allocation_lp"
        # Prefer IR compilation when provided
        if inp.model_plan:
            try:
                ir = IRPlan.model_validate(inp.model_plan)
                model, meta, var_map = compile_ir_to_gurobi(ir)
                handle = ModelHandle(archetype_id=arch, model=model, variables=var_map, metadata={"ir_meta": meta})
                return ModelBuilderOutput(handle=handle, model_plan=inp.model_plan)
            except Exception:
                pass
        if arch == "allocation_lp":
            return self._build_allocation(inp.canonical_schema)
        if arch == "transport_flow":
            return self._build_transport(inp.canonical_schema)
        if arch == "assignment":
            return self._build_assignment(inp.canonical_schema)
        if arch == "facility_location":
            return self._build_facility(inp.canonical_schema)
        if arch == "retail_inventory":
            return self._build_retail(inp.canonical_schema)
        return self._build_placeholder(arch)

    def _build_allocation(self, schema) -> ModelBuilderOutput:
        alloc_data = {
            "items": [{"name": item.name, "cost": item.cost, "features": item.features} for item in schema.items],
            "requirements": schema.requirements,
            "feature_order": schema.feature_order,
        }
        if gp is None:
            handle = ModelHandle(archetype_id="allocation_lp", model=None, variables={}, metadata={"allocation_data": alloc_data})
            plan = {
                "variables": [item.name for item in schema.items],
                "constraints": [f"req_{n}" for n in schema.feature_order],
                "objective": "minimize cost (fallback)",
                "solver": "fallback",
            }
            return ModelBuilderOutput(handle=handle, model_plan=plan)

        model = gp.Model("allocation_lp")
        model.Params.OutputFlag = 0
        x = {}
        for item in schema.items:
            x[item.name] = model.addVar(lb=0.0, name=f"x_{item.name}")
        model.update()
        # Objective
        model.setObjective(gp.quicksum(item.cost * x[item.name] for item in schema.items), gp.GRB.MINIMIZE)
        # Feature constraints
        for feat in schema.feature_order:
            lhs = gp.quicksum(item.features.get(feat, 0.0) * x[item.name] for item in schema.items)
            model.addConstr(lhs >= schema.requirements.get(feat, 0.0), name=f"req_{feat}")
        model.update()
        handle = ModelHandle(
            archetype_id="allocation_lp",
            model=model,
            variables=x,
            metadata={"num_items": len(x), "allocation_data": alloc_data},
        )
        model_plan = {
            "variables": list(x.keys()),
            "constraints": [f"req_{n}" for n in schema.feature_order],
            "objective": "minimize cost",
        }
        return ModelBuilderOutput(handle=handle, model_plan=model_plan)

    def _build_transport(self, schema: Dict[str, Any]) -> ModelBuilderOutput:
        data = schema
        if gp is None:
            handle = ModelHandle(archetype_id="transport_flow", model=None, metadata={"transport_data": data})
            return ModelBuilderOutput(handle=handle, model_plan={"objective": "min cost flow", "solver": "fallback"})

        model = gp.Model("transport_flow")
        model.Params.OutputFlag = 0
        x = {}
        for i in data["sources"]:
            for j in data["sinks"]:
                key = f"{i}__{j}"
                x[key] = model.addVar(lb=0.0, name=f"x_{i}_{j}")
        model.update()
        model.setObjective(
            gp.quicksum(data["costs"][i][j] * x[f"{i}__{j}"] for i in data["sources"] for j in data["sinks"]),
            gp.GRB.MINIMIZE,
        )
        for i in data["sources"]:
            model.addConstr(gp.quicksum(x[f"{i}__{j}"] for j in data["sinks"]) <= data["supply"][i], name=f"supply_{i}")
        for j in data["sinks"]:
            model.addConstr(gp.quicksum(x[f"{i}__{j}"] for i in data["sources"]) >= data["demand"][j], name=f"demand_{j}")
        model.update()
        handle = ModelHandle(archetype_id="transport_flow", model=model, variables=x, metadata={"transport_data": data})
        plan = {"variables": list(x.keys()), "objective": "min cost", "constraints": "supply<=, demand>="}
        return ModelBuilderOutput(handle=handle, model_plan=plan)

    def _build_assignment(self, schema: Dict[str, Any]) -> ModelBuilderOutput:
        data = schema
        if gp is None:
            handle = ModelHandle(archetype_id="assignment", model=None, metadata={"assignment_data": data})
            return ModelBuilderOutput(handle=handle, model_plan={"objective": "min assignment cost", "solver": "fallback"})

        model = gp.Model("assignment")
        model.Params.OutputFlag = 0
        y = {}
        for w in data["workers"]:
            for t in data["tasks"]:
                key = f"{w}__{t}"
                y[key] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{w}_{t}")
        model.update()
        model.setObjective(
            gp.quicksum(data["costs"][w][t] * y[f"{w}__{t}"] for w in data["workers"] for t in data["tasks"]), gp.GRB.MINIMIZE
        )
        for w in data["workers"]:
            model.addConstr(gp.quicksum(y[f"{w}__{t}"] for t in data["tasks"]) == 1, name=f"assign_worker_{w}")
        for t in data["tasks"]:
            model.addConstr(gp.quicksum(y[f"{w}__{t}"] for w in data["workers"]) == 1, name=f"assign_task_{t}")
        model.update()
        handle = ModelHandle(archetype_id="assignment", model=model, variables=y, metadata={"assignment_data": data})
        return ModelBuilderOutput(handle=handle, model_plan={"objective": "min cost", "constraints": "bipartite assignment"})

    def _build_facility(self, schema: Dict[str, Any]) -> ModelBuilderOutput:
        data = schema
        if gp is None:
            handle = ModelHandle(archetype_id="facility_location", model=None, metadata={"facility_data": data})
            return ModelBuilderOutput(handle=handle, model_plan={"objective": "min cost facility", "solver": "fallback"})

        model = gp.Model("facility_location")
        model.Params.OutputFlag = 0
        open_vars = {}
        ship_vars = {}
        for f in data["facilities"]:
            open_vars[f["name"]] = model.addVar(vtype=gp.GRB.BINARY, name=f"open_{f['name']}")
            for c in data["customers"]:
                ship_vars[(f["name"], c)] = model.addVar(lb=0.0, name=f"ship_{f['name']}_{c}")
        model.update()
        model.setObjective(
            gp.quicksum(f["open_cost"] * open_vars[f["name"]] for f in data["facilities"])
            + gp.quicksum(
                data["ship_cost"][f["name"]][c] * ship_vars[(f["name"], c)] for f in data["facilities"] for c in data["customers"]
            ),
            gp.GRB.MINIMIZE,
        )
        for c in data["customers"]:
            model.addConstr(gp.quicksum(ship_vars[(f["name"], c)] for f in data["facilities"]) == data["demand"][c], name=f"demand_{c}")
        for f in data["facilities"]:
            for c in data["customers"]:
                model.addConstr(ship_vars[(f["name"], c)] <= open_vars[f["name"]], name=f"link_{f['name']}_{c}")
            # optional capacity if finite
            if f["capacity"] != float("inf"):
                model.addConstr(
                    gp.quicksum(ship_vars[(f["name"], c)] for c in data["customers"]) <= f["capacity"],
                    name=f"cap_{f['name']}",
                )
        model.update()
        handle = ModelHandle(
            archetype_id="facility_location",
            model=model,
            variables={"open": open_vars, "ship": ship_vars},
            metadata={"facility_data": data},
        )
        return ModelBuilderOutput(handle=handle, model_plan={"objective": "min open+ship", "constraints": "demand eq, linking"})

    def _build_retail(self, schema) -> ModelBuilderOutput:
        data = getattr(schema, "data", None) if schema else None
        if data is None:
            handle = ModelHandle(archetype_id="retail_inventory", model=None, metadata={"missing_inputs": ["data_dict"]})
            return ModelBuilderOutput(
                handle=handle,
                model_plan={"status": "NotYetImplemented", "missing_inputs": ["data_dict"], "coverage_level": getattr(schema, "coverage_level", "baseline_no_aging")},
            )
        if gp is None:
            handle = ModelHandle(archetype_id="retail_inventory", model=None, metadata={"retail_data": data})
            return ModelBuilderOutput(handle=handle, model_plan={"objective": "min retail cost", "solver": "fallback"})

        model = gp.Model("retail_inventory")
        model.Params.OutputFlag = 0
        prod = {}
        inv = {}
        sales = {}
        lost = {}
        products = data["products"]
        locations = data["locations"]
        periods = data["periods"]
        demand = data["demand"]
        demand_share = data.get("demand_share", {loc: 1.0 for loc in locations})
        cold_capacity = data.get("cold_capacity", {loc: float("inf") for loc in locations})
        cold_usage = data.get("cold_usage", {p: 0.0 for p in products})
        prod_cap = data.get("production_cap", {p: [float('inf')]*len(periods) for p in products})
        purchase_cost = data.get("purchase_cost", {p: 0.0 for p in products})
        hold_cost = data.get("hold_cost", {p: 0.0 for p in products})
        lost_penalty = data.get("lost_penalty", {p: 0.0 for p in products})

        for p in products:
            for l in locations:
                for t_idx, t in enumerate(periods):
                    prod[(p, l, t)] = model.addVar(lb=0.0, name=f"prod_{p}_{l}_{t}")
                    inv[(p, l, t)] = model.addVar(lb=0.0, name=f"inv_{p}_{l}_{t}")
                    sales[(p, l, t)] = model.addVar(lb=0.0, name=f"sales_{p}_{l}_{t}")
                    lost[(p, l, t)] = model.addVar(lb=0.0, name=f"lost_{p}_{l}_{t}")
        model.update()

        # Objective
        model.setObjective(
            gp.quicksum(purchase_cost.get(p, 0.0) * prod[(p, l, t)] for p in products for l in locations for t in periods)
            + gp.quicksum(hold_cost.get(p, 0.0) * inv[(p, l, t)] for p in products for l in locations for t in periods)
            + gp.quicksum(lost_penalty.get(p, 0.0) * lost[(p, l, t)] for p in products for l in locations for t in periods),
            gp.GRB.MINIMIZE,
        )

        # Constraints
        for p in products:
            for l in locations:
                for t_idx, t in enumerate(periods):
                    demand_loc = demand[p][t_idx] * demand_share.get(l, 1.0)
                    model.addConstr(sales[(p, l, t)] + lost[(p, l, t)] == demand_loc, name=f"demand_{p}_{l}_{t}")
                    prev_inv = inv[(p, l, periods[t_idx - 1])] if t_idx > 0 else 0.0
                    model.addConstr(inv[(p, l, t)] == prev_inv + prod[(p, l, t)] - sales[(p, l, t)], name=f"balance_{p}_{l}_{t}")
                # cold capacity per period
                for t_idx, t in enumerate(periods):
                    model.addConstr(
                        gp.quicksum(cold_usage.get(p2, 0.0) * inv[(p2, l, t)] for p2 in products) <= cold_capacity.get(l, float("inf")),
                        name=f"cold_{l}_{t}",
                    )
            for t_idx, t in enumerate(periods):
                model.addConstr(
                    gp.quicksum(prod[(p, l, t)] for l in locations) <= prod_cap.get(p, [float("inf")]*len(periods))[t_idx],
                    name=f"prod_cap_{p}_{t}",
                )
        model.update()
        handle = ModelHandle(
            archetype_id="retail_inventory",
            model=model,
            variables={"prod": prod, "inv": inv, "sales": sales, "lost": lost},
            metadata={"retail_data": data},
        )
        return ModelBuilderOutput(handle=handle, model_plan={"objective": "min retail cost", "coverage": data.get("coverage_level", "baseline_no_aging")})

    def _build_placeholder(self, archetype_id: str) -> ModelBuilderOutput:
        plan = {
            "status": "NotImplemented",
            "todo": [
                "Define decision variables for inventory levels",
                "Add holding/backorder costs",
                "Add demand satisfaction constraints",
            ],
        }
        handle = ModelHandle(archetype_id=archetype_id, model=None, metadata={"not_implemented": True})
        return ModelBuilderOutput(handle=handle, model_plan=plan)


__all__ = ["ModelBuilderSkill", "ModelBuilderInput", "ModelBuilderOutput"]
