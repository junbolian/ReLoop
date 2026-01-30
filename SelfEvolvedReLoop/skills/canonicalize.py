from __future__ import annotations

from typing import Dict, List, Any

from pydantic import BaseModel

from ..state import (
    AllocationSchema,
    AllocationItem,
    CanonicalAllocationSchema,
    TransportFlowSchema,
    AssignmentSchema,
    FacilityLocationSchema,
    RetailInventorySchema,
)


class CanonicalizeInput(BaseModel):
    archetype_id: str
    schema: Any


class CanonicalizeOutput(BaseModel):
    canonical: Any


class CanonicalizationSkill:
    """
    Purpose:
        Normalize extracted schemas to deterministic shapes per archetype.
    Determinism:
        Purely rule-based; no stochastic steps.
    Failure modes:
        Negative costs or requirements are clipped to zero with warnings.
    Logging:
        Returns warnings embedded in canonical objects.
    """

    def run(self, inp: CanonicalizeInput) -> CanonicalizeOutput:
        archetype = inp.archetype_id
        if archetype == "diet_lp":
            archetype = "allocation_lp"

        if archetype == "allocation_lp":
            return CanonicalizeOutput(canonical=self._canon_allocation(inp.schema))
        if archetype == "transport_flow":
            return CanonicalizeOutput(canonical=self._canon_transport(inp.schema))
        if archetype == "assignment":
            return CanonicalizeOutput(canonical=self._canon_assignment(inp.schema))
        if archetype == "facility_location":
            return CanonicalizeOutput(canonical=self._canon_facility(inp.schema))
        if archetype == "retail_inventory":
            return CanonicalizeOutput(canonical=inp.schema)
        return CanonicalizeOutput(canonical=None)

    # --- per archetype canonicalizers ---
    def _canon_allocation(self, schema: AllocationSchema) -> CanonicalAllocationSchema:
        warnings: List[str] = []

        feats = set(schema.requirements.keys())
        for item in schema.items:
            feats.update(item.features.keys())
        feature_order = sorted(feats)

        items: List[AllocationItem] = []
        for item in schema.items:
            norm_name = item.name.strip()
            cost = max(0.0, float(item.cost))
            if cost != item.cost:
                warnings.append(f"Cost for {norm_name} clipped to non-negative.")
            norm_features: Dict[str, float] = {}
            for n in feature_order:
                val = float(item.features.get(n, 0.0))
                if val < 0:
                    warnings.append(f"Feature {n} for {norm_name} clipped to zero.")
                    val = 0.0
                norm_features[n] = val
            items.append(AllocationItem(name=norm_name, cost=cost, features=norm_features))

        reqs = {}
        for n in feature_order:
            val = float(schema.requirements.get(n, 0.0))
            if val < 0:
                warnings.append(f"Requirement {n} clipped to zero.")
                val = 0.0
            reqs[n] = val

        canonical = CanonicalAllocationSchema(
            items=items, requirements=reqs, feature_order=feature_order, warnings=warnings
        )
        return canonical

    def _canon_transport(self, schema: TransportFlowSchema) -> Dict[str, Any]:
        warnings: List[str] = []
        sources = [s["name"] for s in schema.sources]
        sinks = [d["name"] for d in schema.sinks]
        supply = {s["name"]: max(0.0, float(s.get("supply", 0.0))) for s in schema.sources}
        demand = {d["name"]: max(0.0, float(d.get("demand", 0.0))) for d in schema.sinks}
        costs: Dict[str, Dict[str, float]] = {}
        for s in sources:
            row = {}
            for t in sinks:
                row[t] = float(schema.costs.get(s, {}).get(t, 0.0))
            costs[s] = row
        return {
            "sources": sources,
            "sinks": sinks,
            "supply": supply,
            "demand": demand,
            "costs": costs,
            "warnings": warnings,
        }

    def _canon_assignment(self, schema: AssignmentSchema) -> Dict[str, Any]:
        warnings: List[str] = []
        workers = list(dict.fromkeys(schema.workers))
        tasks = list(dict.fromkeys(schema.tasks))
        costs: Dict[str, Dict[str, float]] = {}
        for w in workers:
            row = {}
            for t in tasks:
                row[t] = float(schema.costs.get(w, {}).get(t, 0.0))
            costs[w] = row
        return {"workers": workers, "tasks": tasks, "costs": costs, "warnings": warnings}

    def _canon_facility(self, schema: FacilityLocationSchema) -> Dict[str, Any]:
        warnings: List[str] = []
        facilities = []
        for f in schema.facilities:
            name = f.get("name", "").strip()
            facilities.append(
                {
                    "name": name,
                    "open_cost": float(f.get("open_cost", 0.0)),
                    "capacity": float(f.get("capacity", float("inf"))),
                }
            )
        customers = list(dict.fromkeys(schema.customers))
        demand = {c: float(schema.demand.get(c, 1.0)) for c in customers}
        ship_cost: Dict[str, Dict[str, float]] = {}
        for f in facilities:
            row = {}
            for c in customers:
                row[c] = float(schema.ship_cost.get(f["name"], {}).get(c, 0.0))
            ship_cost[f["name"]] = row
        return {
            "facilities": facilities,
            "customers": customers,
            "demand": demand,
            "ship_cost": ship_cost,
            "warnings": warnings,
        }


__all__ = ["CanonicalizationSkill", "CanonicalizeInput", "CanonicalizeOutput"]
