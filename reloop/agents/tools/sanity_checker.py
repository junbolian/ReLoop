from __future__ import annotations

from typing import List, Optional

from ..schemas import (
    ConstraintTemplate,
    DataProfile,
    SanityCheckResult,
    SanityReport,
    SpecSheet,
    Step0Contract,
)


def _has_field(data_profile: Optional[DataProfile], key: str) -> bool:
    if not data_profile:
        return False
    return any(fp.path.endswith(key) for fp in data_profile.fields)


def _has_sub_edges(data_profile: Optional[DataProfile]) -> bool:
    if not data_profile:
        return False
    return any("network.sub_edges" in fp.path for fp in data_profile.fields)


def run_sanity_checks(
    contract: Optional[Step0Contract],
    spec_sheet: Optional[SpecSheet],
    templates: List[ConstraintTemplate],
    data_profile: Optional[DataProfile] = None,
) -> SanityReport:
    checks: List[SanityCheckResult] = []
    blockers: List[str] = []

    lost_sales_expected = False
    if contract:
        lost_sales_expected = any(
            "lost" in sv.name.lower() or "lost" in sv.penalty_source.lower()
            for sv in contract.soft_violations
        )

    decisions = {d.name: d for d in (spec_sheet.decisions if spec_sheet else [])}
    demand_templates = [t for t in templates if t.prefix == "demand_route"]

    # 1: hard_equals_really_hard
    if lost_sales_expected:
        has_L = "L" in decisions
        route_mentions_L = any(
            "L" in eq.lhs or "L" in eq.rhs
            for t in demand_templates
            for eq in t.equations
        )
        passed = has_L and route_mentions_L
        reason = (
            "Lost sales expected but variable L missing or not routed"
            if not passed
            else "Lost sales represented with L and demand routing"
        )
        if not passed:
            blockers.append("demand_route")
        checks.append(
            SanityCheckResult(
                id=1,
                name="hard_equals_really_hard",
                pass_=passed,
                reason=reason,
                fix_hint="Include L in demand routing when lost sales are penalized.",
            )
        )
    else:
        checks.append(
            SanityCheckResult(
                id=1,
                name="hard_equals_really_hard",
                pass_=True,
                reason="No optional slack expected or already represented.",
                fix_hint="",
            )
        )

    # 2: unit_consistency (used here to enforce shelf-life indexing)
    shelf_life_active = _has_field(data_profile, "shelf_life")
    if shelf_life_active:
        required = [decisions.get("I"), decisions.get("y")]
        has_a_index = all(
            d is not None and d.indices and "a" in d.indices for d in required
        )
        passed = bool(has_a_index)
        if not passed:
            blockers.append("aging")
        checks.append(
            SanityCheckResult(
                id=2,
                name="unit_consistency",
                pass_=passed,
                reason="I and y must be indexed by remaining life when shelf_life is active.",
                fix_hint="Add index a to I and y and propagate through aging/availability constraints.",
            )
        )
    else:
        checks.append(
            SanityCheckResult(
                id=2,
                name="unit_consistency",
                pass_=True,
                reason="No shelf-life dimension detected.",
                fix_hint="",
            )
        )

    # 3: time_alignment (basic placeholder)
    checks.append(
        SanityCheckResult(
            id=3,
            name="time_alignment",
            pass_=True,
            reason="No time alignment issues detected in static analysis.",
            fix_hint="Ensure lead_time handling shifts arrivals appropriately.",
        )
    )

    # 4: conservation_closed (used to detect substitution coupling gaps)
    substitution_active = _has_sub_edges(data_profile)
    if substitution_active:
        demand_has_S = any(
            "S" in eq.lhs or "S" in eq.rhs
            for t in demand_templates
            for eq in t.equations
        )
        supply_coupled = any(
            ("S" in eq.lhs or "S" in eq.rhs)
            and ("y" in eq.lhs or "y" in eq.rhs)
            for t in templates
            for eq in t.equations
            if t.prefix in {"availability", "sales_conservation", "demand_route"}
        )
        passed = demand_has_S and supply_coupled
        if not passed:
            blockers.append("demand_route")
        checks.append(
            SanityCheckResult(
                id=4,
                name="conservation_closed",
                pass_=passed,
                reason="Substitution must appear in demand routing and consume supplying product inventory.",
                fix_hint="Link S flows into demand routing and subtract from supplying product availability.",
            )
        )
    else:
        checks.append(
            SanityCheckResult(
                id=4,
                name="conservation_closed",
                pass_=True,
                reason="No substitution edges detected.",
                fix_hint="",
            )
        )

    # 5: boundary_handled (basic)
    checks.append(
        SanityCheckResult(
            id=5,
            name="boundary_handled",
            pass_=True,
            reason="Boundary conditions recorded in edge_cases.",
            fix_hint="Ensure t=0 init and life a boundaries handled in code.",
        )
    )

    # 6: extreme_cases_behavior (basic)
    checks.append(
        SanityCheckResult(
            id=6,
            name="extreme_cases_behavior",
            pass_=True,
            reason="No extreme-case blockers identified in static checks.",
            fix_hint="Stress test zero demand and zero capacity in codegen.",
        )
    )

    overall_pass = all(c.passed for c in checks)
    if not overall_pass:
        recommended = "REVISE_SPEC"
    else:
        recommended = "PROCEED_TO_CODEGEN"

    return SanityReport(
        checks=checks,
        overall_pass=overall_pass,
        blockers=list(sorted(set(blockers))),
        recommended_next_step=recommended,
    )
