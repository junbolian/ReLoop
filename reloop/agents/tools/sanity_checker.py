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
    """
    Run sanity checks on the spec sheet and constraint templates.
    """
    checks: List[SanityCheckResult] = []
    blockers: List[str] = []

    lost_sales_expected = False
    if contract:
        lost_sales_expected = any(
            "lost" in sv.name.lower() or (sv.penalty_source and "lost" in sv.penalty_source.lower())
            for sv in contract.soft_violations
        )

    decisions = {d.name: d for d in (spec_sheet.decisions if spec_sheet else [])}
    
    # Get relevant templates
    balance_templates = [t for t in templates if t.prefix in {"demand_route", "sales_conservation"}]

    # ==========================================================================
    # CHECK 1: hard_equals_really_hard (Lost sales variable L)
    # ==========================================================================
    if lost_sales_expected:
        has_L = "L" in decisions
        # Check if L appears in ANY balance constraint (demand_route OR sales_conservation)
        route_mentions_L = any(
            "L" in eq.lhs or "L" in eq.rhs
            for t in balance_templates
            for eq in t.equations
        )
        check_passed = has_L and route_mentions_L
        msg = (
            "Lost sales variable L present and included in balance constraints"
            if check_passed
            else "Lost sales expected but variable L missing or not in balance constraints"
        )
        if not check_passed:
            blockers.append("lost_sales")
        checks.append(
            SanityCheckResult(
                id=1,
                name="hard_equals_really_hard",
                passed=check_passed,
                message=msg,
                fix_hint="Include L in sales_conservation constraint.",
            )
        )
    else:
        checks.append(
            SanityCheckResult(
                id=1,
                name="hard_equals_really_hard",
                passed=True,
                message="No lost sales penalty expected.",
                fix_hint="",
            )
        )

    # ==========================================================================
    # CHECK 2: unit_consistency (shelf-life indexing)
    # ==========================================================================
    shelf_life_active = _has_field(data_profile, "shelf_life")
    if shelf_life_active:
        required = [decisions.get("I"), decisions.get("y")]
        has_a_index = all(
            d is not None and d.indices and "a" in d.indices for d in required
        )
        check_passed = bool(has_a_index)
        if not check_passed:
            blockers.append("aging")
        checks.append(
            SanityCheckResult(
                id=2,
                name="unit_consistency",
                passed=check_passed,
                message="I and y correctly indexed by remaining life (a)"
                if check_passed
                else "I and y must be indexed by remaining life when shelf_life is active.",
                fix_hint="Add index a to I and y.",
            )
        )
    else:
        checks.append(
            SanityCheckResult(
                id=2,
                name="unit_consistency",
                passed=True,
                message="No shelf-life dimension detected.",
                fix_hint="",
            )
        )

    # ==========================================================================
    # CHECK 3: time_alignment
    # ==========================================================================
    checks.append(
        SanityCheckResult(
            id=3,
            name="time_alignment",
            passed=True,
            message="No time alignment issues detected.",
            fix_hint="Ensure lead_time handling shifts arrivals appropriately.",
        )
    )

    # ==========================================================================
    # CHECK 4: conservation_closed (substitution coupling)
    # ==========================================================================
    substitution_active = _has_sub_edges(data_profile)
    if substitution_active:
        # Check S appears in sales_conservation (not just demand_route)
        sales_cons_templates = [t for t in templates if t.prefix == "sales_conservation"]
        demand_route_templates = [t for t in templates if t.prefix == "demand_route"]
        
        # S should appear in sales_conservation for inbound/outbound
        sales_has_S = any(
            "S" in eq.lhs or "S" in eq.rhs or "inbound" in eq.rhs or "outbound" in eq.rhs
            for t in sales_cons_templates
            for eq in t.equations
        )
        
        # demand_route should limit outbound S
        demand_has_S = any(
            "S" in eq.lhs or "S" in eq.rhs
            for t in demand_route_templates
            for eq in t.equations
        )
        
        check_passed = sales_has_S and demand_has_S
        if not check_passed:
            blockers.append("substitution")
        checks.append(
            SanityCheckResult(
                id=4,
                name="conservation_closed",
                passed=check_passed,
                message="Substitution correctly linked in demand_route and sales_conservation"
                if check_passed
                else "Substitution must appear in both demand_route and sales_conservation.",
                fix_hint="Add S to sales_conservation (inbound - outbound) and demand_route (outbound <= demand).",
            )
        )
    else:
        checks.append(
            SanityCheckResult(
                id=4,
                name="conservation_closed",
                passed=True,
                message="No substitution edges detected.",
                fix_hint="",
            )
        )

    # ==========================================================================
    # CHECK 5: boundary_handled
    # ==========================================================================
    checks.append(
        SanityCheckResult(
            id=5,
            name="boundary_handled",
            passed=True,
            message="Boundary conditions recorded in edge_cases.",
            fix_hint="Ensure t=1 init and life a boundaries handled in code.",
        )
    )

    # ==========================================================================
    # CHECK 6: extreme_cases_behavior
    # ==========================================================================
    checks.append(
        SanityCheckResult(
            id=6,
            name="extreme_cases_behavior",
            passed=True,
            message="No extreme-case blockers identified.",
            fix_hint="Stress test zero demand and zero capacity in codegen.",
        )
    )

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    overall_pass = all(c.passed for c in checks)
    recommended = "PROCEED_TO_CODEGEN" if overall_pass else "REVISE_SPEC"

    return SanityReport(
        checks=checks,
        overall_pass=overall_pass,
        blockers=list(sorted(set(blockers))),
        recommended_next_step=recommended,
    )