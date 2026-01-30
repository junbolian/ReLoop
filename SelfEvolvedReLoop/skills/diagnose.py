from __future__ import annotations

from typing import List

from pydantic import BaseModel

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover - fallback when gurobi missing
    gp = None

from ..state import Diagnosis, ModelHandle, SolveResult


class DiagnoseInput(BaseModel):
    handle: ModelHandle
    solve_result: SolveResult


class DiagnoseOutput(BaseModel):
    diagnosis: Diagnosis


class DiagnoseSkill:
    """
    Purpose:
        Explain solver outcomes and highlight bottlenecks.
    Determinism:
        Deterministic rules over solver artifacts.
    Failure modes:
        Gurobi IIS computation may fail; handled gracefully.
    Logging:
        Returns symptom_tags and any IIS constraint names.
    """

    def run(self, inp: DiagnoseInput) -> DiagnoseOutput:
        tags: List[str] = []
        summary = inp.solve_result.status
        recommended_params = {}
        iis_constraints: List[str] = []

        status = inp.solve_result.status.lower()
        model = inp.handle.model if gp is not None else None

        if "inf" in status:
            tags.append("SOLVER_INFEASIBLE")
        if "unbounded" in status:
            tags.append("SOLVER_UNBOUNDED")
        if "inf" in status and model is not None:
            try:
                model.computeIIS()
                for constr in model.getConstrs():
                    if getattr(constr, "IISConstr", 0) > 0:
                        iis_constraints.append(constr.ConstrName)
            except Exception as exc:  # noqa: BLE001
                summary += f" (IIS failed: {exc})"
        if inp.solve_result.runtime and inp.solve_result.runtime > 5.0:
            tags.append("SOLVER_SLOW")
            recommended_params["Method"] = 1  # dual simplex
        if inp.solve_result.mip_gap and inp.solve_result.mip_gap > 0.05:
            tags.append("HIGH_GAP")
            recommended_params["MIPGap"] = 0.01

        diagnosis = Diagnosis(
            summary=summary, symptom_tags=tags, recommended_params=recommended_params, iis_constraints=iis_constraints
        )
        return DiagnoseOutput(diagnosis=diagnosis)


__all__ = ["DiagnoseSkill", "DiagnoseInput", "DiagnoseOutput"]
