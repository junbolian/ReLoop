from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

try:
    import gurobipy as gp
except ImportError:  # pragma: no cover
    gp = None


class IRVariable(BaseModel):
    name: str
    domain: str = "continuous"  # continuous|binary
    indices: List[str] = Field(default_factory=list)
    lb: float = 0.0
    ub: Optional[float] = None


class IRConstraint(BaseModel):
    name: str
    expr: List[Any]  # [["coefficient", "term"], ...] generic linear list
    sense: str  # <=, >=, ==
    rhs: float


class IRObjective(BaseModel):
    sense: str = "minimize"
    expr: List[Any] = Field(default_factory=list)


class IRPlan(BaseModel):
    sets: Dict[str, List[str]] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    variables: List[IRVariable] = Field(default_factory=list)
    constraints: List[IRConstraint] = Field(default_factory=list)
    objective: IRObjective = Field(default_factory=IRObjective)
    missing_fields: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


def compile_ir_to_gurobi(ir: IRPlan):
    if gp is None:
        return None, {"solver": "missing_gurobi"}, {}
    model = gp.Model("ir_model")
    model.Params.OutputFlag = 0
    var_map = {}
    # currently only support scalar variables (ignore indices metadata)
    for var in ir.variables:
        vtype = gp.GRB.CONTINUOUS if var.domain == "continuous" else gp.GRB.BINARY
        var_map[var.name] = model.addVar(lb=var.lb, ub=var.ub, vtype=vtype, name=var.name)
    model.update()
    # objective
    obj_expr = 0.0
    for term in ir.objective.expr:
        if isinstance(term, list) and len(term) == 2:
            coeff, name = term
            obj_expr += float(coeff) * var_map.get(name, 0.0)
    sense = gp.GRB.MINIMIZE if ir.objective.sense.lower().startswith("min") else gp.GRB.MAXIMIZE
    model.setObjective(obj_expr, sense)
    # constraints
    for c in ir.constraints:
        lhs = 0.0
        for term in c.expr:
            if isinstance(term, list) and len(term) == 2:
                coeff, name = term
                lhs += float(coeff) * var_map.get(name, 0.0)
        if c.sense == ">=":
            model.addConstr(lhs >= c.rhs, name=c.name)
        elif c.sense == "<=":
            model.addConstr(lhs <= c.rhs, name=c.name)
        else:
            model.addConstr(lhs == c.rhs, name=c.name)
    model.update()
    return model, {"variables": list(var_map.keys()), "constraints": [c.name for c in ir.constraints]}, var_map
