"""
ReLoop Code Executor

Functions:
- Syntax checking
- Subprocess isolated execution
- Output parsing
- IIS diagnostics for INFEASIBLE models
- Unbounded ray diagnostics for UNBOUNDED models
"""

import subprocess
import sys
import json
import base64
import re
from typing import Dict, Any, Tuple, Optional


# ---------------------------------------------------------------------------
# IIS / Unbounded diagnostics code executed INLINE after the LLM code finishes
# (or crashes). Runs while the Gurobi model is still alive in memory.
# Uses _reloop_ prefix on all names to avoid collisions with LLM code.
# ---------------------------------------------------------------------------
_IIS_DIAGNOSTICS_INLINE = r"""
# --- IIS / Unbounded diagnostics (runs while model is still alive) ---
_reloop_m = globals().get('m') or globals().get('model')
if _reloop_m is not None and hasattr(_reloop_m, 'Status'):
    try:
        import gurobipy as _reloop_grb

        # Disambiguate INF_OR_UNBD
        if _reloop_m.Status == _reloop_grb.GRB.INF_OR_UNBD:
            _reloop_m.Params.DualReductions = 0
            _reloop_m.Params.OutputFlag = 0
            _reloop_m.optimize()
            print(f"status: {_reloop_m.Status}")

        # --- INFEASIBLE: compute IIS ---
        if _reloop_m.Status == _reloop_grb.GRB.INFEASIBLE:
            try:
                _reloop_m.Params.TimeLimit = 10
                _reloop_m.computeIIS()
                _reloop_constrs = []
                for _reloop_c in _reloop_m.getConstrs():
                    if _reloop_c.IISConstr:
                        _reloop_constrs.append(f"{_reloop_c.ConstrName} ({_reloop_c.Sense} {_reloop_c.RHS})")
                _reloop_bounds = []
                for _reloop_v in _reloop_m.getVars():
                    if _reloop_v.IISLB:
                        _reloop_bounds.append(f"{_reloop_v.VarName} LB={_reloop_v.LB}")
                    if _reloop_v.IISUB:
                        _reloop_bounds.append(f"{_reloop_v.VarName} UB={_reloop_v.UB}")
                print(f"iis_n_constraints: {len(_reloop_constrs)}")
                print(f"iis_n_bounds: {len(_reloop_bounds)}")
                for _reloop_ic in _reloop_constrs[:20]:
                    print(f"iis_constr: {_reloop_ic}")
                for _reloop_ib in _reloop_bounds[:10]:
                    print(f"iis_bound: {_reloop_ib}")
            except Exception as _reloop_e:
                print(f"iis_error: {str(_reloop_e)[:200]}")

        # --- UNBOUNDED: report unbounded ray ---
        elif _reloop_m.Status == _reloop_grb.GRB.UNBOUNDED:
            try:
                # Re-solve with InfUnbdInfo to get unbounded rays
                _reloop_m.Params.InfUnbdInfo = 1
                _reloop_m.Params.OutputFlag = 0
                _reloop_m.optimize()
                _reloop_uvars = []
                for _reloop_v in _reloop_m.getVars():
                    try:
                        _reloop_ray = _reloop_v.UnbdRay
                        if abs(_reloop_ray) > 1e-6:
                            _reloop_uvars.append(f"{_reloop_v.VarName} (ray={_reloop_ray:.4f})")
                    except Exception:
                        pass
                if _reloop_uvars:
                    print(f"unbounded_n_vars: {len(_reloop_uvars)}")
                    for _reloop_uv in _reloop_uvars[:10]:
                        print(f"unbounded_var: {_reloop_uv}")
            except Exception:
                pass
    except Exception:
        pass
"""


class CodeExecutor:
    """Execute optimization code in isolated subprocess."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check Python syntax.

        Returns: (passed, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def execute(self, code: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute code in subprocess.

        The wrapper uses exec() + try/except so that IIS/unbounded diagnostics
        run INLINE while the Gurobi model is still alive in memory. This avoids
        the atexit approach where the model gets garbage-collected before the
        handler fires.

        Returns: {
            'exit_code': int,
            'status': str | None,
            'objective': float | None,
            'dual_objective': float | None,
            'solution': dict | None,
            'iis_constraints': list | None,
            'iis_bounds': list | None,
            'unbounded_vars': list | None,
            'stdout': str,
            'stderr': str,
            'error': str | None
        }
        """
        data_b64 = base64.b64encode(json.dumps(data).encode()).decode()
        code_b64 = base64.b64encode(code.encode()).decode()

        # Build wrapper: data setup → exec(LLM code) in try/except → diagnostics inline
        wrapper = (
            'import sys, json, base64, traceback\n'
            + f'data = json.loads(base64.b64decode("{data_b64}").decode())\n'
            + f'_reloop_code = base64.b64decode("{code_b64}").decode()\n'
            + '_reloop_err = None\n'
            + '_reloop_exit = None\n'
            + 'try:\n'
            + '    exec(_reloop_code, globals())\n'
            + 'except SystemExit as _se:\n'
            + '    _reloop_exit = _se.code\n'
            + 'except Exception as _e:\n'
            + '    _reloop_err = _e\n'
            + '    traceback.print_exc()\n'
            + _IIS_DIAGNOSTICS_INLINE
            + '\nif _reloop_exit is not None:\n'
            + '    sys.exit(_reloop_exit)\n'
            + 'if _reloop_err is not None:\n'
            + '    sys.exit(1)\n'
        )

        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return self._parse_output(result)
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "status": "TIMEOUT",
                "objective": None,
                "error": "Solver timeout"
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "status": "ERROR",
                "objective": None,
                "error": str(e)
            }

    def _parse_output(self, result) -> Dict[str, Any]:
        """Parse subprocess output including IIS/unbounded diagnostics."""
        output = {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "status": None,
            "objective": None,
            "dual_objective": None,
            "solution": None,
            "iis_constraints": None,
            "iis_bounds": None,
            "unbounded_vars": None,
            "error": result.stderr if result.returncode != 0 else None
        }

        stdout = result.stdout

        # Parse status: X
        status_match = re.search(r"status:\s*(\d+)", stdout)
        if status_match:
            code = int(status_match.group(1))
            output["status"] = {
                2: "OPTIMAL",
                3: "INFEASIBLE",
                5: "UNBOUNDED",
                9: "TIME_LIMIT"
            }.get(code, f"CODE_{code}")

        # Parse objective: X
        obj_match = re.search(
            r"objective:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            stdout
        )
        if obj_match:
            output["objective"] = float(obj_match.group(1))

        # Parse dual_objective: X
        dual_match = re.search(
            r"dual_objective:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            stdout
        )
        if dual_match:
            output["dual_objective"] = float(dual_match.group(1))

        # Parse solution: {...}
        sol_match = re.search(r"solution:\s*(\{[^}]+\})", stdout)
        if sol_match:
            try:
                output["solution"] = json.loads(sol_match.group(1))
            except Exception:
                pass

        # --- Parse IIS diagnostics ---
        iis_constrs = re.findall(r"iis_constr:\s*(.+)", stdout)
        if iis_constrs:
            output["iis_constraints"] = iis_constrs

        iis_bounds = re.findall(r"iis_bound:\s*(.+)", stdout)
        if iis_bounds:
            output["iis_bounds"] = iis_bounds

        # --- Parse unbounded diagnostics ---
        ub_vars = re.findall(r"unbounded_var:\s*(.+)", stdout)
        if ub_vars:
            output["unbounded_vars"] = ub_vars

        return output
