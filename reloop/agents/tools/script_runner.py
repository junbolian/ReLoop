from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import textwrap
from typing import Any, Dict, Optional, Tuple

from ..schemas import IISReport, SolveReport
from .iis_tools import summarize_prefix_counts


def run_script(
    code: str, data: Dict[str, Any], timeout: int = 180
) -> Tuple[SolveReport, Optional[IISReport], str, str]:
    """Exec the generated code in an isolated subprocess with pre-loaded data."""

    env = os.environ.copy()
    env["RELOOP_CODE_B64"] = base64.b64encode(code.encode("utf-8")).decode("utf-8")
    env["RELOOP_DATA_B64"] = base64.b64encode(
        json.dumps(data).encode("utf-8")
    ).decode("utf-8")

    runner = textwrap.dedent(
        """
        import base64, json, os, sys, traceback, time
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception:  # pragma: no cover - gurobipy may be missing in tests
            gp = None
            GRB = None

        code = base64.b64decode(os.environ["RELOOP_CODE_B64"]).decode("utf-8")
        data = json.loads(base64.b64decode(os.environ["RELOOP_DATA_B64"]).decode("utf-8"))
        globals_ns = {"data": data, "__name__": "__main__"}

        start = time.time()
        solve_meta = {"status": None, "obj_val": None, "obj_bound": None, "mip_gap": None}
        iis_constraints = []

        try:
            exec(compile(code, "<generated>", "exec"), globals_ns)
        except SystemExit as exc:  # pragma: no cover - passthrough
            solve_meta["exit_code"] = int(exc.code) if isinstance(exc.code, int) else 0
        except Exception:
            traceback.print_exc()
            solve_meta["error"] = "traceback"
        finally:
            solve_meta["elapsed_sec"] = time.time() - start
            model = globals_ns.get("m") or globals_ns.get("model")
            if model is not None:
                try:
                    status_val = getattr(model, "Status", None)
                    solve_meta["status"] = status_val
                    solve_meta["obj_val"] = getattr(model, "objVal", None)
                    solve_meta["obj_bound"] = getattr(model, "objBound", None)
                    solve_meta["mip_gap"] = getattr(model, "MIPGap", None)
                    if gp and status_val in (
                        gp.GRB.INFEASIBLE,
                        gp.GRB.INF_OR_UNBD,
                        gp.GRB.UNBOUNDED,
                    ):
                        try:
                            model.computeIIS()
                            iis_constraints = [
                                c.ConstrName
                                for c in model.getConstrs()
                                if getattr(c, "IISConstr", 0)
                            ]
                        except Exception:
                            pass
                except Exception:
                    pass
            print("SOLVE_JSON::" + json.dumps(solve_meta))
            if iis_constraints:
                print("IIS_JSON::" + json.dumps(iis_constraints))
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )

    solve_meta: Dict[str, Any] = {}
    iis_constraints: Optional[list] = None
    for line in proc.stdout.splitlines():
        if line.startswith("SOLVE_JSON::"):
            payload = line.split("SOLVE_JSON::", 1)[1]
            try:
                solve_meta = json.loads(payload)
            except json.JSONDecodeError:
                solve_meta = {"error": "could not parse solve metadata", "raw": payload}
        if line.startswith("IIS_JSON::"):
            payload = line.split("IIS_JSON::", 1)[1]
            try:
                iis_constraints = json.loads(payload)
            except json.JSONDecodeError:
                iis_constraints = []

    solve_report = SolveReport(
        status=str(solve_meta.get("status")) if "status" in solve_meta else None,
        obj_val=solve_meta.get("obj_val"),
        obj_bound=solve_meta.get("obj_bound"),
        mip_gap=solve_meta.get("mip_gap"),
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
        elapsed_sec=solve_meta.get("elapsed_sec"),
    )

    iis_report: Optional[IISReport] = None
    if iis_constraints is not None:
        prefix_summary = summarize_prefix_counts(iis_constraints)
        iis_report = IISReport(constraints=iis_constraints, prefix_summary=prefix_summary)

    return solve_report, iis_report, proc.stdout, proc.stderr
