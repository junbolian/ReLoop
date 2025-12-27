import io
import traceback as tb
import multiprocessing as mp
import time
import importlib.util
from typing import Any, Dict, Set, List

from .agent_types import ExecutionResult

# 仅采样少量原始 name（调试用），不影响语义检查
_MAX_NAME_SAMPLES = 2000

# semantic_check 里用到的“二段式约束前缀”
_TWO_TOKEN_CONSTR_PREFIXES = {
    "prod_cap",
    "storage_cap",
    "sales_conservation",
    "demand_route",
    "expire_clear",
    "moq_lb",
    "moq_ub",
}

def _find_model(globals_env: Dict[str, Any]):
    for v in globals_env.values():
        if hasattr(v, "getConstrs") and hasattr(v, "Status"):
            return v
    return None


def _var_token(name: str) -> str:
    # semantic_check 期望：I_ / S_ / Q_ / X_ / TR_
    if "_" in name:
        return name.split("_", 1)[0] + "_"
    # 如果没有下划线，至少返回整个 name（不一定能命中 semantic_check，但不会报错）
    return name


def _constr_token(name: str) -> str:
    # semantic_check 期望：aging / expire_clear / prod_cap / storage_cap / ...
    parts = name.split("_")
    if len(parts) >= 2:
        first2 = "_".join(parts[:2])
        if first2 in _TWO_TOKEN_CONSTR_PREFIXES:
            return first2
    return parts[0] if parts else name


def _install_name_capture_hooks():
    """
    在子进程里 patch gurobipy.Model.addVar/addConstr 以捕获前缀 token。
    避免事后 getVars/getConstrs 的巨量枚举 + pickle 回传。
    """
    captured = {
        "var_tokens": set(),       # type: Set[str]
        "constr_tokens": set(),    # type: Set[str]
        "var_samples": [],         # type: List[str]
        "constr_samples": [],      # type: List[str]
    }

    try:
        import gurobipy as gp
    except Exception:
        def restore():
            return
        return restore, captured

    ModelCls = gp.Model
    orig_addVar = getattr(ModelCls, "addVar", None)
    orig_addConstr = getattr(ModelCls, "addConstr", None)

    if orig_addVar is None or orig_addConstr is None:
        def restore():
            return
        return restore, captured

    def addVar_wrap(self, *args, **kwargs):
        name = kwargs.get("name", None)
        if name is None and args and isinstance(args[-1], str):
            name = args[-1]
        if isinstance(name, str) and name:
            captured["var_tokens"].add(_var_token(name))
            if len(captured["var_samples"]) < _MAX_NAME_SAMPLES:
                captured["var_samples"].append(name)
        return orig_addVar(self, *args, **kwargs)

    def addConstr_wrap(self, *args, **kwargs):
        name = kwargs.get("name", None)
        if name is None and args and isinstance(args[-1], str):
            name = args[-1]
        if isinstance(name, str) and name:
            captured["constr_tokens"].add(_constr_token(name))
            if len(captured["constr_samples"]) < _MAX_NAME_SAMPLES:
                captured["constr_samples"].append(name)
        return orig_addConstr(self, *args, **kwargs)

    ModelCls.addVar = addVar_wrap
    ModelCls.addConstr = addConstr_wrap

    def restore():
        ModelCls.addVar = orig_addVar
        ModelCls.addConstr = orig_addConstr

    return restore, captured


def _worker(script_text: str, data: Dict[str, Any], request_iis: bool, queue: mp.Queue):
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    start = time.time()

    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "traceback": "",
        "duration": 0.0,
        "status_code": None,
        "status_str": "",
        "feasible": None,
        "objective": None,
        "constr_names": [],
        "var_names": [],
        "iis_constraints": [],
        "license_limited": False,
    }

    sys_stdout = None
    sys_stderr = None
    restore_hooks = None
    captured = None

    try:
        import sys
        sys_stdout, sys_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        # 捕获 token（不做全量枚举）
        restore_hooks, captured = _install_name_capture_hooks()

        # 关键：给 exec 环境补 __file__，避免 prelude 用 __file__ 报错
        globals_env = {
            "data": data,
            "__name__": "__main__",
            "__file__": __file__,
        }

        exec(script_text, globals_env)

        if restore_hooks:
            try:
                restore_hooks()
            except Exception:
                pass

        # 用 token 填充 var_names / constr_names，semantic_check 直接 startswith 命中
        if captured:
            var_tokens = sorted(list(captured["var_tokens"]))
            constr_tokens = sorted(list(captured["constr_tokens"]))

            # 直接放 token 本身即可（例如 "I_"、"prod_cap"）
            result["var_names"] = var_tokens
            result["constr_names"] = constr_tokens

            # 可选：再附加少量样本名（调试用），不会爆炸
            # （如果你完全不想要样本，可删除下面两行）
            result["var_names"].extend(captured["var_samples"][:50])
            result["constr_names"].extend(captured["constr_samples"][:50])

        model = _find_model(globals_env)
        if model:
            status = getattr(model, "Status", None)
            result["status_code"] = status
            result["status_str"] = str(status)

            try:
                result["objective"] = model.objVal if getattr(model, "SolCount", 0) > 0 else None
            except Exception:
                result["objective"] = None

            try:
                from gurobipy import GRB
                result["feasible"] = status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)
                if status == GRB.OPTIMAL:
                    result["status_str"] = "OPTIMAL"
                elif status == GRB.TIME_LIMIT and getattr(model, "SolCount", 0) > 0:
                    result["status_str"] = "OPTIMAL (TL)"
                elif status == GRB.TIME_LIMIT:
                    result["status_str"] = "TIME_LIMIT"
            except Exception:
                result["feasible"] = status not in (3, 4)

            # IIS：只有 infeasible 且 request_iis=True 才做
            if request_iis:
                need_iis = False
                try:
                    from gurobipy import GRB
                    need_iis = status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD)
                except Exception:
                    need_iis = status in (3, 4)

                if need_iis:
                    try:
                        model.computeIIS()
                        iis_list = []
                        # 这里必须遍历约束，才能找 IISConstr=True 的那部分
                        for c in model.getConstrs():
                            try:
                                if getattr(c, "IISConstr", False):
                                    iis_list.append(
                                        {
                                            "name": c.ConstrName,
                                            "sense": getattr(c, "Sense", ""),
                                            "rhs": getattr(c, "RHS", None),
                                        }
                                    )
                            except Exception:
                                continue
                        result["iis_constraints"] = iis_list
                    except Exception:
                        pass

        result["success"] = True

    except Exception:
        result["traceback"] = tb.format_exc()

    finally:
        result["duration"] = time.time() - start
        result["stdout"] = stdout_buf.getvalue()
        result["stderr"] = stderr_buf.getvalue()

        err_text = (result["stderr"] + result.get("traceback", "")).lower()
        if "size-limited license" in err_text or "size limited license" in err_text:
            result["license_limited"] = True

        try:
            import sys
            if sys_stdout is not None:
                sys.stdout = sys_stdout
            if sys_stderr is not None:
                sys.stderr = sys_stderr
        except Exception:
            pass

        if restore_hooks:
            try:
                restore_hooks()
            except Exception:
                pass

        queue.put(result)


def execute_script(
    script_text: str,
    data: Dict[str, Any],
    timeout_s: float = 60.0,
    request_iis: bool = False,
) -> ExecutionResult:
    if importlib.util.find_spec("gurobipy") is None:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            traceback="Missing dependency: gurobipy (install and configure Gurobi license).",
            duration=0.0,
            license_limited=False,
        )

    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(script_text, data, request_iis, queue), daemon=True)
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            traceback="Timeout reached",
            duration=timeout_s,
            license_limited=False,
        )

    if queue.empty():
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            traceback="No result returned from execution process.",
            duration=timeout_s,
            license_limited=False,
        )

    res = queue.get()
    return ExecutionResult(
        success=res.get("success", False),
        stdout=res.get("stdout", ""),
        stderr=res.get("stderr", ""),
        traceback=res.get("traceback", ""),
        duration=res.get("duration", timeout_s),
        status_code=res.get("status_code"),
        status_str=res.get("status_str", ""),
        feasible=res.get("feasible"),
        objective=res.get("objective"),
        constr_names=res.get("constr_names", []),
        var_names=res.get("var_names", []),
        iis_constraints=res.get("iis_constraints", []),
        license_limited=res.get("license_limited", False),
    )
