import importlib.util
import os
import shutil
import sys
from typing import Optional


def _resolve_exec(target: str) -> Optional[str]:
    if os.path.isabs(target):
        return target if os.path.exists(target) else None
    return shutil.which(target)


def _default_python() -> Optional[str]:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if os.name == "nt":
        candidate = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    else:
        candidate = os.path.join(base_dir, ".venv", "bin", "python")
    if os.path.exists(candidate):
        return candidate
    return "python"


def maybe_reexec_with_gurobi() -> None:
    if os.environ.get("RELOOP_REEXEC") == "1":
        return
    target = os.environ.get("RELOOP_PYTHON_EXEC")
    if not target:
        if importlib.util.find_spec("gurobipy") is not None:
            return
        target = _default_python()
    resolved = _resolve_exec(target) if target else None
    if not resolved:
        return
    if os.path.realpath(resolved) == os.path.realpath(sys.executable):
        return
    os.environ["RELOOP_REEXEC"] = "1"
    os.execvp(resolved, [resolved] + sys.argv)
