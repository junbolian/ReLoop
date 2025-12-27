from typing import List, Optional

from .agent_types import ContractReport, ExecutionResult, SemanticReport


def build_repair_brief(
    contract: Optional[ContractReport],
    exec_result: ExecutionResult,
    semantic: Optional[SemanticReport],
    iis_summary: Optional[dict],
) -> str:
    lines: List[str] = []
    if contract and not contract.ok:
        lines.append("Contract violations detected:")
        lines.extend(f"- {r}" for r in contract.reasons)
    if exec_result.traceback:
        lines.append("Runtime error:")
        lines.append(exec_result.traceback.strip().splitlines()[-1])
    if exec_result.success and exec_result.feasible is False:
        lines.append("Model infeasible. Focus on missing/contradictory constraints.")
    if semantic and not semantic.valid:
        if semantic.missing_prefixes:
            lines.append("Missing structural pieces: " + ", ".join(semantic.missing_prefixes))
        if semantic.unexpected_modules:
            lines.append("Remove inactive modules: " + ", ".join(semantic.unexpected_modules))
    if iis_summary and iis_summary.get("clusters"):
        top = iis_summary["clusters"][:3]
        lines.append("IIS hotspots:")
        for c in top:
            lines.append(f"- {c['prefix']} (x{c['count']})")

    if not lines:
        return "No issues detected. You may focus on improving objective quality while keeping semantics intact."
    return "Repair Brief:\n" + "\n".join(lines)
