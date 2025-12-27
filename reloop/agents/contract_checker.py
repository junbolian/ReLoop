import re
from typing import List

from .agent_types import ContractReport


BANNED_PATTERNS = [
    r"\bopen\s*\(",
    r"\bjson\.load",
    r"Path\.read_text",
    r"os\.remove",
    r"shutil\.rmtree",
    r"subprocess\.",
    r"sys\.argv",
    # Block only assignments/mutations to data (reads like data[...] are allowed).
    r"data\s*\[[^\]]+\]\s*=",
]


def check_contract(script: str) -> ContractReport:
    reasons: List[str] = []
    for pat in BANNED_PATTERNS:
        if re.search(pat, script):
            reasons.append(f"contract violation: pattern '{pat}' detected")
    ok = len(reasons) == 0
    return ContractReport(ok=ok, reasons=reasons)
