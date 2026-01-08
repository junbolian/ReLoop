from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..schemas import (
    ConversationTurn,
    SanityReport,
    SemanticProbeReport,
    SolveReport,
    StaticAuditReport,
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=path.parent
    ) as handle:
        handle.write(text)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, default=str)
    _atomic_write_text(path, text)


class PersistenceManager:
    def __init__(self, root: Path):
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def init_run(self, run_id: str, meta: Dict[str, Any]) -> None:
        run_path = self.run_dir(run_id)
        run_path.mkdir(parents=True, exist_ok=True)
        meta_with_time = dict(meta)
        meta_with_time["created_at"] = datetime.utcnow().isoformat()
        _atomic_write_json(run_path / "meta.json", meta_with_time)

    def log_event(self, run_id: str, event: Dict[str, Any]) -> None:
        return

    def persist_turn(
        self,
        run_id: str,
        turn_index: int,
        step_name: str,
        messages: Optional[ConversationTurn] = None,
        step_outputs: Optional[Any] = None,
        code: Optional[str] = None,
        static_audit: Optional[StaticAuditReport] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        solve_report: Optional[SolveReport] = None,
        semantic_probe_report: Optional[SemanticProbeReport] = None,  # NEW: replaces iis_report
        sanity_report: Optional[SanityReport] = None,
    ) -> None:
        turn_dir = self.run_dir(run_id) / "turns" / str(turn_index)
        turn_dir.mkdir(parents=True, exist_ok=True)
        
        if messages:
            payload = messages.model_dump(mode="json")
            # raw_response may not be JSON serializable; keep string fallback
            if messages.raw_response is not None and not isinstance(
                messages.raw_response, (dict, list, str, int, float, bool, type(None))
            ):
                payload["raw_response"] = str(messages.raw_response)
            _atomic_write_json(turn_dir / "messages.json", payload)
        
        if step_outputs is not None:
            if hasattr(step_outputs, "model_dump"):
                dumpable = step_outputs.model_dump(mode="json", by_alias=True)
            else:
                dumpable = step_outputs
            _atomic_write_json(turn_dir / "step_outputs.json", dumpable)
        
        if code is not None:
            _atomic_write_text(turn_dir / "code.py", code)
        
        if static_audit is not None:
            _atomic_write_json(
                turn_dir / "static_audit.json",
                static_audit.model_dump(mode="json", by_alias=True),
            )
        
        if stdout is not None:
            _atomic_write_text(turn_dir / "stdout.txt", stdout)
        
        if stderr is not None:
            _atomic_write_text(turn_dir / "stderr.txt", stderr)
        
        if solve_report is not None:
            _atomic_write_json(
                turn_dir / "solve.json",
                solve_report.model_dump(mode="json", by_alias=True),
            )
        
        # NEW: Semantic probe report (replaces IIS)
        if semantic_probe_report is not None:
            _atomic_write_json(
                turn_dir / "semantic_probe.json",
                semantic_probe_report.model_dump(mode="json", by_alias=True),
            )
        
        if sanity_report is not None:
            _atomic_write_json(
                turn_dir / "sanity.json",
                sanity_report.model_dump(mode="json", by_alias=True),
            )
