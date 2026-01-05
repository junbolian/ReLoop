from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .schemas import ConversationTurn, MessageEntry


# ==============================================================================
# STEP PROMPT MAPPING
# ==============================================================================
# 
# Pipeline Steps (matching user's original numbering):
#   step1: Contract extraction (what to optimize, controls, constraints)
#   step2: Spec sheet (sets, decisions, constraints, objective)
#   step3: Constraint templates (mathematical formulas)
#   step4: Code generation (Python/Gurobi script)
#   step5: Repair brief (diagnose and fix)
#
# ==============================================================================

STEP_PROMPT_MAP = {
    "step1": "01_step1_contract.txt",
    "step2": "02_step2_spec_sheet.txt",
    "step3": "03_step3_constraint_templates.txt",
    "step4": "04_step4_codegen.txt",
    "step5": "07_step5_repair_brief.txt",
}


class PromptStack:
    """Loads and assembles guardrails + base + step prompts in the required order."""

    def __init__(self, base_prompt: str, step_prompts_dir: Path):
        self.base_prompt = base_prompt
        self.step_prompts_dir = step_prompts_dir
        
        # Load guardrails
        guardrails_path = step_prompts_dir / "00_global_guardrails.txt"
        self.guardrails = guardrails_path.read_text(encoding="utf-8") if guardrails_path.exists() else ""
        
        # Load format repair prompts
        format_json_path = step_prompts_dir / "05_format_repair_json.txt"
        format_code_path = step_prompts_dir / "06_format_repair_code.txt"
        
        self.format_repair_json = format_json_path.read_text(encoding="utf-8") if format_json_path.exists() else ""
        self.format_repair_code = format_code_path.read_text(encoding="utf-8") if format_code_path.exists() else ""

    @property
    def base_prompt_hash(self) -> str:
        return hashlib.sha256(self.base_prompt.encode("utf-8")).hexdigest()

    def _step_prompt(self, step: str) -> str:
        """Load step prompt file and prepend base prompt."""
        filename = STEP_PROMPT_MAP.get(step)
        if not filename:
            raise ValueError(f"No step prompt configured for {step}")
        path = self.step_prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Step prompt file not found: {path}")
        prompt_text = path.read_text(encoding="utf-8")
        
        # Prepend base prompt to step prompt
        if self.base_prompt:
            return f"{self.base_prompt.rstrip()}\n\n{prompt_text}"
        return prompt_text

    def assemble_messages(
        self,
        step: str,
        scenario_text: str,
        data_profile: Optional[Dict[str, Any]],
        prior_outputs: Optional[Dict[str, Any]] = None,
        runtime_extra: Optional[List[str]] = None,
        repair_mode: Optional[str] = None,
        previous_response: Optional[str] = None,
        probe_diagnosis: Optional[str] = None,
    ) -> List[HumanMessage | SystemMessage | AIMessage]:
        """Compose the chat messages for a given step following injection order."""

        messages: List[HumanMessage | SystemMessage | AIMessage] = []
        
        # 1. Global guardrails (if exists)
        if self.guardrails:
            messages.append(SystemMessage(content=self.guardrails))
        
        # 2. Step-specific prompt (with base prompt prepended)
        messages.append(SystemMessage(content=self._step_prompt(step)))

        # 3. Runtime context
        runtime_blocks: List[str] = []
        
        if scenario_text:
            runtime_blocks.append(f"Scenario narrative:\n{scenario_text}")
        
        if data_profile:
            runtime_blocks.append(
                "data_profile (types/indexing only):\n"
                + json.dumps(data_profile, indent=2)
            )
        
        if prior_outputs:
            runtime_blocks.append(
                "Previous outputs:\n" + json.dumps(prior_outputs, indent=2)
            )
        
        # Include semantic probe diagnosis if available
        if probe_diagnosis:
            runtime_blocks.append(f"Semantic Probe Diagnosis:\n{probe_diagnosis}")
        
        if runtime_extra:
            runtime_blocks.append("\n".join(runtime_extra))
        
        if runtime_blocks:
            messages.append(HumanMessage(content="\n\n".join(runtime_blocks)))

        # 4. Format repair prompts (if needed)
        if repair_mode == "json":
            messages.append(SystemMessage(content=self.format_repair_json))
        elif repair_mode == "code":
            messages.append(SystemMessage(content=self.format_repair_code))

        # 5. Previous response (for repair context)
        if previous_response:
            messages.append(AIMessage(content=previous_response))
        
        return messages

    @staticmethod
    def log_turn(
        step: str, 
        messages: List[HumanMessage | SystemMessage | AIMessage], 
        response: Any
    ) -> ConversationTurn:
        """Log a conversation turn for persistence."""
        msg_entries = [
            MessageEntry(
                role="system" if isinstance(m, SystemMessage)
                     else "assistant" if isinstance(m, AIMessage)
                     else "human",
                content=m.content,
                step=step,
            )
            for m in messages
        ]
        
        resp_text = getattr(response, "content", None)
        if resp_text is None and isinstance(response, str):
            resp_text = response
        
        resp_entry = MessageEntry(role="assistant", content=resp_text or "", step=step)
        msg_entries.append(resp_entry)
        
        return ConversationTurn(
            step=step, 
            messages=msg_entries, 
            response_text=resp_text, 
            raw_response=response
        )