from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .schemas import ConversationTurn, MessageEntry


STEP_PROMPT_MAP = {
    "step0": "01_step0_contract.txt",
    "step1": "02_step1_tag_sentences.txt",
    "step2": "03_step2_spec_sheet.txt",
    "step3": "04_step3_constraint_templates.txt",
    "step4": "05_step4_sanity_check.txt",
    "step5": "06_step5_codegen_script.txt",
    "step6": "09_step6_repair_brief.txt",
}


class PromptStack:
    """Loads and assembles guardrails + base + step prompts in the required order."""

    def __init__(self, base_prompt: str, step_prompts_dir: Path):
        self.base_prompt = base_prompt
        self.step_prompts_dir = step_prompts_dir
        self.guardrails = (step_prompts_dir / "00_global_guardrails.txt").read_text(
            encoding="utf-8"
        )
        self.format_repair_json = (
            step_prompts_dir / "07_format_repair_json.txt"
        ).read_text(encoding="utf-8")
        self.format_repair_code = (
            step_prompts_dir / "08_format_repair_code.txt"
        ).read_text(encoding="utf-8")

    @property
    def base_prompt_hash(self) -> str:
        return hashlib.sha256(self.base_prompt.encode("utf-8")).hexdigest()

    def _step_prompt(self, step: str) -> str:
        filename = STEP_PROMPT_MAP.get(step)
        if not filename:
            raise ValueError(f"No step prompt configured for {step}")
        path = self.step_prompts_dir / filename
        prompt_text = path.read_text(encoding="utf-8")
        # Treat the base prompt as part of the step prompt payload (prefix before the step file).
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
    ) -> List[HumanMessage | SystemMessage | AIMessage]:
        """Compose the chat messages for a given step following injection order."""

        messages: List[HumanMessage | SystemMessage] = [
            SystemMessage(content=self.guardrails),
            SystemMessage(content=self._step_prompt(step)),
        ]

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
        if runtime_extra:
            runtime_blocks.append("\n".join(runtime_extra))
        if runtime_blocks:
            messages.append(HumanMessage(content="\n\n".join(runtime_blocks)))

        if repair_mode == "json":
            messages.append(SystemMessage(content=self.format_repair_json))
        elif repair_mode == "code":
            messages.append(SystemMessage(content=self.format_repair_code))

        if previous_response:
            messages.append(
                AIMessage(
                    content=previous_response
                )
            )
        return messages

    @staticmethod
    def log_turn(
        step: str, messages: List[HumanMessage | SystemMessage | AIMessage], response: Any
    ) -> ConversationTurn:
        msg_entries = [
            MessageEntry(
                role="system"
                if isinstance(m, SystemMessage)
                else "assistant"
                if isinstance(m, AIMessage)
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
            step=step, messages=msg_entries, response_text=resp_text, raw_response=response
        )
