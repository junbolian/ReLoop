import os
from typing import Tuple


def split_prompt_file(text: str) -> Tuple[str, str]:
    system_marker = "### SYSTEM PROMPT ###"
    user_marker = "### USER PROMPT ###"
    system_txt = ""
    user_txt = ""
    if system_marker in text and user_marker in text:
        parts = text.split(system_marker, 1)[1]
        sys_part, user_part = parts.split(user_marker, 1)
        system_txt = sys_part.strip()
        user_txt = user_part.strip()
    else:
        # Fallback: treat entire file as user prompt.
        user_txt = text.strip()
    return system_txt, user_txt


def load_prompt_for_scenario(
    scenario_id: str, prompts_dir: str
) -> Tuple[str, str]:
    cand_files = [
        os.path.join(prompts_dir, f"{scenario_id}.txt"),
        os.path.join(prompts_dir, f"{scenario_id}.prompt"),
        os.path.join(prompts_dir, f"{scenario_id}.md"),
    ]
    target = None
    for path in cand_files:
        if os.path.exists(path):
            target = path
            break
    if target is None:
        raise FileNotFoundError(
            f"Prompt file for scenario '{scenario_id}' not found in {prompts_dir}"
        )

    with open(target, "r", encoding="utf-8") as f:
        content = f.read()

    return split_prompt_file(content)
