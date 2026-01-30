import os
import sys
import subprocess


def test_smoke_llm_missing_key_exit_code():
    env = os.environ.copy()
    env.pop("OR_LLM_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    proc = subprocess.run(
        [sys.executable, "-m", "SelfEvolvedReLoop.cli", "smoke-llm"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 2
    assert "OR_LLM_API_KEY" in proc.stdout + proc.stderr
