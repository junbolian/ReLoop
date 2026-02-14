#!/usr/bin/env python3
"""
Launch an OpenAI-compatible local model server on specific GPUs.

Backends:
- vLLM for Hugging Face model folders
- llama-cpp-python server for GGUF models
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests


MODEL_REGISTRY = {
    "optmath-gguf": "models/OptMATH-Qwen2.5-32B-Instruct-GGUF",
    "qwen3-32b": "models/Qwen3-32B",
    "sirl-gurobi32b": "models/SIRL-Gurobi32B",
}


def parse_gpus(gpus: str) -> List[str]:
    parsed = [g.strip() for g in gpus.split(",") if g.strip()]
    if not parsed:
        raise ValueError("No GPU ids provided")
    return parsed


def resolve_model_path(model: str) -> Path:
    mapped = MODEL_REGISTRY.get(model, model)
    return Path(mapped).expanduser()


def detect_gguf_file(model_path: Path) -> Optional[Path]:
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return model_path
    if not model_path.is_dir():
        return None

    direct = sorted(model_path.glob("*.gguf"))
    if direct:
        return direct[0]

    recursive = sorted(model_path.rglob("*.gguf"))
    if recursive:
        return recursive[0]
    return None


def choose_backend(requested_backend: str, model_path: Path) -> Tuple[str, Optional[Path]]:
    gguf_file = detect_gguf_file(model_path)
    if requested_backend == "auto":
        if gguf_file is not None:
            return "llama_cpp", gguf_file
        return "vllm", None
    if requested_backend == "llama_cpp":
        if gguf_file is None:
            raise ValueError(f"No .gguf file found under: {model_path}")
        return "llama_cpp", gguf_file
    return "vllm", None


def split_extra_args(extra_args: str) -> List[str]:
    return shlex.split(extra_args) if extra_args.strip() else []


def parse_tensor_split_values(tensor_split: str) -> List[str]:
    # Accept "1,1,1,1" or "1 1 1 1"
    raw = tensor_split.replace(",", " ").split()
    if not raw:
        return []
    values: List[str] = []
    for item in raw:
        # Validate numeric format expected by llama.cpp CLI.
        float(item)
        values.append(item)
    return values


def build_vllm_cmd(args: argparse.Namespace, model_path: Path, served_model_name: str, tp_size: int) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        str(model_path),
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(tp_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--enable-prefix-caching",
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(split_extra_args(args.vllm_extra_args))
    return cmd


def build_llama_cmd(args: argparse.Namespace, gguf_file: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        str(gguf_file),
        "--n_ctx",
        str(args.n_ctx),
        "--n_gpu_layers",
        str(args.n_gpu_layers),
        "--chat_format",
        args.chat_format,
    ]
    if args.tensor_split:
        cmd.append("--tensor_split")
        cmd.extend(parse_tensor_split_values(args.tensor_split))
    cmd.extend(split_extra_args(args.llama_extra_args))
    return cmd


def wait_ready(health_url: str, timeout_s: int, proc: subprocess.Popen) -> Tuple[bool, str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server exited early with code {proc.returncode}"
        try:
            resp = requests.get(health_url, timeout=3)
            if resp.status_code == 200:
                return True, "ready"
        except requests.RequestException:
            pass
        time.sleep(2)
    return False, f"timeout after {timeout_s}s waiting for {health_url}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy local LLM as OpenAI-compatible API.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model alias/path. Aliases: optmath-gguf, qwen3-32b, sirl-gurobi32b.",
    )
    parser.add_argument("--served-model-name", default=None, help="Name exposed to OpenAI API.")
    parser.add_argument("--backend", choices=["auto", "vllm", "llama_cpp"], default="auto")
    parser.add_argument("--gpus", default="4,5,6,7", help="Physical GPU ids, e.g. 4,5,6,7.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ready-timeout", type=int, default=600, help="Wait timeout for /v1/models.")
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    parser.add_argument("--no-wait-ready", action="store_true", help="Do not wait for /v1/models.")

    # vLLM args
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--dtype", default="auto")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--vllm-extra-args", default="", help="Extra raw args appended to vLLM command.")

    # llama.cpp args
    parser.add_argument("--n-ctx", type=int, default=32768)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--tensor-split", default=None, help="For llama.cpp, e.g. 1,1,1,1.")
    parser.add_argument("--chat-format", default="chatml")
    parser.add_argument("--llama-extra-args", default="", help="Extra raw args appended to llama.cpp command.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        print(f"[deploy] model path not found: {model_path}", file=sys.stderr)
        return 1

    gpus = parse_gpus(args.gpus)
    backend, gguf_file = choose_backend(args.backend, model_path)
    served_name = args.served_model_name or model_path.name

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    if backend == "vllm":
        tp_size = args.tensor_parallel_size or len(gpus)
        cmd = build_vllm_cmd(args, model_path, served_name, tp_size)
    else:
        if gguf_file is None:
            print(f"[deploy] no GGUF file found under: {model_path}", file=sys.stderr)
            return 1
        if args.tensor_split is None:
            args.tensor_split = ",".join(["1"] * len(gpus))
        cmd = build_llama_cmd(args, gguf_file)

    local_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    base_url = f"http://{local_host}:{args.port}/v1"
    health_url = f"{base_url}/models"

    print(f"[deploy] backend={backend}")
    print(f"[deploy] model={model_path}")
    if gguf_file:
        print(f"[deploy] gguf={gguf_file}")
    print(f"[deploy] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print("[deploy] command:")
    print(" ".join(shlex.quote(part) for part in cmd))

    if args.dry_run:
        print("[deploy] dry-run complete")
        return 0

    proc = subprocess.Popen(cmd, env=env)

    if not args.no_wait_ready:
        ok, msg = wait_ready(health_url, args.ready_timeout, proc)
        if not ok:
            print(f"[deploy] startup failed: {msg}", file=sys.stderr)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            return 1
        print(f"[deploy] ready: {health_url}")
        print('[deploy] export OPENAI_API_KEY="EMPTY"')
        print(f'[deploy] export OPENAI_BASE_URL="{base_url}"')
        print(f'[deploy] test: python run_ablation.py -d data/RetailOpt-190.jsonl -m "{served_name}" --local --workers 5 --enable-cpt -v')

    def _shutdown_handler(signum, frame):  # noqa: ARG001
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
