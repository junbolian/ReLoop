#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PORT="${PORT:-8002}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}/v1"
GPUS="${GPUS:-4,5,6,7}"
WORKERS="${WORKERS:-8}"
DEPLOY_MODEL_PATH="${DEPLOY_MODEL_PATH:-models/OptMATH-Qwen2.5-32B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-OptMATH-Qwen2.5-32B-Instruct}"

DATASETS=(
  "data/RetailOpt-190.jsonl"
  "data/MAMO_ComplexLP_fixed.jsonl"
  "data/IndustryOR_fixedV2.jsonl"
)

DEPLOY_PID=""

wait_for_endpoint() {
  local url="$1"
  local max_retry="${2:-180}"
  local sleep_s="${3:-2}"
  local i
  for ((i = 1; i <= max_retry; i++)); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${sleep_s}"
  done
  return 1
}

stop_server() {
  if [[ -n "${DEPLOY_PID}" ]] && kill -0 "${DEPLOY_PID}" 2>/dev/null; then
    echo "[runner] stopping deploy process pid=${DEPLOY_PID}"
    kill "${DEPLOY_PID}" || true
    wait "${DEPLOY_PID}" || true
  fi
  DEPLOY_PID=""

  # Clean up any leftover process still listening on target port.
  local pids
  pids="$(lsof -tiTCP:${PORT} -sTCP:LISTEN || true)"
  if [[ -n "${pids}" ]]; then
    echo "[runner] killing leftover listener(s) on :${PORT} -> ${pids}"
    kill ${pids} || true
    sleep 2
  fi
}

cleanup_on_exit() {
  stop_server || true
}
trap cleanup_on_exit EXIT INT TERM

start_server() {
  local deploy_model="$1"
  local backend="$2"
  local served_model_name="$3"

  stop_server

  echo "[runner] starting ${deploy_model} (backend=${backend}) on ${BASE_URL}, gpus=${GPUS}"
  python3 scripts/deploy_local_llm.py \
    --model "${deploy_model}" \
    --backend "${backend}" \
    --served-model-name "${served_model_name}" \
    --gpus "${GPUS}" \
    --port "${PORT}" \
    --trust-remote-code &
  DEPLOY_PID=$!

  if ! wait_for_endpoint "${BASE_URL}/models" 240 2; then
    echo "[runner] endpoint not ready: ${BASE_URL}/models" >&2
    return 1
  fi

  echo "[runner] endpoint ready: ${BASE_URL}/models"
  curl -fsS "${BASE_URL}/models" || true
  echo
}

run_suite_for_model() {
  local model_name="$1"

  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export OPENAI_BASE_URL="${BASE_URL}"

  echo "[runner] OPENAI_BASE_URL=${OPENAI_BASE_URL}"
  echo "[runner] running verified suite for model=${model_name}"
  python3 run_ablation.py -d "${DATASETS[0]}" -m "${model_name}" --local --workers "${WORKERS}" --enable-cpt -v
  python3 run_ablation.py -d "${DATASETS[1]}" -m "${model_name}" --local --workers "${WORKERS}" --enable-cpt -v
  python3 run_ablation.py -d "${DATASETS[2]}" -m "${model_name}" --local --workers "${WORKERS}" --enable-cpt -v

  echo "[runner] running direct/no-verify suite for model=${model_name}"
  python3 run_ablation.py -d "${DATASETS[0]}" -m "${model_name}" --local --workers "${WORKERS}" --no-cot --no-verify -v
  python3 run_ablation.py -d "${DATASETS[1]}" -m "${model_name}" --local --workers "${WORKERS}" --no-cot --no-verify -v
  python3 run_ablation.py -d "${DATASETS[2]}" -m "${model_name}" --local --workers "${WORKERS}" --no-cot --no-verify -v
}

main() {
  # Run only OptMATH-Qwen2.5-32B-Instruct (HF format via vLLM).
  start_server "${DEPLOY_MODEL_PATH}" "vllm" "${SERVED_MODEL_NAME}"
  run_suite_for_model "${SERVED_MODEL_NAME}"
  stop_server

  echo "[runner] ${SERVED_MODEL_NAME} suite completed."
}

main "$@"
