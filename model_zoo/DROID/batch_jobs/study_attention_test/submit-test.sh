#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || -z "$1" ]]; then
  echo "Usage: $0 gs://.../artifacts/training_runs/RUN [embeddings_dir]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_DIR="${1%/}"
EMBEDDINGS_DIR="${2:-}"
COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
BRANCH="$(git -C "${REPO_DIR}" branch --show-current)"
if [[ "${BRANCH}" != feat/droid-study-attention-survival ]]; then
  echo "Run from the feat/droid-study-attention-survival worktree." >&2
  exit 1
fi
REMOTE_COMMIT="$(git -C "${REPO_DIR}" rev-parse origin/feat/droid-study-attention-survival)"
if [[ "${COMMIT}" != "${REMOTE_COMMIT}" ]]; then
  echo "Push the current ml4h commit before submitting the Batch job." >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/generated_configs"
JOB_NAME="droid-af-attention-pool-test-$(date +%Y%m%d-%H%M%S)"
CONFIG="${SCRIPT_DIR}/generated_configs/${JOB_NAME}.yml"
python3 - "${SCRIPT_DIR}/config-test.yml" "${CONFIG}" "${RUN_DIR}" "${EMBEDDINGS_DIR}" "${COMMIT}" <<'PY'
from pathlib import Path
import sys

template, output, run_dir, embeddings_dir, commit = sys.argv[1:]
config = Path(template).read_text()
for key, value in {
    '__TRAINING_RUN_DIR__': run_dir,
    '__EMBEDDINGS_DIR__': embeddings_dir,
    '__ML4H_COMMIT__': commit,
}.items():
    if '\n' in value or '\r' in value:
        raise ValueError(f'Newlines are not allowed in {key}')
    config = config.replace(key, value)
Path(output).write_text(config)
PY

echo "Submitting ${JOB_NAME} for ${RUN_DIR} using ${CONFIG}"
gcloud batch jobs submit "${JOB_NAME}" \
  --location=us-central1 \
  --config="${CONFIG}"
