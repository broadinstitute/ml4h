#!/usr/bin/env bash
set -euo pipefail

: "${TRAINING_RUN_DIR:?Set TRAINING_RUN_DIR to the source DROID run folder}"
RUN_DIR="${TRAINING_RUN_DIR%/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-gs://mgb-home/alalusim/droid-af/artifacts/study_attention_tests}"

export PYTHONPATH="/workspace/ml4h:${PYTHONPATH:-}"

# Match the shared DROID inference jobs' TensorFlow and survival dependencies.
pip install --no-cache-dir "tensorflow[and-cuda]==2.19.0" scikit-survival

ARGS=(
  --source_run_dir "${RUN_DIR}"
  --smoke_test_patients 100
  --pooling attention
  --epochs 2
  --batch_size 8
  --warmup_epochs 0
  --output_dir "${OUTPUT_ROOT%/}"
  --disable_survival_metrics_callback
  --no-run_validation_inference
)
if [[ -n "${EMBEDDINGS_DIR:-}" ]]; then
  ARGS+=(--embeddings_dir "${EMBEDDINGS_DIR}")
fi

echo "Testing study attention pooling on 100 patients from ${RUN_DIR}"
python3 /workspace/ml4h/model_zoo/DROID/echo_study_attention_training_recipe.py "${ARGS[@]}"
