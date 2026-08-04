#!/usr/bin/env bash
#
# End-to-end pipeline test for DROID-RV on the checked-in synthetic dataset.
#
# Runs three stages against ./data:
#   1. validate   check the dataset matches the format the recipes require
#   2. inference  run the published DROID-RV and DROID-RVEF checkpoints over it
#   3. training   one epoch of echo_supervised_training_recipe.py
#
# Intended to be run inside the DROID docker image, which has TensorFlow, ml4ht,
# and the video dependencies:
#
#   docker run -it -v {PATH TO CLONED ML4H DIRECTORY}:/ml4h/ alalusim/droid:latest
#   cd /ml4h/model_zoo/DROID-RV/synthetic_data/
#   ./run_smoke_test.sh                    # all three stages, DROID-RV heads
#   ./run_smoke_test.sh rvef               # training uses DROID-RVEF heads
#   STAGES=validate,inference ./run_smoke_test.sh    # skip training
#
# Requires the git-lfs data and checkpoints (see README.md):
#   git lfs pull --include model_zoo/DROID-RV/synthetic_data/data/*
#   git lfs pull --include model_zoo/DROID-RV/movinet_a2_base/*
#   git lfs pull --include model_zoo/DROID-RV/droid_rv_checkpoint/*
#   git lfs pull --include model_zoo/DROID-RV/droid_rvef_checkpoint/*
#
set -euo pipefail

VARIANT="${1:-rv}"
STAGES="${STAGES:-validate,inference,training}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROID_DIR="$(cd "${SCRIPT_DIR}/../../DROID" && pwd)"
MOVINET_CHKP_DIR="$(cd "${SCRIPT_DIR}/../movinet_a2_base" && pwd)"

DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${SCRIPT_DIR}/smoke_test_output_${VARIANT}"

N_INPUT_FRAMES=16
SKIP_MODULO=1     # 16-frame videos, so consume every frame
BATCH_SIZE=4      # small dataset: the recipe drops partial batches
EPOCHS=1

case "${VARIANT}" in
  rv)
    # Matches the readme's DROID-RV training example and output heads:
    # [[Age, RVEDD]], [[Dilated, Not Dilated]], [[Hypokinetic, Not Hypokinetic]], [[Female, Male]]
    OUTPUT_ARGS=(
      --output_labels age
      --output_labels rvedd
      --output_labels rv_size
      --output_labels rv_function
      --output_labels sex
      --output_labels_types rrccc
    )
    ;;
  rvef)
    # Matches the readme's DROID-RVEF output heads:
    # [[RVEF, RVEDV, RVESV, Age]], [[Female, Male]]
    OUTPUT_ARGS=(
      --output_labels rvef
      --output_labels rvedv
      --output_labels rvesv
      --output_labels age
      --output_labels sex
      --output_labels_types rrrrc
    )
    ;;
  *)
    echo "Unknown variant '${VARIANT}'. Use 'rv' or 'rvef'." >&2
    exit 1
    ;;
esac

if [ ! -f "${DATA_DIR}/wide_file.parquet" ]; then
  echo "No dataset at ${DATA_DIR}." >&2
  echo "Pull it with: git lfs pull --include model_zoo/DROID-RV/synthetic_data/data/*" >&2
  exit 1
fi

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

if has_stage validate; then
  echo "=============================================================="
  echo "Validating dataset format"
  echo "=============================================================="
  python "${SCRIPT_DIR}/validate_synthetic_data.py" \
      --data_dir "${DATA_DIR}" \
      --n_frames "${N_INPUT_FRAMES}" \
      --n_input_frames "${N_INPUT_FRAMES}" \
      --skip_modulo "${SKIP_MODULO}" \
      --batch_size "${BATCH_SIZE}"
fi

if has_stage inference; then
  echo "=============================================================="
  echo "Inference with the published DROID-RV and DROID-RVEF checkpoints"
  echo "=============================================================="
  python "${SCRIPT_DIR}/run_inference_smoke_test.py" \
      --data_dir "${DATA_DIR}" \
      --split test \
      --models both \
      --batch_size "${BATCH_SIZE}" \
      --n_input_frames "${N_INPUT_FRAMES}" \
      --skip_modulo "${SKIP_MODULO}" \
      --movinet_chkp_dir "${MOVINET_CHKP_DIR}" \
      --output_dir "${SCRIPT_DIR}/inference_smoke_test_output"
fi

if has_stage training; then
  echo "=============================================================="
  echo "Training (variant: ${VARIANT}, ${EPOCHS} epoch)"
  echo "=============================================================="
  mkdir -p "${OUTPUT_DIR}"

  # The recipe imports `echo_defines`, `data_descriptions` and `model_descriptions`
  # as top-level modules, so it must run from the DROID directory.
  cd "${DROID_DIR}"
  python echo_supervised_training_recipe.py \
      --n_input_frames "${N_INPUT_FRAMES}" \
      "${OUTPUT_ARGS[@]}" \
      --wide_file "${DATA_DIR}/wide_file.parquet" \
      --splits_file "${DATA_DIR}/splits.json" \
      --lmdb_folder "${DATA_DIR}/lmdb" \
      --selected_views A4C \
      --selected_views RV_focused \
      --selected_doppler standard \
      --selected_quality good \
      --selected_canonical on_axis \
      --n_train_patients all \
      --batch_size "${BATCH_SIZE}" \
      --epochs "${EPOCHS}" \
      --es_patience 5 \
      --scale_outputs \
      --skip_modulo "${SKIP_MODULO}" \
      --adam 1e-4 \
      --movinet_chkp_dir "${MOVINET_CHKP_DIR}" \
      --output_dir "${OUTPUT_DIR}"

  echo "Training artifacts in ${OUTPUT_DIR}"
  ls -R "${OUTPUT_DIR}"
fi

echo "=============================================================="
echo "Smoke test complete (stages: ${STAGES})"
echo "=============================================================="
