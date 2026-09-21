#!/usr/bin/env bash
#
# Trains DROID-RV for one epoch on the sample data in ./test_data, to demonstrate
# the training recipe end to end. Run from model_zoo/DROID-RV/ inside the DROID
# docker image (see README.md for setup):
#
#   docker run -it -v {PATH TO CLONED ML4H DIRECTORY}:/ml4h/ alalusim/droid:latest
#   cd /ml4h/model_zoo/DROID-RV/
#   ./run_training_example.sh
#
set -euo pipefail

# The recipe imports `echo_defines`, `data_descriptions`, and `model_descriptions`
# as top-level modules, so it must run from the DROID directory.
cd ../DROID
python echo_supervised_training_recipe.py \
    --n_input_frames 16 \
    --output_labels age \
    --output_labels rvedd \
    --output_labels rv_size \
    --output_labels rv_function \
    --output_labels sex \
    --output_labels_types rrccc \
    --wide_file ../DROID-RV/test_data/wide_file.parquet \
    --splits_file ../DROID-RV/test_data/splits.json \
    --lmdb_folder ../DROID-RV/test_data/lmdb \
    --selected_views A4C \
    --selected_views RV_focused \
    --selected_doppler standard \
    --selected_quality good \
    --selected_canonical on_axis \
    --n_train_patients all \
    --batch_size 4 \
    --epochs 1 \
    --es_patience 5 \
    --scale_outputs \
    --skip_modulo 1 \
    --adam 1e-4 \
    --movinet_chkp_dir ../DROID-RV/movinet_a2_base \
    --output_dir ../DROID-RV/training_example_output
