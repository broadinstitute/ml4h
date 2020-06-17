#!/bin/bash
# Define a string variable with a value
tensor_maps="strip_I strip_II strip_III strip_V1 strip_V2 strip_V3 strip_V4 strip_V5 strip_V6 strip_aVF strip_aVL strip_aVR ecg_rest_raw"

# Iterate the string variable using for loop
for tm in $tensor_maps; do
    echo ./scripts/tf.sh /home/sam/ml/ml4cvd/recipes.py --mode train --tensors /mnt/disks/ecg-rest-38k-tensors/2020-03-14/ \
    --input_tensors ${tm} \
    --output_tensors adjusted_myocardium_mass --training_steps 72 --validation_steps 36 --test_steps 36 --epochs 92 \
    --patience 32 --batch_size 32 --output_folder /home/sam/ml/trained_models/ --id ${tm}_amm \
    --test_csv /home/sam/lvh_hold_out.txt --learning_rate 0.0001
done