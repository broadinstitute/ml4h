#!/bin/bash
# Define a string variable with a value
tensor_maps="strip_I strip_II strip_III strip_V1 strip_V2 strip_V3 strip_V4 strip_V5 strip_V6 strip_aVF strip_aVL strip_aVR ecg_rest_raw"
output_folder="/home/sam/ml/trained_models/"
model_files="--model_files"
# Iterate the string variable using for loop
for tm in $tensor_maps; do
    ./scripts/tf.sh /home/sam/ml/ml4cvd/recipes.py --mode train --tensors /mnt/disks/ecg-rest-38k-tensors/2020-03-14/ \
        --input_tensors ${tm} age_2 sex bmi_21 \
        --output_tensors adjusted_myocardium_mass_asym_outlier --training_steps 92 --validation_steps 36 --test_steps 36 --epochs 96 \
        --patience 32 --batch_size 32 --output_folder ${output_folder} --id ${tm}_age_sex_bmi_asym_amm \
        --test_csv /home/sam/lvh_hold_out.txt --learning_rate 0.00005
    model_files="${model_files} ${output_folder}${tm}_age_sex_bmi_asym_amm/${tm}_age_sex_bmi_asym_amm.h5"
done

./scripts/tf.sh /home/sam/ml/ml4cvd/recipes.py --mode compare_scalar --tensors /mnt/disks/ecg-rest-38k-tensors/2020-03-14/ \
    --input_tensors ${tensor_maps} age_2 sex bmi_21 \
    --output_tensors adjusted_myocardium_mass_asym_outlier --test_steps 128 \
    --patience 32 --batch_size 32 --output_folder ${output_folder} --id compare_leads_asym \
    ${model_files} \
    --test_csv /home/sam/lvh_hold_out.txt

model_files="--model_files"
# Iterate the string variable using for loop
for tm in $tensor_maps; do
    ./scripts/tf.sh /home/sam/ml/ml4cvd/recipes.py --mode train --tensors /mnt/disks/ecg-rest-38k-tensors/2020-03-14/ \
        --input_tensors ${tm} age_2 sex bmi_21 \
        --output_tensors adjusted_myocardium_mass_asym_outlier --training_steps 92 --validation_steps 36 --test_steps 36 --epochs 96 \
        --patience 32 --batch_size 32 --output_folder ${output_folder} --id ${tm}_age_sex_bmi_amm \
        --test_csv /home/sam/lvh_hold_out.txt --learning_rate 0.00005
    model_files="${model_files} ${output_folder}${tm}_age_sex_bmi_amm/${tm}_age_sex_bmi_amm.h5"
done

./scripts/tf.sh /home/sam/ml/ml4cvd/recipes.py --mode compare_scalar --tensors /mnt/disks/ecg-rest-38k-tensors/2020-03-14/ \
    --input_tensors ${tensor_maps} age_2 sex bmi_21 \
    --output_tensors adjusted_myocardium_mass --test_steps 128 \
    --patience 32 --batch_size 32 --output_folder ${output_folder} --id compare_leads_amm \
    ${model_files} \
    --test_csv /home/sam/lvh_hold_out.txt