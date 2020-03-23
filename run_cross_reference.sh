#!/bin/bash

HOME_DIR=/home/${USER}

source ${HOME_DIR}/miniconda3/etc/profile.d/conda.sh
conda activate ml4cvd
pip install ${HOME_DIR}/repos/ml/.

python ${HOME}/repos/ml/ml4cvd/recipes.py \
    --mode cross_reference \
    --src_tensors /data/partners_ecg/mgh_muse_rest_ecg_metadata.csv \
    --src_key_join mrn \
    --src_key_time acquisition_date '%Y-%m-%d %H:%M:%S.%f' \
    --dst_tensors /data/icu/blake_8_metadata.csv \
    --dst_key_join MRN \
    --dst_key_time tDischarge '%Y-%m-%d %H:%M:%S' \
    --dst_key_outcome stpIn \
    --numeric_join \
    --dst_key_before_outcome_time tAdmit '%Y-%m-%d %H:%M:%S' \
    --test_modulo 0 \
    --output_folder /home/${USER}/recipes_output \
    --id blake_8_ecg_cross_ref

#    --days_before_outcome 30 \
