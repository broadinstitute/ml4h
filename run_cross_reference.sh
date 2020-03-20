#!/bin/bash

HOME_DIR=/home/${USER}

source ${HOME_DIR}/miniconda3/etc/profile.d/conda.sh
conda activate ml4cvd
pip install ${HOME_DIR}/repos/ml/.

python ${HOME}/repos/ml/ml4cvd/recipes.py \
    --mode cross_reference \
    --src_tensors /data/partners_ecg/hd5/2003-01 \
    --src_key_join partners_ecg_patientid \
    --src_key_time partners_ecg_date %m-%d-%Y \
    --dst_tensors /data/sts/all_cases_metadata_padded.csv \
    --dst_key_join mrn \
    --dst_key_time surgerydate %d%b%Y \
    --dst_key_outcome operativemortality \
    --numeric_join \
    --days_before_outcome 30 \
    --test_modulo 0