DIRNAME=/data/partners_ecg/covid
./scripts/tf.sh -t -m $DIRNAME ${HOME}/ml/ml4cvd/recipes.py \
    --mode explore \
    --tensors $DIRNAME \
    --input_tensors \
        partners_ecg_datetime \
        partners_ecg_patientid \
        partners_ecg_patientid_clean \
        partners_ecg_firstname \
        partners_ecg_lastname \
        partners_ecg_age \
        partners_ecg_rate \
        partners_ecg_dob \
        partners_ecg_qrs \
        partners_ecg_pr \
        partners_ecg_qt \
        partners_ecg_qtc \
        partners_ecg_read_md_raw \
        partners_ecg_read_pc_raw \
        partners_ecg_sitename \
        partners_ecg_location \
        partners_ecg_gender \
    --output_folder $DIRNAME \
    --id explore
