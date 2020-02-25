TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode explore \
    --tensors /data/partners_ecg/hd5_subset \
    --input_tensors \
        partners_ecg_patientid \
        partners_ecg_date \
        partners_ecg_dob \
    --test_modulo 0 \
    --output_folder "/home/${USER}/ml4cvd_results/" \
    --id explore_mrn_date_dob_subset
#partners_ecg_read_md_raw \
#partners_ecg_read_pc_raw \
#partners_ecg_rate \
#partners_ecg_qrs \
#partners_ecg_pr \
#partners_ecg_qt \
#partners_ecg_qtc \
#partners_ecg_date \
#voltage_len \
