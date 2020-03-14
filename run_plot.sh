TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode plot_partners_ecg \
    --tensors /data/partners_ecg/hd5 \
    --input_tensors \
        partners_ecg_patientid \
        partners_ecg_firstname \
        partners_ecg_lastname \
        partners_ecg_gender \
        partners_ecg_dob \
        partners_ecg_age \
        partners_ecg_date \
        partners_ecg_time \
        partners_ecg_sitename \
        partners_ecg_location \
        partners_ecg_read_md_raw \
        partners_ecg_voltage \
        partners_ecg_rate \
        partners_ecg_pr \
        partners_ecg_qrs \
        partners_ecg_qt \
        partners_ecg_qtc \
        partners_ecg_paxis \
        partners_ecg_raxis \
        partners_ecg_taxis \
    --output_folder "/home/${USER}/ml4cvd_results/" \
    --id plot_partners_ecg
