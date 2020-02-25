TFSCRIPT=tf.sh
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode explore \
    --tensors /data/partners_ecg/hd5_subset \
    --input_tensors \
        partners_ecg_patientid_cross_reference_apollo \
        partners_ecg_date \
        partners_ecg_dob \
    --validator_csv /data/apollo/demographics.csv \
    --validator_key_tensor patientid \
    --validator_key_csv Patient_ID \
    --test_modulo 0 \
    --output_folder /home/${USER}/ml4cvd_results/ \
    --id explore_cross_reference_apollo

