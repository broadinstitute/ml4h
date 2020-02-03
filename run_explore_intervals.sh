TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode explore \
    --tensors /home/${USER}/partners_ecg/hd5_subset \
    --input_tensors \
                    partners_ecg_rate \
                    partners_ecg_qrs \
                    partners_ecg_pr \
                    partners_ecg_qt \
                    partners_ecg_qtc \
    --test_modulo 0 \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id explore_intervals
