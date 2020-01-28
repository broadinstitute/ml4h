./scripts/tf_gpu2.sh -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode train \
    --tensors /home/${USER}/partners_ecg/hd5_copies \
    --input_tensors partners_ecg_voltage \
    --output_tensors \
                        partners_ecg_rate_norm \
                        partners_ecg_qrs_norm \
                        partners_ecg_pr_norm \
                        partners_ecg_qt_norm \
                        partners_ecg_qtc_norm \
    --inspect_model \
    --epochs 50 \
    --batch_size 128 \
    --training_steps 400 \
    --validation_steps 16 \
    --test_steps 16 \
    --patience 10 \
    --test_modulo 0 \
    --learning_rate 0.0002 \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id intervals_all_norm
