./scripts/tf.sh -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode train \
    --tensors /home/${USER}/partners_ecg/hd5 \
    --input_tensors partners_ecg_voltage \
    --output_tensors \
                        partners_ecg_rate \
                        partners_ecg_qrs \
                        partners_ecg_pr \
                        partners_ecg_qt \
                        partners_ecg_qtc \
    --inspect_model \
    --epochs 150 \
    --batch_size 256 \
    --training_steps 1024 \
    --validation_steps 32 \
    --test_steps 16 \
    --patience 10 \
    --test_modulo 0 \
    --learning_rate 0.0002 \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id intervals_all_epochs150_batch256_trainingsteps1024_learning0002
