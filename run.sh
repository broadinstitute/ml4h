./scripts/tf.sh -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode train \
    --tensors /home/${USER}/partners_ecg/hd5 \
    --input_tensors partners_ecg_voltage \
    --output_tensors \
                        partners_ecg_qt \
    --inspect_model \
    --epochs 50 \
    --batch_size 128 \
    --training_steps 256 \
    --validation_steps 32 \
    --test_steps 16 \
    --patience 10 \
    --test_modulo 0 \
    --learning_rate 0.00002 \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id qt
