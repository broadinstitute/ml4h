TFSCRIPT="tf_gpu2.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode train \
    --tensors /home/${USER}/partners_ecg/hd5 \
    --input_tensors partners_ecg_voltage \
    --output_tensors \
                        supranodal_rhythms \
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
    --id supranodal_rhythms_epochs50_batch128_trainingsteps400_learning0002
