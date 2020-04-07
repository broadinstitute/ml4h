TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -t -d 0 \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode train \
    --logging_level INFO \
    --tensors /data/partners_ecg/hd5 \
    --input_tensors \
        partners_ecg_2500_raw \
    --output_tensors \
        outcome_death \
        outcome_stroke \
        outcome_renal_failure \
        outcome_prolonged_ventilation \
        outcome_dsw_infection \
        outcome_reoperation \
        outcome_any_morbidity \
        outcome_long_stay \
    --test_modulo 0 \
    --batch_size 32 \
    --epochs 100 \
    --output_folder "/home/${USER}/repos/ml/results/" \
    --id sts_ecg
