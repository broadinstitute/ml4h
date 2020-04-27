./scripts/tf.sh -t -m "/home/${USER}/" -d 1 \
    /home/${USER}/ml/ml4cvd/recipes.py \
    --mode train \
    --logging_level INFO \
    --tensors "/data/partners_ecg/mgh/hd5" \
    --input_tensors \
        partners_ecg_2500_raw \
    --output_tensors \
        outcome_death \
    --inspect_model \
    --test_modulo 0 \
    --epochs 5 \
    --output_folder "/home/${USER}/sts-ecg-results/" \
    --id 2020-04-17
    #outcome_stroke \
    #outcome_renal_failure \
    #outcome_prolonged_ventilation \
    #outcome_dsw_infection \
    #outcome_reoperation \
    #outcome_any_morbidity \
    #outcome_long_stay \

