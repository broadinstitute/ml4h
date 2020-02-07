./scripts/tf.sh -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode train \
    --tensors /home/${USER}/partners_ecg/hd5_subset \
    --input_tensors \
                    partners_ecg_read_md_raw \
                    partners_ecg_rate \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id er_explore_tensors