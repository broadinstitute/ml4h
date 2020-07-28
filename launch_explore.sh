/home/lia.harrington/ml4cvd/ml4cvd/recipes.py \
        --mode explore \
        --tensors ${SLURM_JOB_SCRATCHDIR}/mgh/ \
        --input_tensors partners_ecg_rate_pc \
                        partners_ecg_rate_md \
                        partners_ecg_qrs_pc \
                        partners_ecg_qrs_md \
                        partners_ecg_pr_pc \
                        partners_ecg_pr_md \
        --output_folder /home/lia.harrington/explore_ecg_features/ \
        --id explore_ecg_features
