/home/${USER}/ml4cvd/ml4cvd/recipes.py \
        --mode explore \
        --tensors ${SLURM_JOB_SCRATCHDIR}/mgh/ \
        --input_tensors partners_ecg_rate_pc \
                        partners_ecg_rate_md \
                        partners_ecg_qrs_pc \
                        partners_ecg_qrs_md \
                        partners_ecg_pr_pc \
                        partners_ecg_pr_md \
                        partners_ecg_qt_pc \
                        partners_ecg_qt_md \
                        partners_ecg_paxis_pc \
                        partners_ecg_paxis_md \
                        partners_ecg_raxis_pc \
                        partners_ecg_raxis_md \
                        partners_ecg_taxis_pc \
                        partners_ecg_taxis_md \
        --output_folder /home/${USER}/explore_ecg_features/ \
        --id explore_ecg_features
