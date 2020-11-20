python /home/${USER}/ml4cvd/ml4h/recipes.py \
        --mode explore \
        --tensors ${SLURM_JOB_SCRATCHDIR} \
        --time_tensor partners_ecg_datetime \
        --tensormap_prefix ml4h.tensormap.mgb.ecg \
        --sample_csv /home/${USER}/mgh_mgb_ids.csv \
        --input_tensors partners_ecg_patientid \
                        partners_ecg_datetime \
                        partners_ecg_dob \
                        partners_ecg_age \
                        partners_ecg_sitename \
                        partners_ecg_pr_pc \
                        partners_ecg_pr_md \
                        partners_ecg_measurement_matrix_max_pdur \
                        partners_ecg_measurement_matrix_max_bmpar \
                        partners_ecg_measurement_matrix_V1_bmpi \
                        partners_ecg_measurement_matrix_V1_bmppi \
                        partners_ecg_measurement_matrix_V1_ppamp \
                        partners_ecg_measurement_matrix_V1_ppdur \
                        partners_ecg_measurement_matrix_V1_bmpar \
                        partners_ecg_qrs_pc \
                        partners_ecg_qrs_md \
                        partners_ecg_measurement_matrix_max_qrsa \
                        partners_ecg_qt_pc \
                        partners_ecg_qt_md \
                        partners_ecg_qtc_pc \
                        partners_ecg_qtc_md \
                        partners_ecg_rate_pc \
                        partners_ecg_rate_md \
                        partners_ecg_measurement_matrix_I_ramp \
                        partners_ecg_measurement_matrix_II_ramp \
                        partners_ecg_measurement_matrix_III_ramp \
                        partners_ecg_measurement_matrix_V1_ramp \
                        partners_ecg_measurement_matrix_V2_ramp \
                        partners_ecg_measurement_matrix_V3_ramp \
                        partners_ecg_measurement_matrix_V4_ramp \
                        partners_ecg_measurement_matrix_V5_ramp \
                        partners_ecg_measurement_matrix_V6_ramp \
                        partners_ecg_measurement_matrix_aVR_ramp \
                        partners_ecg_measurement_matrix_aVL_ramp \
                        partners_ecg_measurement_matrix_aVF_ramp \
                        partners_ecg_measurement_matrix_I_samp \
                        partners_ecg_measurement_matrix_II_samp \
                        partners_ecg_measurement_matrix_III_samp \
                        partners_ecg_measurement_matrix_V1_samp \
                        partners_ecg_measurement_matrix_V2_samp \
                        partners_ecg_measurement_matrix_V3_samp \
                        partners_ecg_measurement_matrix_V4_samp \
                        partners_ecg_measurement_matrix_V5_samp \
                        partners_ecg_measurement_matrix_V6_samp \
                        partners_ecg_measurement_matrix_aVR_samp \
                        partners_ecg_measurement_matrix_aVL_samp \
                        partners_ecg_measurement_matrix_aVF_samp \
                        partners_ecg_measurement_matrix_max_bmri \
                        partners_ecg_measurement_matrix_I_stj \
                        partners_ecg_measurement_matrix_II_stj \
                        partners_ecg_measurement_matrix_III_stj \
                        partners_ecg_measurement_matrix_V1_stj \
                        partners_ecg_measurement_matrix_V2_stj \
                        partners_ecg_measurement_matrix_V3_stj \
                        partners_ecg_measurement_matrix_V4_stj \
                        partners_ecg_measurement_matrix_V5_stj \
                        partners_ecg_measurement_matrix_V6_stj \
                        partners_ecg_measurement_matrix_aVR_stj \
                        partners_ecg_measurement_matrix_aVL_stj \
                        partners_ecg_measurement_matrix_aVF_stj \
                        partners_ecg_measurement_matrix_I_stm \
                        partners_ecg_measurement_matrix_II_stm \
                        partners_ecg_measurement_matrix_III_stm \
                        partners_ecg_measurement_matrix_V1_stm \
                        partners_ecg_measurement_matrix_V2_stm \
                        partners_ecg_measurement_matrix_V3_stm \
                        partners_ecg_measurement_matrix_V4_stm \
                        partners_ecg_measurement_matrix_V5_stm \
                        partners_ecg_measurement_matrix_V6_stm \
                        partners_ecg_measurement_matrix_aVR_stm \
                        partners_ecg_measurement_matrix_aVL_stm \
                        partners_ecg_measurement_matrix_aVF_stm \
                        partners_ecg_measurement_matrix_I_ste \
                        partners_ecg_measurement_matrix_II_ste \
                        partners_ecg_measurement_matrix_III_ste \
                        partners_ecg_measurement_matrix_V1_ste \
                        partners_ecg_measurement_matrix_V2_ste \
                        partners_ecg_measurement_matrix_V3_ste \
                        partners_ecg_measurement_matrix_V4_ste \
                        partners_ecg_measurement_matrix_V5_ste \
                        partners_ecg_measurement_matrix_V6_ste \
                        partners_ecg_measurement_matrix_aVR_ste \
                        partners_ecg_measurement_matrix_aVL_ste \
                        partners_ecg_measurement_matrix_aVF_ste \
                        partners_ecg_measurement_matrix_I_tdur \
                        partners_ecg_measurement_matrix_II_tdur \
                        partners_ecg_measurement_matrix_III_tdur \
                        partners_ecg_measurement_matrix_V1_tdur \
                        partners_ecg_measurement_matrix_V2_tdur \
                        partners_ecg_measurement_matrix_V3_tdur \
                        partners_ecg_measurement_matrix_V4_tdur \
                        partners_ecg_measurement_matrix_V5_tdur \
                        partners_ecg_measurement_matrix_V6_tdur \
                        partners_ecg_measurement_matrix_aVR_tdur \
                        partners_ecg_measurement_matrix_aVL_tdur \
                        partners_ecg_measurement_matrix_aVF_tdur \
                        partners_ecg_measurement_matrix_max_tarea \
                        partners_ecg_measurement_matrix_I_bmti \
                        partners_ecg_measurement_matrix_II_bmti \
                        partners_ecg_measurement_matrix_III_bmti \
                        partners_ecg_measurement_matrix_V1_bmti \
                        partners_ecg_measurement_matrix_V2_bmti \
                        partners_ecg_measurement_matrix_V3_bmti \
                        partners_ecg_measurement_matrix_V4_bmti \
                        partners_ecg_measurement_matrix_V5_bmti \
                        partners_ecg_measurement_matrix_V6_bmti \
                        partners_ecg_measurement_matrix_aVR_bmti \
                        partners_ecg_measurement_matrix_aVL_bmti \
                        partners_ecg_measurement_matrix_aVF_bmti \
                        partners_ecg_read_pc \
                        partners_ecg_read_md \
        --output_folder /home/${USER}/${MOUNT_BUCKETS}/ \
        --id mgb_biobank
