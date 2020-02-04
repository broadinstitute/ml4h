TFSCRIPT="tf.sh"
#TFSCRIPT="tf_gpu2.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode explore \
    --tensors \
                /home/${USER}/partners_ecg/ \
    --input_tensors \
                    partners_ecg_read_md_clean_supranodal_rhythms \
                    partners_ecg_read_pc_clean_supranodal_rhythms \
                    partners_ecg_read_md_raw \
                    partners_ecg_read_pc_raw \
    --test_modulo 0 \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id explore_supranodal_rhythms
