#TFSCRIPT="tf.sh"
TFSCRIPT="tf_gpu2.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode explore \
    --tensors \
                /home/${USER}/partners_ecg/hd5_subset \
    --input_tensors \
                    partners_ecg_get_read \
                    supranodal_rhythms \
    --test_modulo 0 \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id explore_supranodal_rhythms_get_read
