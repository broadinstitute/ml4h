TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode explore \
    --tensors /home/${USER}/partners_ecg/hd5/1995-01 \
    --input_tensors supranodal_rhythms \
    --test_modulo 0 \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id explore_supranodal_rhythms
