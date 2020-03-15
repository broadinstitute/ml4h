TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode plot_partners_ecg \
    --tensors /data/partners_ecg/hd5 \
    --output_folder "/home/${USER}/ml4cvd_results/" \
    --id plot_partners_ecg
