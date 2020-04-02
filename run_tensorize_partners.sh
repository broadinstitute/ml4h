TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode tensorize_partners \
    --tensors /data/partners_ecg/hd5_new \
    --xml_folder /data/partners_ecg/xml_not_tensorized
