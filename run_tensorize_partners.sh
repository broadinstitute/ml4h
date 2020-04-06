TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -c -m /home/${USER} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode tensorize_partners \
    --tensors /home/${USER}/data/hd5_new \
    --xml_folder /data/partners_ecg/xml \
    --output_folder /home/${USER}/recipes_output \
    --id retensorize
