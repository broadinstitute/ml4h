TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -m /home/sn69/data/ -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode tensorize_partners \
    --tensors /home/sn69/data/hd5 \
    --xml_folder /data/partners_ecg/xml_not_tensorized/
    # --xml_folder /home/sn69/data/xml/
