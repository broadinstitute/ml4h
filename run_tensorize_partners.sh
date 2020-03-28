TFSCRIPT="tf.sh"
./scripts/${TFSCRIPT} -m /home/${USER}/data/ -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode tensorize_partners \
    --tensors /home/${USER}/data/hd5 \
    --xml_folder /data/partners_ecg/xml/xml_not_tensorized/
#    --xml_folder /home/sn69/data/xml/temp/
