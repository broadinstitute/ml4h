MODELFILE="/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/supranodal_rhythms/supranodal_rhythms.hd5"
TFSCRIPT="tf_gpu2.sh"
./scripts/${TFSCRIPT} -t \
    /home/${USER}/repos/ml/ml4cvd/recipes.py \
    --mode test_scalar \
    --tensors /home/${USER}/partners_ecg/hd5 \
    --input_tensors partners_ecg_voltage \
    --output_tensors \
                        supranodal_rhythms \
    --inspect_model \
    --batch_size 100 \
    --test_steps 25000 \
    --patience 10 \
    --test_modulo 0 \
    --learning_rate 0.001 \
    --model_file $MODELFILE \
    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
    --id supranodal_rhythms_eval_test
    
#TFSCRIPT="tf_gpu2.sh"
#./scripts/${TFSCRIPT} -t \
#    /home/${USER}/repos/ml/ml4cvd/recipes.py \
#    --mode train \
#    --tensors /home/${USER}/partners_ecg/hd5 \
#    --input_tensors partners_ecg_voltage \
#    --output_tensors \
#                        supranodal_rhythms \
#    --inspect_model \
#    --epochs 50 \
#    --batch_size 128 \
#    --training_steps 100 \
#    --validation_steps 40 \
#    --test_steps 20 \
#    --patience 10 \
#    --test_modulo 0 \
#    --learning_rate 0.001 \
#    --output_folder "/home/${USER}/Dropbox\ \(Partners\ HealthCare\)/partners_ecg/ml4cvd_results/" \
#    --id supranodal_rhythms
