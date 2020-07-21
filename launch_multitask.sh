# /home/pdiachil/ml/scripts/tf.sh -c \
#     /home/pdiachil/ml/ml4cvd/recipes.py \
#         --mode explore \
#         --tensors /mnt/disks/multitask-data/2020-07-14/common \
#         --input_tensors \
#             d_age \
#             c_sex \
#             c_antihypertensive \
#             c_lipidlowering \
#             d_systolic_bp \
#             d_diastolic_bp \
#             d_pulse_rate \
#             c_diabetes \
#             d_weight \
#             d_height \
#             d_waist \
#             d_hip \
#             c_ethnicity_white \
#             c_ethnicity_black \
#             c_chest_pain \
#             d_biochemistry_30010 \
#         --output_folder /home/pdiachil/explore_output \
#         --id all


# /home/pdiachil/ml/scripts/tf.sh -c \
#     /home/pdiachil/ml/ml4cvd/recipes.py \
#         --mode train \
#         --tensors /mnt/disks/multitask-data/2020-07-14/common \
#         --input_tensors \
#             d_age \
#             c_sex \
#             c_antihypertensive \
#             c_lipidlowering \
#             d_systolic_bp \
#             d_diastolic_bp \
#             d_pulse_rate \
#             c_diabetes \
#             d_weight \
#             d_height \
#             d_waist \
#             d_hip \
#             c_ethnicity_white \
#             c_ethnicity_black \
#             c_chest_pain \
#             d_biochemistry_30010 \
#         --output_tensors \
#             incident_afib \
#             incident_mi \
#         --output_folder /home/pdiachil/explore_output \
#         --id all