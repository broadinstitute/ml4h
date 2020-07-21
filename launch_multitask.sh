/home/pdiachil/ml/scripts/tf.sh \
    /home/pdiachil/ml/ml4cvd/recipes.py \
        --mode explore \
        --tensors /mnt/disks/multitask-data/2020-07-14/common \
        --input_tensors \
        c_antihypertensive \
c_diabetes \
c_ethnicity_black \
c_ethnicity_white \
c_sex \
d_age \
d_biochemistry_30010 \
d_biochemistry_30020 \
d_biochemistry_30030 \
d_biochemistry_30040 \
d_biochemistry_30050 \
d_biochemistry_30060 \
d_biochemistry_30070 \
d_biochemistry_30080 \
d_biochemistry_30090 \
d_biochemistry_30100 \
d_biochemistry_30110 \
d_biochemistry_30120 \
d_biochemistry_30130 \
d_biochemistry_30140 \
d_biochemistry_30150 \
d_biochemistry_30160 \
d_biochemistry_30170 \
d_biochemistry_30180 \
d_biochemistry_30190 \
d_biochemistry_30200 \
d_biochemistry_30210 \
d_biochemistry_30220 \
d_biochemistry_30230 \
d_biochemistry_30240 \
d_biochemistry_30250 \
d_biochemistry_30260 \
d_biochemistry_30270 \
d_biochemistry_30280 \
d_biochemistry_30290 \
d_biochemistry_30300 \
d_biochemistry_30600 \
d_biochemistry_30610 \
d_biochemistry_30620 \
d_biochemistry_30630 \
d_biochemistry_30640 \
d_biochemistry_30650 \
d_biochemistry_30670 \
d_biochemistry_30680 \
d_biochemistry_30690 \
d_biochemistry_30700 \
d_biochemistry_30710 \
d_biochemistry_30720 \
d_biochemistry_30730 \
d_biochemistry_30740 \
d_biochemistry_30750 \
d_biochemistry_30760 \
d_biochemistry_30770 \
d_biochemistry_30780 \
d_biochemistry_30810 \
d_biochemistry_30830 \
d_biochemistry_30840 \
d_biochemistry_30850 \
d_biochemistry_30860 \
d_biochemistry_30870 \
d_biochemistry_30880 \
d_biochemistry_30890 \
d_diastolic_bp \
d_height \
d_hip \
d_pulse_rate \
d_systolic_bp \
d_waist \
d_weight \
incident_afib \
            incident_mi \
        --output_folder /home/pdiachil/multitask_output \
        --id explore


# /home/pdiachil/ml/scripts/tf.sh \
#     /home/pdiachil/ml/ml4cvd/recipes.py \
#         --mode train \
#         --tensors /mnt/disks/multitask-data/2020-07-14/common/ \
#         --input_tensors \
#         c_antihypertensive \
# c_diabetes \
# c_ethnicity_black \
# c_ethnicity_white \
# c_sex \
# d_age \
# d_biochemistry_30010 \
# d_biochemistry_30020 \
# d_biochemistry_30030 \
# d_biochemistry_30040 \
# d_biochemistry_30050 \
# d_biochemistry_30060 \
# d_biochemistry_30070 \
# d_biochemistry_30080 \
# d_biochemistry_30090 \
# d_biochemistry_30100 \
# d_biochemistry_30110 \
# d_biochemistry_30120 \
# d_biochemistry_30130 \
# d_biochemistry_30140 \
# d_biochemistry_30150 \
# d_biochemistry_30160 \
# d_biochemistry_30170 \
# d_biochemistry_30180 \
# d_biochemistry_30190 \
# d_biochemistry_30200 \
# d_biochemistry_30210 \
# d_biochemistry_30220 \
# d_biochemistry_30230 \
# d_biochemistry_30240 \
# d_biochemistry_30250 \
# d_biochemistry_30260 \
# d_biochemistry_30270 \
# d_biochemistry_30280 \
# d_biochemistry_30290 \
# d_biochemistry_30300 \
# d_biochemistry_30600 \
# d_biochemistry_30610 \
# d_biochemistry_30620 \
# d_biochemistry_30630 \
# d_biochemistry_30640 \
# d_biochemistry_30650 \
# d_biochemistry_30670 \
# d_biochemistry_30680 \
# d_biochemistry_30690 \
# d_biochemistry_30700 \
# d_biochemistry_30710 \
# d_biochemistry_30720 \
# d_biochemistry_30730 \
# d_biochemistry_30740 \
# d_biochemistry_30750 \
# d_biochemistry_30760 \
# d_biochemistry_30770 \
# d_biochemistry_30780 \
# d_biochemistry_30810 \
# d_biochemistry_30830 \
# d_biochemistry_30840 \
# d_biochemistry_30850 \
# d_biochemistry_30860 \
# d_biochemistry_30870 \
# d_biochemistry_30880 \
# d_biochemistry_30890 \
# d_diastolic_bp \
# d_height \
# d_hip \
# d_pulse_rate \
# d_systolic_bp \
# d_waist \
# d_weight \
#         --output_tensors \
#             incident_afib \
#             incident_mi \
#         --output_folder /home/pdiachil/multitask_output \
#         --id all