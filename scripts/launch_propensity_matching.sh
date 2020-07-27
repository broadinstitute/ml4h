# for normalization in _raw_
# do
#     for samples in 5000
#     do
#     /home/paolo/ml/scripts/tf.sh \
#     /home/paolo/ml/ml4cvd/recipes.py \
#         --mode train \
#         --tensors /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/ \
#         --input_tensors partners_ecg_${samples}${normalization}oldest \
#         --output_tensors partners_ecg_bias_locationcardiology_oldest \
#         --train_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/train.csv \
#         --valid_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/valid.csv \
#         --test_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/test.csv \
#         --training_steps 1676 --validation_steps 718 --test_steps 266 \
#         --batch_size 32 --conv_x 71 \
#         --epochs 5 \
#         --output_folder /home/paolo/mgh_mrns_to_extract/waveform_analysis \
#         --id all_${samples}${normalization}
#     done
# done


# for normalization in _raw_
# do
#     for samples in 2500
#     do
#     /home/paolo/ml/scripts/tf.sh \
#     /home/paolo/ml/ml4cvd/recipes.py \
#         --mode train \
#         --tensors /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/ \
#         --input_tensors partners_ecg_${samples}${normalization}oldest \
#         --output_tensors partners_ecg_bias_locationcardiology_oldest \
#         --train_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/train.csv \
#         --valid_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/valid.csv \
#         --test_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/test.csv \
#         --training_steps 1676 --validation_steps 718 --test_steps 266 \
#         --batch_size 32 --conv_x 36 \
#         --epochs 5 \
#         --output_folder /home/paolo/mgh_mrns_to_extract/waveform_analysis \
#         --id all_${samples}${normalization}
#     done
# done


!usr/bin/bash
for normalization in _raw_
do
    for samples in 2500
    do
    /home/paolo/ml/scripts/tf.sh \
    /home/paolo/ml/ml4cvd/recipes.py \
        --mode train \
        --tensors /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_${samples}/ \
        --input_tensors partners_ecg_${samples}${normalization}oldest_notch \
        --output_tensors partners_ecg_bias_locationcardiology_oldest \
        --train_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_${samples}/train.csv \
        --valid_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_${samples}/valid.csv \
        --test_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_${samples}/test.csv \
        --training_steps 1096 --validation_steps 469 --test_steps 174 \
        --batch_size 32 --conv_x 36 \
        --epochs 5 \
        --output_folder /home/paolo/mgh_mrns_to_extract/waveform_analysis \
        --id ${samples}${normalization}_notch
    done
done


# for normalization in _raw_
# do
#     for samples in 5000
#     do
#     /home/paolo/ml/scripts/tf.sh \
#     /home/paolo/ml/ml4cvd/recipes.py \
#         --mode train \
#         --tensors /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_${samples}/ \
#         --input_tensors partners_ecg_2500${normalization}oldest \
#         --output_tensors partners_ecg_bias_locationcardiology_oldest \
#         --train_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_${samples}/train.csv \
#         --valid_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_${samples}/valid.csv \
#         --test_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_${samples}/test.csv \
#         --training_steps 389 --validation_steps 166 --test_steps 61 \
#         --batch_size 32 --conv_x 71 \
#         --epochs 5 \
#         --learning_rate 0.00002 \
#         --output_folder /home/paolo/mgh_mrns_to_extract/waveform_analysis \
#         --id ${samples}${normalization}
#     done
# done

# for normalization in _raw_
# do
#     for samples in 5000
#     do
#     /home/paolo/ml/scripts/tf.sh \
#     /home/paolo/ml/ml4cvd/recipes.py \
#         --mode train \
#         --tensors /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/ \
#         --input_tensors partners_ecg_age_newest partners_adult_gender_newest \
#         --output_tensors partners_ecg_bias_locationcardiology_oldest \
#         --train_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/train.csv \
#         --valid_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/valid.csv \
#         --test_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/test.csv \
#         --training_steps 1676 --validation_steps 718 --test_steps 266 \
#         --batch_size 32 \
#         --epochs 5 \
#         --output_folder /home/paolo/mgh_mrns_to_extract/waveform_analysis \
#         --id all_age_sex
#     done
# done


# for normalization in _raw_
# do
#     for samples in 2500
#     do
#     /home/paolo/ml/scripts/tf.sh \
#     /home/paolo/ml/ml4cvd/recipes.py \
#         --mode train \
#         --tensors /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_2500/ \
#         --input_tensors partners_ecg_age_newest partners_adult_gender_newest \
#         --output_tensors partners_ecg_bias_locationcardiology_oldest \
#         --train_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_2500/train.csv \
#         --valid_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_2500/valid.csv \
#         --test_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_2500/test.csv \
#         --training_steps 1096 --validation_steps 469 --test_steps 174 \
#         --batch_size 32 \
#         --epochs 5 \
#         --output_folder /home/paolo/mgh_mrns_to_extract/waveform_analysis \
#         --id 2500_age_sex
#     done
# done


# for normalization in _raw_
# do
#     for samples in 5000
#     do
#     /home/paolo/ml/scripts/tf.sh \
#     /home/paolo/ml/ml4cvd/recipes.py \
#         --mode train \
#         --tensors /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_5000/ \
#         --input_tensors partners_ecg_age_newest partners_adult_gender_newest \
#         --output_tensors partners_ecg_bias_locationcardiology_oldest \
#         --train_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_5000/train.csv \
#         --valid_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_5000/valid.csv \
#         --test_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_5000/test.csv \
#         --training_steps 389 --validation_steps 166 --test_steps 61 \
#         --batch_size 32 \
#         --epochs 5 \
#         --output_folder /home/paolo/mgh_mrns_to_extract/waveform_analysis \
#         --id 5000_age_sex
#     done
# done