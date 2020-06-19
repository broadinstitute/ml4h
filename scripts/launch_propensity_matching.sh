#!usr/bin/bash
for match in _I_len
do
for normalization in '' _raw_
do
for samples in 2500 5000
do
/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched_{bias_keys}/
/home/paolo/ml/scripts/tf.sh \
    /home/paolo/ml/ml4cvd/recipes.py \
        --mode train \
        --tensors /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched${match}/ \
        --input_tensors partners_ecg_${samples}${raw}_oldest \
        --output_tensors partners_ecg_bias_locationcardiology_oldest \
        --train_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched${match}/train.csv \
        --valid_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched${match}/valid.csv \
        --test_csv /home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched${match}/test.csv \
        --training_steps 466 --validation_steps 200 --test_steps 74 \
        --batch_size 32 --conv_x 36 \
        --epochs 5 \
        --output_folder /home/paolo/mgh_mrns_to_extract/propensity_matching \
        --id ${samples}${normalization}${match}
done
done
done

