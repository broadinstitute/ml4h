
./scripts/tf.sh /home/${USER}/ml/ml4cvd/recipes.py \
--mode train \
--input_tensors partners_ecg_geisinger_1 \
partners_ecg_geisinger_2 \
partners_ecg_geisinger_3 \
partners_ecg_geisinger_4 \
partners_ecg_geisinger_5 \
partners_ecg_sex \
partners_ecg_age_newest \
--dense_blocks \
--bottleneck_type global_average_pool \
--conv_layers 64 128 256 512 \
--conv_normalize batch_norm \
--aligned_dimension 512 \
--block_size 1 \
--pool_x 1 \
--output_tensors partners_ecg_bias_locationcardiology_oldest \
--sample_csv /home/paolo/mgh_mrns_to_extract/sample_csv_same_waveform.csv \
--tensors /home/paolo/mgh_mrns_to_extract/${hospital}_3yrs_hd5s/ \
--output_folder /home/paolo/recipes_output/ \
--id geisinger \
--inspect_model --conv_x 71 --epochs 50 --patience 10 --batch_size 2048 --test_steps 266 --validation_steps 532 --training_steps 1862

