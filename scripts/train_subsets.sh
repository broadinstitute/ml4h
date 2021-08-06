ECHO=
MODEL_FILES=
TENSORS="/mnt/disks/annotated-cardiac-tensors-45k-2021-03-25/2020-09-21/"
TENSOR_MAPS="ecg.ecg_rest_median_raw_10 mri.lax_4ch_heart_center "
array=( "drop_fuse_unsupervised_train_64.csv" "drop_fuse_unsupervised_train_128.csv" "drop_fuse_unsupervised_train_256.csv" "drop_fuse_unsupervised_train_512.csv" "drop_fuse_unsupervised_train_1024.csv" "drop_fuse_unsupervised_train_2048.csv" "drop_fuse_unsupervised_train_4096.csv")
for i in "${array[@]}"
do
    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode train_block \
    --tensors "$TENSORS" --input_tensors "$TENSOR_MAPS" --output_tensors "$TENSOR_MAPS" \
    --encoder_blocks /home/sam/trained_models/hypertuned_48m_16e_ecg_median_raw_10_autoencoder_256d/encoder_ecg_rest_median_raw_10.h5 \
        /home/sam/trained_models/hypertuned_64m_18e_lax_4ch_heart_center_autoencoder_256d/encoder_lax_4ch_heart_center.h5 \
    --merge_blocks pair \
    --decoder_blocks /home/sam/trained_models/hypertuned_48m_16e_ecg_median_raw_10_autoencoder_256d/decoder_ecg_rest_median_raw_10.h5 \
        /home/sam/trained_models/hypertuned_64m_18e_lax_4ch_heart_center_autoencoder_256d/decoder_lax_4ch_heart_center.h5 \
    --pairs "$TENSOR_MAPS" --pair_loss contrastive --pair_loss_weight 0.1 --pair_merge dropout \
    --batch_size 4 --epochs 1 --training_steps 128 --validation_steps 32 --test_steps 1 \
    --num_workers 4 --patience 16 --tensormap_prefix ml4h.tensormap.ukb \
    --id "drop_fuse_early_stop_v3_${i%.*}" --output_folder /home/sam/trained_models/ \
    --inspect_model --activation mish --dense_layers 256 \
    --train_csv "/home/sam/csvs/${i}" \
    --valid_csv /home/sam/csvs/drop_fuse_unsupervised_valid.csv \
    --test_csv /home/sam/csvs/sample_id_returned_lv_mass.csv


    $ECHO ./scripts/tf.sh /home/sam/ml4h/ml4h/recipes.py --mode infer_encoders \
    --tensors "$TENSORS" --input_tensors "$TENSOR_MAPS" --output_tensors "$TENSOR_MAPS" \
    --model_file "/home/sam/trained_models/drop_fuse_early_stop_v3_${i%.*}/drop_fuse_early_stop_v3_${i%.*}.h5" \
    --id "drop_fuse_early_stop_v3_${i%.*}" --output_folder /home/sam/trained_models/ \
    --sample_csv /home/sam/csvs/sample_id_returned_lv_mass.csv \
    --tensormap_prefix ml4h.tensormap.ukb \
    --dense_layers 256
done
