./scripts/tf.sh -d $GPU -m /media -t \
    $HOME/ml_working/ml4cvd/recipes.py \
    --mode train \
    --logging_level INFO \
    --tensors /data/partners_ecg/mgh/hd5 \
    --train_csv /data/sts-data/bootstrap/$BOOTSTRAP/train.csv \
    --valid_csv /data/sts-data/bootstrap/$BOOTSTRAP/valid.csv \
    --test_csv /data/sts-data/bootstrap/$BOOTSTRAP/test.csv \
    --input_tensors \
        partners_ecg_2500_std_newest_sts \
    --output_tensors \
        sts_death \
    --inspect_model \
    --conv_layers \
    --dense_blocks 16 32 64 128 256 512 \
    --block_size 1 \
    --conv_x 80 \
    --pool_type max \
    --pool_x 2 \
    --bottleneck_type global_average_pool \
    --dense_layers 16 64 \
    --dropout 0.0 \
    --optimizer adam \
    --epochs 1 \
    --patience 8 \
    --batch_size 64 \
    --training_steps 400 \
    --validation_steps 60 \
    --test_steps 32 \
    --output_folder $HOME/sts-ecg/train-varied-filters-test-plot \
    --id $BOOTSTRAP \
    --conv_normalize batch_norm \
    --plot_train_curves
