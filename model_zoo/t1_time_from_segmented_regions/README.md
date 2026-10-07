# T1 time from segmented regions

Test on an x86_64 Linux machine

"The Docker image is x86_64; on Apple Silicon Macs, TensorFlow's AVX requirement prevents it from running under Docker Desktop's emulation — demo must use an x86_64 machine, or Colima/QEMU." 

## Directory setup
mkdir ~/demo
mkdir ~/demo/zip_folder
mkdir ~/demo/dicoms
mkdir ~/demo/csvs
mkdir ~/demo/outputs
mkdir ~/demo/outputs/model
mkdir ~/demo/outputs/plot_predictions
mkdir ~/demo/outputs/medians_inference
mkdir ~/demo/outputs/compare

## Get public UK Biobank example experimental shMOLLI sequence image in DICOM format
curl -sL -o ~/demo/zip_folder/1000001_20214_2_0.zip "https://biobank.ndph.ox.ac.uk/ukb/ukb/examples/20214_2_0.zip"

## UK Biobank provides one publicly downloadable example image. The pipeline needs distinct train/validation/test samples, so we make copies under different sample IDs.
cp ~/demo/zip_folder/1000001_20214_2_0.zip ~/demo/zip_folder/1000002_20214_2_0.zip
cp ~/demo/zip_folder/1000001_20214_2_0.zip ~/demo/zip_folder/1000003_20214_2_0.zip

# Setup splits
cat > ~/demo/csvs/all.csv << 'EOF'
sample_id
1000001
1000002
1000003
EOF

cat > ~/demo/csvs/train.csv << 'EOF'
sample_id
1000001
EOF

cat > ~/demo/csvs/valid.csv << 'EOF'
sample_id
1000002
EOF

cat > ~/demo/csvs/test.csv << 'EOF'
sample_id
1000003
EOF


# Get ml4h repo and the correct branch
cd ~
git clone https://github.com/broadinstitute/ml4h.git
cd ~/ml4h
git checkout --track origin/dfp_pap_gt_update_before_merge

## Get the model

# if you don't have git lfs:
sudo apt-get install -y git-lfs 
git lfs install

cd ~/ml4h
git lfs pull --include="model_zoo/t1_time_from_segmented_regions/*.h5"

## Tensorize
## Includes the initial Docker image pull so may take 5-15 minutes the first time
cd ~/ml4h
./scripts/tensorize.sh -t $HOME/demo/inputs/ -n 1 -a " --zip_folder $HOME/demo/zip_folder/ --mri_field_ids 20214 --xml_field_ids --dicoms $HOME/demo/dicoms/ "

## 1st tutorial - run inference of the trained model

# visualize the segmentation
cd ~/ml4h
./scripts/tf.sh -c $HOME/ml4h/ml4h/recipes.py  --num_workers 1 --mode plot_predictions --tensors $HOME/demo/inputs/ --input_tensors mri.t1map_b2 --output_tensors mri.t1map_b2_segmentation --encoder_blocks conv_encode --decoder_blocks unet_conv_decode --u_connect mri.t1map_b2 mri.t1map_b2_segmentation --activation swish --batch_size 1 --epochs 200 --training_steps 1 --validation_steps 1 --test_steps 1 --patience 50 --learning_rate 0.0003 --tensormap_prefix ml4h.tensormap.ukb --inspect_model --sample_csv ~/demo/csvs/all.csv --train_csv ~/demo/csvs/train.csv --valid_csv ~/demo/csvs/valid.csv --test_csv ~/demo/csvs/test.csv --merge_blocks merge_conv_encode --conv_layers 32 32 32 --dense_blocks 64 128 256 --merge_dense_blocks 256 --decoder_dense_blocks 256 128 64 32 --merge_dimension 3 --dense_layers 0 --block_size 3 --skip_ground_truth --model_file $HOME/ml4h/model_zoo/t1_time_from_segmented_regions/batch8_lr0_0003_patience50.h5 --output_folder $HOME/demo/outputs/plot_predictions/ --id public_model_infer

# resulting images are in ~/demo/outputs/plot_predictions/public_model_infer/prediction_pngs

# Dummy MRI date file required for median T1 times
sudo tee ~/demo/csvs/dates.csv > /dev/null << 'EOF'
sample_id,value
1000001,2026-01-01
1000002,2026-01-01
1000003,2026-01-01
EOF

# get median T1 times
cd ~/ml4h
./scripts/tf.sh -c $HOME/ml4h/ml4h/recipes.py --num_workers 1 --mode infer_stats_from_segmented_regions --tensors $HOME/demo/inputs/ --input_tensors mri.t1map_b2 --output_tensors mri.t1map_b2_segmentation --encoder_blocks conv_encode --decoder_blocks unet_conv_decode --u_connect mri.t1map_b2 mri.t1map_b2_segmentation --activation swish --batch_size 1 --epochs 200 --training_steps 1 --validation_steps 1 --test_steps 1  --patience 50 --learning_rate 0.0003 --tensormap_prefix ml4h.tensormap.ukb --sample_csv ~/demo/csvs/all.csv --train_csv ~/demo/csvs/train.csv --valid_csv ~/demo/csvs/valid.csv --test_csv ~/demo/csvs/test.csv --merge_blocks merge_conv_encode --conv_layers 32 32 32 --dense_blocks 64 128 256 --merge_dense_blocks 256 --decoder_dense_blocks 256 128 64 32 --merge_dimension 3 --dense_layers 0 --block_size 3 --app_csv $HOME/demo/csvs/dates.csv --structures_to_analyze interventricular_septum LV_free_wall anterolateral_pap posteromedial_pap LV_cavity interventricular_septum+LV_free_wall anterolateral_pap++posteromedial_pap --erosion_radius 1 1 0 0 1 1 --intensity_thresh_auto region_hist --intensity_thresh_in_structures anterolateral_pap posteromedial_pap --intensity_thresh_out_structure LV_cavity --intensity_thresh_auto_region_radius 5 --intensity_thresh_auto_clip_low 0.6580 --intensity_thresh_auto_clip_high 2.0181 --model_file $HOME/ml4h/model_zoo/t1_time_from_segmented_regions/batch8_lr0_0003_patience50.h5 --output_folder $HOME/demo/outputs/medians_inference/ --id public_model_infer

cat ~/demo/outputs/medians_inference/public_model_infer/pred_public_model_infer_input_shmolli_192i_sax_b2s_sax_b2s_sax_b2s_t1map_continuous_output_shmolli_192i_sax_b2s_sax_b2s_sax_b2s_t1map_vnauffal_annotated_categorical.tsv 
sample_id	interventricular_septum_mean	LV_free_wall_mean	anterolateral_pap_mean	posteromedial_pap_mean	LV_cavity_mean	interventricular_septum+LV_free_wall_mean	anterolateral_pap++posteromedial_pap_mean	interventricular_septum_median	LV_free_wall_median	anterolateral_pap_median	posteromedial_pap_median	LV_cavity_median	interventricular_septum+LV_free_wall_median	anterolateral_pap++posteromedial_pap_median	interventricular_septum_std LV_free_wall_std	anterolateral_pap_std	posteromedial_pap_std	LV_cavity_std	interventricular_septum+LV_free_wall_std	anterolateral_pap++posteromedial_pap_std	interventricular_septum_iqr	LV_free_wall_iqr	anterolateral_pap_iqr	posteromedial_pap_iqr	LV_cavity_iqr	interventricular_septum+LV_free_wall_iqr	anterolateral_pap++posteromedial_pap_iqr	interventricular_septum_count	LV_free_wall_count	anterolateral_pap_count	posteromedial_pap_count	LV_cavity_count	interventricular_septum+LV_free_wall_count	anterolateral_pap++posteromedial_pap_count	mri_date
1000003	945.6223564954682	960.2349272349272	1064.9285714285713	1030.4464285714287	1458.5428259683579	954.5983112183353	1045.2244897959183	939.0	955.0	1068.5	1021.5	1538.0	947.0	1040.0	47.171468236583074	68.8405076627488	44.97533337558375	60.927626751175964	171.16928972125166	61.29291768472139	57.265498857807664	57.0	89.0	63.25	82.0	220.0	73.0	77.0	331.0	481.0	42.0	56.0	1833.0	829.0	98.0	2026-01-01

# Second demo - quickly train and test a model. Since train/valid/test are identical copies, this demo is for pipeline mechanics and not generalization (plus the ground truth is model-generated).

# We need a ground truth segmentation for training - use our model's output from plot_predictions. But first we need to reverse-engineer the labels from the png values, create 3 pngs for our train/valid/test images, and tensorize the resulting pngs
cd ~/ml4h
./scripts/tf.sh -c $HOME/ml4h/model_zoo/t1_time_from_segmented_regions/convert_prediction_pngs_to_masks.py --predictions $HOME/demo/outputs/plot_predictions/public_model_infer/prediction_pngs/ --output_folder $HOME/demo/pseudo_gt_pngs/

sudo cp $HOME/demo/pseudo_gt_pngs/1000003_t1map.png.mask.png $HOME/demo/pseudo_gt_pngs/1000001_t1map.png.mask.png
sudo cp $HOME/demo/pseudo_gt_pngs/1000003_t1map.png.mask.png $HOME/demo/pseudo_gt_pngs/1000002_t1map.png.mask.png

printf '%s\t%s\t%s\n' \
  sample_id dicom_file instance_number \
  1000001 1000001_t1map 1 \
  1000002 1000002_t1map 1 \
  1000003 1000003_t1map 1 \
  | sudo tee ~/demo/csvs/manifest.tsv > /dev/null

./scripts/tensorize.sh -m tensorize_pngs -t $HOME/demo/inputs/ -n 1 -a " --dicoms $HOME/demo/pseudo_gt_pngs/ --app_csv $HOME/demo/csvs/manifest.tsv --dicom_series shmolli_192i_sax_b2s_sax_b2s_sax_b2s_t1map_vnauffal --x 384 --y 384"

# Train (since this is a demo on a CPU machine to demonstrate pipeline mechanics, we optimize for training speed. Use batch_size of 1, low training_steps, low # epochs, high learning rate, and no augmentation)
cd ~/ml4h
epochs=20
lr=0.001
./scripts/tf.sh -c $HOME/ml4h/ml4h/recipes.py  --num_workers 1 --mode train --tensors $HOME/demo/inputs/ --input_tensors mri.t1map_b2 --output_tensors mri.t1map_b2_segmentation --encoder_blocks conv_encode --decoder_blocks unet_conv_decode --u_connect mri.t1map_b2 mri.t1map_b2_segmentation --activation swish --batch_size 1 --epochs $epochs --training_steps 16 --validation_steps 1 --test_steps 1 --patience 50 --learning_rate $lr --tensormap_prefix ml4h.tensormap.ukb --inspect_model --sample_csv ~/demo/csvs/all.csv --train_csv ~/demo/csvs/train.csv --valid_csv ~/demo/csvs/valid.csv --test_csv ~/demo/csvs/test.csv --merge_blocks merge_conv_encode --conv_layers 32 32 32 --dense_blocks 64 128 256 --merge_dense_blocks 256 --decoder_dense_blocks 256 128 64 32 --merge_dimension 3 --dense_layers 0 --block_size 3 --output_folder $HOME/demo/outputs/model/ --id overfit_single_image

# plot predictions

./scripts/tf.sh -c $HOME/ml4h/ml4h/recipes.py  --num_workers 1 --mode plot_predictions --tensors $HOME/demo/inputs/ --input_tensors mri.t1map_b2 --output_tensors mri.t1map_b2_segmentation --encoder_blocks conv_encode --decoder_blocks unet_conv_decode --u_connect mri.t1map_b2 mri.t1map_b2_segmentation --activation swish --batch_size 1 --epochs $epochs --training_steps 1 --validation_steps 1 --test_steps 1 --patience 50 --learning_rate $lr --tensormap_prefix ml4h.tensormap.ukb --inspect_model --sample_csv ~/demo/csvs/all.csv --train_csv ~/demo/csvs/train.csv --valid_csv ~/demo/csvs/valid.csv --test_csv ~/demo/csvs/test.csv --merge_blocks merge_conv_encode --conv_layers 32 32 32 --dense_blocks 64 128 256 --merge_dense_blocks 256 --decoder_dense_blocks 256 128 64 32 --merge_dimension 3 --dense_layers 0 --block_size 3 --model_file $HOME/demo/outputs/model/overfit_single_image/overfit_single_image.h5 --output_folder $HOME/demo/outputs/plot_predictions/ --id overfit_single_image

# dice scores (compare)

./scripts/tf.sh -c $HOME/ml4h/ml4h/recipes.py  --num_workers 1 --mode compare --tensors $HOME/demo/inputs/ --input_tensors mri.t1map_b2 --output_tensors mri.t1map_b2_segmentation --encoder_blocks conv_encode --decoder_blocks unet_conv_decode --u_connect mri.t1map_b2 mri.t1map_b2_segmentation --activation swish --batch_size 1 --epochs $epochs --training_steps 1 --validation_steps 1 --test_steps 1 --patience 50 --learning_rate $lr --tensormap_prefix ml4h.tensormap.ukb --inspect_model --sample_csv ~/demo/csvs/all.csv --train_csv ~/demo/csvs/train.csv --valid_csv ~/demo/csvs/valid.csv --test_csv ~/demo/csvs/test.csv --merge_blocks merge_conv_encode --conv_layers 32 32 32 --dense_blocks 64 128 256 --merge_dense_blocks 256 --decoder_dense_blocks 256 128 64 32 --merge_dimension 3 --dense_layers 0 --block_size 3 --model_files $HOME/demo/outputs/model/overfit_single_image/overfit_single_image.h5 --output_folder $HOME/demo/outputs/compare/ --id overfit_single_image