# T1 time from segmented regions

Test on an x86_64 Linux machine

"The Docker image is x86_64; on Apple Silicon Macs, TensorFlow's AVX requirement prevents it from running under Docker Desktop's emulation — demo must use an x86_64 machine, or Colima/QEMU." 

## Get public UK Biobank example experimental shMOLLI sequence image in DICOM format
mkdir ~/demo
cd ~/demo
curl -sL -o 20214_2_0.zip "https://biobank.ndph.ox.ac.uk/ukb/ukb/examples/20214_2_0.zip"
mkdir zip_folder
mkdir dicoms
mv 20214_2_0.zip zip_folder/1000001_20214_2_0.zip

# Get ml4h repo and the correct branch
cd ~
git clone https://github.com/broadinstitute/ml4h.git
git checkout --track origin/dfp_pap_gt_update_before_merge

## Tensorize
## Includes the initial Docker image pull so may take 5-15 minutes
cd ~/ml4h
./scripts/tensorize.sh -t $HOME/demo/inputs/ -n 1 -a " --zip_folder $HOME/demo/zip_folder/ --mri_field_ids 20214 --xml_field_ids --dicoms $HOME/demo/dicoms/ "

## Get the model

# if you don't have git lfs:
sudo apt-get install -y git-lfs 

cd ~/ml4h
git lfs pull --include="model_zoo/t1_time_from_segmented_regions/*.h5"

## 1st tutorial - run inference of the trained model
cd ~/ml4h
mkdir $HOME/demo/outputs
mkdir $HOME/demo/outputs/plot_predictions
mkdir $HOME/demo/outputs/medians_inference

# Dummy MRI date file required for median T1 times
sudo tee $HOME/demo/inputs/dates.csv > /dev/null << 'EOF'
sample_id,value
1000001,2022-03-04
EOF

# get median T1 times
./scripts/tf.sh -c $HOME/ml4h/ml4h/recipes.py  --num_workers 1 --mode infer_stats_from_segmented_regions --tensors $HOME/demo/inputs/ --input_tensors mri.t1map_b2 --output_tensors mri.t1map_b2_segmentation --encoder_blocks conv_encode --decoder_blocks unet_conv_decode --u_connect mri.t1map_b2 mri.t1map_b2_segmentation --activation swish --batch_size 1 --epochs 200 --valid_ratio 0 --test_ratio 1 --patience 50 --learning_rate 0.0003 --tensormap_prefix ml4h.tensormap.ukb --merge_blocks merge_conv_encode --conv_layers 32 32 32 --dense_blocks 64 128 256 --merge_dense_blocks 256 --decoder_dense_blocks 256 128 64 32 --merge_dimension 3 --dense_layers 0 --block_size 3 --app_csv $HOME/demo/inputs/dates.csv --structures_to_analyze interventricular_septum LV_free_wall anterolateral_pap posteromedial_pap LV_cavity interventricular_septum+LV_free_wall anterolateral_pap++posteromedial_pap --erosion_radius 1 1 0 0 1 1 --intensity_thresh_auto region_hist --intensity_thresh_in_structures anterolateral_pap posteromedial_pap --intensity_thresh_out_structure LV_cavity --intensity_thresh_auto_region_radius 5 --intensity_thresh_auto_clip_low 0.6580 --intensity_thresh_auto_clip_high 2.0181 --model_file $HOME/ml4h/model_zoo/t1_time_from_segmented_regions/batch8_lr0_0003_patience50.h5 --output_folder $HOME/demo/outputs/medians_inference/ --id public_model_infer

cat ~/demo/outputs/medians_inference/public_model_infer/pred_public_model_infer_input_shmolli_192i_sax_b2s_sax_b2s_sax_b2s_t1map_continuous_output_shmolli_192i_sax_b2s_sax_b2s_sax_b2s_t1map_vnauffal_annotated_categorical.tsv 
sample_id	interventricular_septum_mean	LV_free_wall_mean	anterolateral_pap_mean	posteromedial_pap_mean	LV_cavity_mean	interventricular_septum+LV_free_wall_mean	anterolateral_pap++posteromedial_pap_mean	interventricular_septum_median	LV_free_wall_median	anterolateral_pap_median	posteromedial_pap_median	LV_cavity_median	interventricular_septum+LV_free_wall_median	anterolateral_pap++posteromedial_pap_median	interventricular_septum_stLV_free_wall_std	anterolateral_pap_std	posteromedial_pap_std	LV_cavity_std	interventricular_septum+LV_free_wall_std	anterolateral_pap++posteromedial_pap_std	interventricular_septum_iqr	LV_free_wall_iqr	anterolateral_pap_iqr	posteromedial_pap_iqr	LV_cavity_iqr	interventricular_septum+LV_free_wall_iqr	anterolateral_pap++posteromedial_pap_iqr	interventricular_septum_count	LV_free_wall_count	anterolateral_pap_count	posteromedial_pap_count	LV_cavity_count	interventricular_septum+LV_free_wall_count	anterolateral_pap++posteromedial_pap_count	mri_date
1000001	945.6223564954682	960.2349272349272	1064.9285714285713	1030.4464285714287	1458.5428259683579	954.5983112183353	1045.2244897959183	939.0	955.0	1068.5	1021.5	1538.0	947.0	1040.0	47.171468236583074	68.8405076627488	44.97533337558375	60.927626751175964	171.16928972125166	61.29291768472139	57.265498857807664	57.0	89.0	63.25	82.0	220.0	73.0	77.0	331.0	481.0	42.0	56.0	1833.0	829.0	98.0	2022-03-04

# visualize the segmentation
./scripts/tf.sh -c $HOME/ml4h/ml4h/recipes.py  --num_workers 1 --mode plot_predictions --tensors $HOME/demo/inputs/ --input_tensors mri.t1map_b2 --output_tensors mri.t1map_b2_segmentation --encoder_blocks conv_encode --decoder_blocks unet_conv_decode --u_connect mri.t1map_b2 mri.t1map_b2_segmentation --activation swish --batch_size 1 --epochs 200 --valid_ratio 0 --test_ratio 1 --patience 100000 --learning_rate 0.0003 --tensormap_prefix ml4h.tensormap.ukb --inspect_model --merge_blocks merge_conv_encode --conv_layers 32 32 32 --dense_blocks 64 128 256 --merge_dense_blocks 256 --decoder_dense_blocks 256 128 64 32 --merge_dimension 3 --dense_layers 0 --block_size 3 --rotation_factor 0.014 --zoom_factor 0.05 --translation_factor 0.042 --model_file $HOME/ml4h/model_zoo/t1_time_from_segmented_regions/batch8_lr0_0003_patience50.h5 --output_folder $HOME/demo/outputs/plot_predictions/ --id public_model_infer