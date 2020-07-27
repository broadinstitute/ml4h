for hospital in mgh
do
    #for output in I_len I_nonzero acquisitionyear acquisitionsoftwareversion acquisitiondevice locationname testreason
    for output in locationcardiology
    do
	for j in 2500 5000
	do
	    ./scripts/tf.sh /home/paolo/ml/ml4cvd/recipes.py \
		    --mode train \
		    --input_tensors partners_ecg_${j}_raw_oldest \
		    --output_tensors partners_ecg_bias_${output}_oldest \
			--sample_csv /home/paolo/mgh_mrns_to_extract/sample_csv_same_waveform.csv \
		    --tensors /home/paolo/mgh_mrns_to_extract/${hospital}_3yrs_hd5s/ \
		    --output_folder /home/paolo/recipes_output/ \
		    --id ${hospital}_partners_ecg_${j}_raw_oldest__partners_ecg_bias_${output}_oldest \
		    --inspect_model --conv_x 71 --epochs 12 --batch_size 32 --test_steps 266 --validation_steps 532 --training_steps 1862

	    # ./scripts/tf.sh /home/paolo/ml/ml4cvd/recipes.py \
		#     --mode train \
		#     --input_tensors partners_ecg_${j}_oldest \
		#     --output_tensors partners_ecg_bias_${output}_oldest \
		#     --tensors /home/paolo/mgh_mrns_to_extract/${hospital}_3yrs_hd5s/ \
		#     --output_folder /home/paolo/recipes_output/ \
		#     --id ${hospital}_partners_ecg_${j}_oldest__partners_ecg_bias_${output}_oldest \
		#     --inspect_model --conv_x 71 --epochs 16 --batch_size 32 --test_steps 266 --validation_steps 532 --training_steps 1862
	done	
    done
done
