for hospital in mgh
do
    # for output in I_len I_nonzero acquisitionyear acquisitionsoftwareversion acquisitiondevice locationname testreason
    for output in I_len
    do
	    for j in 2500 5000
	    do
	        sbatch --time 01:00:00 --job-name=ml4cvd_${j}_${output} \
                logs/ml/scripts/tf_cluster.sh -b 2017P001650/mgh_3yrs \
                /home/paolo.achille/ml/ml4cvd/recipes.py \
                    --mode train \
                    --num_workers 20 \
                    --input_tensors partners_ecg_${j}_raw_oldest \
                    --output_tensors partners_ecg_bias_${output}_oldest \
                    --output_folder /home/paolo.achille/recipes_output/ \
                    --id ${hospital}_partners_ecg_${j}_raw_oldest__partners_ecg_bias_${output}_oldest \
                    --conv_x 71 --epochs 16 --batch_size 32 --test_steps 266 --validation_steps 532 \
                    --training_steps 1862
	    done	
    done
done
