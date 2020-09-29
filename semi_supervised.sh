pip install -e .
id_folder=ids_for_mgh
output_folder=semi_supervised
train_folder=train_runs
#epochs=200
#training_steps=64
#validation_steps=4
#num_workers=10
epochs=200
training_steps=4
validation_steps=1
num_workers=0
conv_x=16
for latents in 32 80 128 176 224 272 320 368 416 464 512
do
	model_ids=""
	model_files=""
	for pair in "ecg_vae,vae_deep_v1" "ecg_simclr,simclr_deep_v1" "ecg_supervised,supervised_deep_v0" "ecg_ae,ae_deep_v0"
	do
		id=`echo $pair | cut -d',' -f1`
		id=${id}_${latents}
		folder=`echo $pair | cut -d',' -f2`
		model_file=${train_folder}/${folder}/${id}/${id}.h5

		#for reduced_id in ids_percent_5 ids_percent_10 ids_percent_100
		for reduced_id in ids_percent_1 ids_percent_2
		do
			reduced_id_folder=${id_folder}/${reduced_id}
			model_ids="${model_ids} ${id}_${reduced_id}"
			model_files="${model_files} train_runs/${output_folder}/${id}_${reduced_id}/${id}_${reduced_id}.h5"

			python ml4cvd/recipes.py --mode train --tensors /storage/shared/ecg/mgh --input_tensors partners_ecg_4096_random --output_tensors standardized_partners_ecg_age_newest --training_steps $training_steps --validation_steps $validation_steps --test_steps 1 --batch_size 512 --bottleneck_type flatten_restructure --dense_layers $latents --epochs $epochs --output_folder /storage/ndiamant/ml/train_runs/${output_folder} --id ${id}_${reduced_id}  --learning_rate 1e-3 --cache_size 0 --conv_x ${conv_x} --num_workers 0 --test_csv ${id_folder}/valid2_ids.csv --train_csv ${reduced_id_folder}/train_ids.csv --valid_csv ${reduced_id_folder}/valid1_ids.csv --dense_blocks 16 44 72 100 128 --patience 5 --activation swish --pool_x 4 --inspect_model --model_layers ${model_file} | tee stdout.txt
			pkill python
		done
	done
	python ml4cvd/recipes.py --mode infer --tensors /storage/shared/ecg/mgh --input_tensors partners_ecg_4096 --output_tensors standardized_partners_ecg_age_newest --batch_size 512 --output_folder /storage/ndiamant/ml/train_runs/${output_folder} --id semi_supervised_${latents} --num_workers 10 --test_csv ${id_folder}/valid2_ids.csv --model_files ${model_files} --model_ids ${model_ids}
done
