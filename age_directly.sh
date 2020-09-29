pip install .
epochs=200
id_folder=ids_for_mgh
output_folder=age_directly
train_folder=train_runs
conv_x=16
#for latents in 32 80 128 176 224 272 320 368 416 464 512
for latents in 80 128 176 224 272 320 368 416 464 512
do
	model_ids=""
	model_files=""
	id=ecg_age_${latents}
	for reduced_id in ids_percent_5 ids_percent_10 ids_percent_100
	do
		reduced_id_folder=${id_folder}/${reduced_id}
		model_ids="${model_ids} ${id}_${reduced_id}"
		model_files="${model_files} train_runs/${output_folder}/${id}_${reduced_id}/${id}_${reduced_id}.h5"

		python ml4cvd/recipes.py --mode train --tensors /storage/shared/ecg/mgh --input_tensors partners_ecg_4096_random --output_tensors standardized_partners_ecg_age_newest --training_steps 64 --validation_steps 4 --test_steps 1 --batch_size 512 --bottleneck_type flatten_restructure --dense_layers $latents --epochs $epochs --output_folder /storage/ndiamant/ml/train_runs/${output_folder} --id ${id}_${reduced_id}  --learning_rate 1e-3 --cache_size 0 --conv_x ${conv_x} --num_workers 10 --test_csv ${id_folder}/valid2_ids.csv --train_csv ${reduced_id_folder}/train_ids.csv --valid_csv ${reduced_id_folder}/valid1_ids.csv --dense_blocks 16 44 72 100 128 --patience 5 --activation swish --pool_x 4 --inspect_model
		pkill python
	done
	python ml4cvd/recipes.py --mode infer --tensors /storage/shared/ecg/mgh --input_tensors partners_ecg_4096 --output_tensors standardized_partners_ecg_age_newest --batch_size 512 --output_folder /storage/ndiamant/ml/train_runs/${output_folder} --id infer_${latents} --num_workers 10 --test_csv ${id_folder}/valid2_ids.csv --model_files ${model_files} --model_ids ${model_ids}
done
