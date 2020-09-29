pip install .
id_folder=ids_for_mgh
output_folder=train_runs/simclr_deep_v1
for latents in 32 80 128 176 224 272 320 368 416 464 512
do
	model_id="ecg_simclr_${latents}"
	model_ids+="${model_id} "
	model_files+="${output_folder}/${model_id}/${model_id}.h5 "
done

python ml4cvd/recipes.py --mode infer_hidden --tensors /storage/shared/ecg/mgh --input_tensors partners_ecg_4096 --output_tensors partners_ecg_4096 --batch_size 512 --output_folder /storage/ndiamant/ml/${output_folder} --id ecg_simclr_hidden_infer --num_workers 10 --test_csv ${id_folder}/valid2_ids.csv --model_files ${model_files} --model_ids ${model_ids} --logging_level DEBUG
