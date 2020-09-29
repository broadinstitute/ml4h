pip install -e .
id_folder=ids_for_mgh
output_folder=ae_reconstructions
for latents in 32 80 128 176 224 272 320 368 416 464 512
do
	echo ------------------------------${latents}----------------------------------
	mkdir -p ${output_folder}/${latents}
	python plot_ae_recon.py --mode train --tensors /storage/shared/ecg/mgh --input_tensors partners_ecg_4096_ae --output_tensors partners_ecg_4096_shift --training_steps 64 --validation_steps 4 --test_steps 2 --num_workers 1 --test_csv ${id_folder}/valid2_ids.csv --train_csv ${id_folder}/train_ids.csv --valid_csv ${id_folder}/valid1_ids.csv --dense_blocks 16 44 72 100 128 --patience 5 --activation swish --pool_x 4 --inspect_model --model_file  train_runs/ae_deep_v0/ecg_ae_${latents}/ecg_ae_${latents}.h5 --output_folder ${output_folder}/${latents}
done
