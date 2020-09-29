pip install .
epochs=200
kl_rate=4e-5
id_folder=ids_for_mgh
output_folder=vae_deep_v2
conv_x=16
for latents in 32 80 128 176 224 272 320 368 416 464 512
do
	python ml4cvd/recipes.py --mode train --tensors /storage/shared/ecg/mgh --input_tensors partners_ecg_4096_shift --output_tensors partners_ecg_4096_shift --training_steps 64 --validation_steps 4 --test_steps 50 --batch_size 512 --bottleneck_type variational --dense_layers $latents --epochs $epochs --output_folder /storage/ndiamant/ml/train_runs/${output_folder} --id ecg_vae_${latents} --learning_rate 1e-4 --cache_size 0 --conv_x ${conv_x} --num_workers 5 --test_csv ${id_folder}/valid2_ids.csv --train_csv ${id_folder}/train_ids.csv --valid_csv ${id_folder}/valid1_ids.csv --dense_blocks 16 44 72 100 128 --dense_regularize_rate $kl_rate --patience 1000 --activation swish --pool_x 4 --inspect_model
	pkill python
done
