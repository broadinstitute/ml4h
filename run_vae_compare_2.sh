pip install .
for kl_rate in 1e-11 1e-9 1e-7
do
	for latents in 8 16 32 64 96 128
	do
		python ml4cvd/recipes.py --mode train --tensors /storage/shared/ecg_deidentified/mgh_deidentified --input_tensors partners_ecg_5000_only_scale_1 --output_tensors partners_ecg_5000_only_scale_1 --training_steps 64 --validation_steps 4 --test_steps 50 --batch_size 512 --bottleneck_type variational --dense_layers $latents --epochs 50 --output_folder /storage/ndiamant/ml/train_runs/vae_compare_6 --id ecg_vae_kl_rate_${kl_rate}_latents_${latents} --learning_rate 1e-5 --cache_size 0 --conv_x 71 --num_workers 10 --test_csv valid2_ids.csv --train_csv train_ids.csv --valid_csv valid1_ids.csv --dense_blocks 16 32 48 --dense_regularize_rate $kl_rate --patience 1000 --activation swish
	done
done

