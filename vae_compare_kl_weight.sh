pip install .
latents=96
#epochs=50
epochs=2
for repetition in 0 1
do
	for kl_rate in 1e-9 1e-8 1e-7 1e-6 1e-5 1e-4
	do
		python ml4cvd/recipes.py --mode train --tensors /storage/shared/ecg_deidentified/mgh_deidentified --input_tensors partners_ecg_5000_only_scale_1 --output_tensors partners_ecg_5000_only_scale_1 --training_steps 64 --validation_steps 4 --test_steps 50 --batch_size 512 --bottleneck_type variational --dense_layers $latents --epochs $epochs --output_folder /storage/ndiamant/ml/train_runs/vae_compare_kl_weight --id repetition_${repetition}_ecg_vae_kl_rate_${kl_rate}_latents_${latents} --learning_rate 1e-5 --cache_size 0 --conv_x 71 --num_workers 10 --test_csv valid2_ids.csv --train_csv train_ids.csv --valid_csv valid1_ids.csv --dense_blocks 16 32 48 --dense_regularize_rate $kl_rate --patience 1000 --activation swish
	done
done

