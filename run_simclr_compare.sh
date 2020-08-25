pip install .
epochs=100
id_folder=ids_for_mgh
folder=simclr_compare_latents_1lead_augment
#tmap=partners_ecg_5000_only_random_lead
tmap=partners_ecg_5000_only_random_lead_augment
for latents in 64 96 128 192 256
do
	python ml4cvd/recipes.py --mode train_simclr --tensors /storage/shared/ecg/mgh --input_tensors ${tmap} --training_steps 64 --validation_steps 4 --test_steps 50 --batch_size 128 --bottleneck_type flatten_restructure --dense_layers $latents --epochs $epochs --output_folder /storage/ndiamant/ml/train_runs/${folder} --id simclr_latents_${latents} --learning_rate 1e-4 --cache_size 0 --conv_x 71 --num_workers 10 --test_csv ${id_folder}/valid2_ids.csv --train_csv ${id_folder}/train_ids.csv --valid_csv ${id_folder}/valid1_ids.csv --dense_blocks 16 32 48 --patience 1000 --activation swish
done
