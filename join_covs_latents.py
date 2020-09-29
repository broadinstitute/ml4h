import os
import pandas as pd
import numpy as np


# infer_path = '/storage/ndiamant/ml/train_runs/vae_compare_kl_weight_high_lr/ecg_vae_kl_compare_hidden_infer/hidden_inference_ecg_vae_kl_compare_hidden_infer.tsv'
# infer_path = 'train_runs/simclr_compare_latents_1lead_augment/simclr_compare_hidden_infer/hidden_inference_simclr_compare_hidden_infer.tsv'
#infer_path = 'train_runs/vae_compare_kl_weight_high_lr_256/ecg_vae_kl_compare_hidden_infer/hidden_inference_ecg_vae_kl_compare_hidden_infer.tsv'
infer_path = 'train_runs/ae_deep_v0/ecg_ae_hidden_infer/hidden_inference_ecg_ae_hidden_infer.tsv'
#infer_path = 'train_runs/vae_deep_v1/ecg_vae_hidden_infer/hidden_inference_ecg_vae_hidden_infer.tsv'
#infer_path = 'train_runs/simclr_deep_v1/ecg_simclr_hidden_infer/hidden_inference_ecg_simclr_hidden_infer.tsv'
#infer_path = 'train_runs/supervised_deep_v0/ecg_supervised_hidden_infer/hidden_inference_ecg_supervised_hidden_infer.tsv'
cov_path = 'explorations/ecg_vae_cohort_09-15/tensors_all_union.csv'
valid_path = 'ids_for_mgh/valid2_ids.csv'

infer = pd.read_csv(infer_path, sep='\t')
print('Read', os.path.basename(infer_path))
cov = pd.read_csv(cov_path)
cov['sample_id'] = [int(os.path.basename(path).replace('.hd5', '')) for path in cov['fpath']]
valid = pd.read_csv(valid_path)
df = cov.merge(valid, on='sample_id')
df = df.merge(infer, on='sample_id')
df.to_csv('ae_deep_9-18.tsv', sep='\t', index=False)

