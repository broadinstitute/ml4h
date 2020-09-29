import os
import pandas as pd
import numpy as np


output_folder = 'ids_for_mgh'
df = pd.read_csv('/storage/ndiamant/ml/explorations/ecg_vae_cohort_09-15/tensors_all_union.csv')
non_error_cols = [col for col in df.columns if 'error' not in col]
df = df.dropna(subset=non_error_cols)
sample_ids = np.array([os.path.basename(p).replace('.hd5', '') for p in df['fpath']])
print(sample_ids[:10])
np.random.shuffle(sample_ids)
print(sample_ids[:10])
data_sets = np.split(sample_ids, [int(frac * len(sample_ids)) for frac in (.7, .8, .9)])
data_names = ['train_ids.csv', 'valid1_ids.csv', 'valid2_ids.csv', 'test_ids.csv',]
for name, data_set in zip(data_names, data_sets):
    print(name, len(data_set))
    pd.DataFrame({'sample_id': data_set}).to_csv(os.path.join(output_folder, name), index=False)
