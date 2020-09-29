import os
import pandas as pd
import numpy as np


id_folder = 'ids_for_mgh'
train_ids = pd.read_csv(os.path.join(id_folder, 'train_ids.csv')).sample(frac=1)
valid1_ids = pd.read_csv(os.path.join(id_folder, 'valid1_ids.csv')).sample(frac=1)
valid2_ids = pd.read_csv(os.path.join(id_folder, 'valid2_ids.csv')).sample(frac=1)


for frac in (.05, .1, 1):
    output_folder = os.path.join(id_folder, f'ids_percent_{int(frac * 100)}')
    os.makedirs(output_folder, exist_ok=True)
    train_ids.head(int(len(train_ids) * frac)).to_csv(os.path.join(output_folder, 'train_ids.csv'), index=False)
    valid1_ids.head(max(512, int(len(valid1_ids) * frac))).to_csv(os.path.join(output_folder, 'valid1_ids.csv'), index=False)
    valid2_ids.head(max(512, int(len(valid2_ids) * frac))).to_csv(os.path.join(output_folder, 'valid2_ids.csv'), index=False)

