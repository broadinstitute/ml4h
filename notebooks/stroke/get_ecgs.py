# %%
import shutil
import pandas as pd
import os
registry = pd.read_csv('stroke_registry_mrn_mapped_toast_subtype-062020.txt', sep='\t')

# %%
for i, patient in registry.iterrows():
    if os.path.isfile(f'/data/partners_ecg/mgh/{int(patient["MRN"])}.hd5'):
        shutil.copyfile(f'/data/partners_ecg/mgh/{int(patient["MRN"])}.hd5', f'/home/paolo/stroke_ecgs/{int(patient["MRN"])}.hd5')


# %%
