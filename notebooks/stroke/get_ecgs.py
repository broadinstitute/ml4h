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
from ml4cvd.tensor_maps_partners_ecg import TMAPS

hd5 = h5py.File('/home/paolo/stroke_ecgs/1004775.hd5', 'r')
tm = TMAPS['toast_subtype']
tensor = tm.tensor_from_file(tm, hd5)
hd5.close()