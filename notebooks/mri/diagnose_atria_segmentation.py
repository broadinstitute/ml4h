# %%
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

df = pd.read_csv('/home/pdiachil/ml/notebooks/mri/cleaned_poisson.csv', sep='\t')
idxs = np.argsort(df['error'])
keys = [f'LA_poisson_{d}' for d in range(50)]
petersen_max = df['LA_Biplan_vol_max'].values
petersen_min = df['LA_Biplan_vol_min'].values
err = df['error']

# %%
from ml4cvd.defines import MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES
from ml4cvd.tensor_from_file import _mri_tensor_4d
import imageio
MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP = {'left_atrium': 11}
annot_vals = [MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
              MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
              MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']]
views = ['2ch', '3ch', '4ch']

import h5py
for i in sys.argv[1:]:
    print(i)
    idx = idxs[int(i)]
    fpath = f'/mnt/disks/segmented-sax-lax/2020-06-26/{df.iloc[idx]["sample_id"]}.hd5'
    print(fpath)
    if not os.path.isfile(fpath): 
        continue
    with h5py.File(fpath, 'r') as ff:
        arr_2ch = _mri_tensor_4d(ff, 'cine_segmented_lax_2ch', dest_shape=[256, 256], concatenate=False)
        arr_3ch = _mri_tensor_4d(ff, 'cine_segmented_lax_3ch', dest_shape=[256, 256], concatenate=False)
        arr_4ch = _mri_tensor_4d(ff, 'cine_segmented_lax_4ch', dest_shape=[256, 256], concatenate=False)
        arr_annot_2ch = _mri_tensor_4d(ff, 'cine_segmented_lax_2ch_annotated_', concatenate=True)
        arr_annot_3ch = _mri_tensor_4d(ff, 'cine_segmented_lax_3ch_annotated_', concatenate=True)
        arr_annot_4ch = _mri_tensor_4d(ff, 'cine_segmented_lax_4ch_annotated_', concatenate=True)
        arr_annot_2ch_ma = np.ma.masked_array(data=arr_annot_2ch, 
                                              mask=(arr_annot_2ch != MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium']))       
        arr_annot_3ch_ma = np.ma.masked_array(data=arr_annot_3ch, 
                                              mask=(arr_annot_3ch != MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'])) 
        arr_annot_4ch_ma = np.ma.masked_array(data=arr_annot_4ch, 
                                              mask=(arr_annot_4ch != MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']))
        f, ax = plt.subplots(2, 2)
        f.set_size_inches(9, 9.1)
        with imageio.get_writer(f'rank_{i}_{df.iloc[idx]["sample_id"]}.gif', mode='I') as writer:
            # ax[1, 1].plot(cleaned_dfs[1].iloc[idx][keys]/1000.0)
            ax[1, 1].plot(df.iloc[idx][keys]/1000.0)
            ax[1, 1].plot([0, 50], [petersen_max[idx], petersen_max[idx]])
            ax[1, 1].plot([0, 50], [petersen_min[idx], petersen_min[idx]])
            ax[1, 1].set_xticks(np.arange(0, 60, 10))
            ax[1, 1].set_xticklabels(np.arange(0, 60, 10))
            for t in range(0, MRI_FRAMES, 5):
                ax[0, 0].imshow(arr_2ch[:, :, 0, t], cmap='gray')
                ax[0, 0].imshow(arr_annot_2ch_ma[:, :, 0, t], alpha=0.5)
                ax[0, 1].imshow(arr_3ch[:, :, 0, t], cmap='gray')
                ax[0, 1].imshow(arr_annot_3ch_ma[:, :, 0, t], alpha=0.5)
                ax[1, 0].imshow(arr_4ch[:, :, 0, t], cmap='gray')
                ax[1, 0].imshow(arr_annot_4ch_ma[:, :, 0, t], alpha=0.5)
                ax[0, 0].set_xticks([])   
                ax[0, 1].set_xticks([]) 
                ax[1, 0].set_xticks([]) 
                ax[0, 0].set_yticks([])   
                ax[0, 1].set_yticks([]) 
                ax[1, 0].set_yticks([])          
                f.suptitle(f'Patient: {df.iloc[idx]["sample_id"]}, Poisson error = {err[idx]:.3f}')
                f.savefig(f'tmp_{t}.png', dpi=100)
                image = imageio.imread(f'tmp_{t}.png')
                writer.append_data(image) 


