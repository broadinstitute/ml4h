# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
df_ell = pd.concat([pd.read_csv(f'/home/pdiachil/atria/petersen_processed_{d}.csv', sep='\t') for d in range(5000) \
                if os.path.isfile(f'/home/pdiachil/atria/petersen_processed_{d}.csv')])
df_ell = df_ell.reset_index()
df_poi = pd.concat([pd.read_csv(f'/home/pdiachil/atria_poisson/petersen_processed_{d}.csv', sep='\t') for d in range(5000) \
                if os.path.isfile(f'/home/pdiachil/atria_poisson/petersen_processed_{d}.csv')])
df_poi = df_poi.reset_index()

# %%
import scipy.signal
def reject_outliers(data, m = 4.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    idx = np.arange(len(data), dtype=np.int)
    ri = idx[s<m]
    not_ri = idx[s>=m]
    sig = compressed_sensing_1d(data, ri, not_ri)
    return sig

import scipy.fftpack as spfft
import cvxpy as cvx
def compressed_sensing_1d(data, ri, not_ri):
    n = len(data)
    # m = int(n*sample)
    A = spfft.idct(np.identity(n), norm='ortho', axis=0)
    # ri = np.random.choice(n, m, replace=False) # random sample of indices
    A = A[ri]

    # do L1 optimization
    vx = cvx.Variable(n)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A*vx == data[ri]]
    prob = cvx.Problem(objective, constraints)
    try:
        result = prob.solve(verbose=False)
        x = np.array(vx.value)
        x = np.squeeze(x)
        sig = spfft.idct(x, norm='ortho', axis=0)
        data[:] = sig
    except:
        print('Computing median instead')
        data[not_ri] = np.median(data)
    return data

def plot_volume(df, idx, ax):
    keys = [f'LA_ellipsoid_{d}' for d in range(50)]
    colors = ['gray', 'gray', 'black' ]
    for i, petersen_keys in enumerate(['LA_Biplan_vol']):
        ax.plot([0, 49], [df[petersen_keys+'_max'].values[idx], df[petersen_keys+'_max'].values[idx]], color=colors[i])
        ax.plot([0, 49], [df[petersen_keys+'_min'].values[idx], df[petersen_keys+'_min'].values[idx]], color=colors[i])
    vol_signal = df[keys].values[idx]/1000.0
    vol_signal_filtered = reject_outliers(vol_signal)
    ax.plot(vol_signal_filtered)

# # %%
# idx = 50
# ri, a = reject_outliers(df[keys].values[idx])
# f, ax = plt.subplots()
# ax.plot(a)
# ax.plot(ri, a[ri], '*')
# #ax.plot(df[keys].values[idx])

# %%
cleaned_dfs = []
idxs_sorted = []
import seaborn as sns
import scipy.stats
for df, label in zip([df_ell, df_poi], ['ellipsoid', 'poisson']):
    keys = [f'LA_{label}_{d}' for d in range(50)]
    df = df.dropna()
    cleaned_dfs.append(df[keys].apply(reject_outliers, axis=1))
    # cleaned_df = pd.DataFrame(cleaned_df.tolist(), columns=keys)
    # cleaned_dfs.append(df[keys])
    ell_max = cleaned_dfs[-1].values.max(axis=1)/1000.0
    ell_min = cleaned_dfs[-1].values.min(axis=1)/1000.0
    ell_mean = cleaned_dfs[-1].values.mean(axis=1)/1000.0
    petersen_max = df['LA_Biplan_vol_max'].values
    petersen_min = df['LA_Biplan_vol_min'].values
    petersen_mean = 0.33*df['LA_Biplan_vol_max']+0.67*df['LA_Biplan_vol_min']

    err  = (np.abs((ell_max-petersen_max)) + \
             np.abs((ell_min-petersen_min))) / \
            (petersen_max+petersen_min)
    idxs = err.argsort()

    idxs_sorted.append(idxs)

    for nidxs in [1000, 2000, 3000, 4000]:
        f, ax = plt.subplots(1, 2)
        ax[0].hexbin(ell_max[idxs_sorted[-1]][:nidxs], petersen_max[idxs_sorted[-1]][:nidxs], extent=(0, 200, 0, 200), mincnt=1)
        ax[0].set_xlim([0, 200])
        ax[0].set_ylim([0, 200])
        ax[0].set_aspect('equal')
        ax[0].plot([0, 300], [0, 300])
        ax[0].set_xlabel(f'Maximal {label} volume (ml)')
        ax[0].set_ylabel('Maximal LA biplane volume (ml)')
        ax[0].set_xticks([0, 50, 100, 150, 200])
        ax[0].set_yticks([0, 50, 100, 150, 200])
        ax[0].set_title(f'n={len(ell_max[idxs_sorted[-1]][:nidxs])}, r={scipy.stats.pearsonr(ell_max[idxs_sorted[-1]][:nidxs], petersen_max[idxs_sorted[-1]][:nidxs])[0]:.2f}')

        ax[1].hexbin(ell_min[idxs_sorted[-1]][:nidxs], petersen_min[idxs_sorted[-1]][:nidxs], extent=(0, 150, 0, 150), mincnt=1)
        ax[1].set_xlim([0, 150])
        ax[1].set_ylim([0, 150])
        ax[1].set_aspect('equal')
        ax[1].plot([0, 300], [0, 300])
        ax[1].set_xlabel(f'Minimal {label} volume (ml)')
        ax[1].set_ylabel('Minimal LA biplane volume (ml)')
        ax[1].set_xticks([0, 50, 100, 150])
        ax[1].set_yticks([0, 50, 100, 150])
        ax[1].set_title(f'n={len(ell_min[idxs_sorted[-1]][:nidxs])}, r={scipy.stats.pearsonr(ell_min[idxs_sorted[-1]][:nidxs], petersen_min[idxs_sorted[-1]][:nidxs])[0]:.2f}')
        plt.tight_layout()
        f.savefig(f'/home/pdiachil/ml/notebooks/mri/{label}_vol_{nidxs}.png', dpi=500)

# %%
for idx in range(0, 3000, 100):
    f, ax = plt.subplots()
    ax.plot(cleaned_dfs[1].iloc[idxs[idx]][keys]/1000.0)
    ax.plot(df.iloc[idxs[idx]][keys]/1000.0)
    ax.plot([0, 50], [petersen_max[idxs[idx]], petersen_max[idxs[idx]]])
    ax.plot([0, 50], [petersen_min[idxs[idx]], petersen_min[idxs[idx]]])
    ax.set_xticks(np.arange(0, 60, 10))
    ax.set_xticklabels(np.arange(0, 60, 10))

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
for i in range(0, len(idxs), 100):
    idx = idxs[i]
    fpath = f'/mnt/disks/segmented-sax-lax/2020-06-26/{df.iloc[idx]["sample_id"]}.hd5'
    if not os.path.isfile(fpath): continue
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
            ax[1, 1].plot(cleaned_dfs[1].iloc[idx][keys]/1000.0)
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
                f.savefig(f'tmp_{t}.png', dpi=50)
                image = imageio.imread(f'tmp_{t}.png')
                writer.append_data(image) 




# %%
import scipy.stats
print(scipy.stats.pearsonr(ell_max[idxs[-3000:]], petersen_max[idxs[-3000:]]))

# %%
for i, idx in enumerate([2856]):
    f, ax = plt.subplots()
    plot_volume(df, idx, ax)
    ax.set_title(f'{ell_max[idx]}, {petersen_max[idx]}')
    if i == 10: break

# %%
sample_idxs = df['sample_id'].iloc[idxs[2000:20110]].values
string_idxs = f'{*sample_idxs,}'.replace('(', '').replace(')', '').replace(',', '')
print(f'for i in {string_idxs}; do gsutil cp gs://ml4cvd/pdiachil/atria/atria_ellipsoid/$i* ./; done')


# %%
