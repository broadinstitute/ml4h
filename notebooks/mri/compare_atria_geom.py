# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
df_poi = pd.concat([pd.read_csv(f'/home/pdiachil/atria_boundary/petersen_processed_{d}_{d+100}.csv') for d in range(0, 4000, 100)])
df_poi = df_poi.reset_index()

df_pet = pd.read_csv('/home/pdiachil/ml/notebooks/mri/returned_lv_mass.tsv', sep='\t')
df_pet = df_pet.merge(df_poi, on='sample_id')

# df_pet.to_csv('/home/pdiachil/ml/notebooks/mri/all_atria_boundary.csv')

# %%
import scipy.signal


def get_max(data, window=5):
    filtered_data = median_filter(data, size=5, mode='wrap')
    arg_max = np.argmax(filtered_data)
    return filtered_data[arg_max]


def get_min(data, window=5):
    filtered_data = median_filter(data, size=5, mode='wrap')
    arg_min = np.argmin(filtered_data)
    return filtered_data[arg_min]


def reject_outliers(data, m = 3.):
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

def plot_volume(df, idx):
    f, ax = plt.subplots()
    f.set_size_inches(3.5, 3)
    keys = [f'LA_poisson_{d}' for d in range(50)]
    colors = ['gray', 'gray', 'black' ]
    for i, petersen_keys in enumerate(['LA_Biplan_vol']):
        ax.plot([0, 49], [df[petersen_keys+'_max'].values[idx], df[petersen_keys+'_max'].values[idx]], color=colors[i], linewidth=3, label='Petersen et al. (2017)')
        ax.plot([0, 49], [df[petersen_keys+'_min'].values[idx], df[petersen_keys+'_min'].values[idx]], color=colors[i], linewidth=3)
    vol_signal = df[keys].values[idx]
    vol_signal_filtered = reject_outliers(vol_signal)
    print(df['sample_id'].values[idx])
    ax.plot(vol_signal_filtered, linewidth=3, color='k', label='3-D reconstruction')
    ax.set_xlim([0, 50.0])
    ax.set_xlabel("Frames")
    ax.set_ylabel('LA volume (ml)')
    ax.legend()
    plt.tight_layout()
    f.savefig(f"{df['sample_id'].values[idx]}_la_vols.png", dpi=500)

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
for df, label in zip([df_pet], ['poisson']):
    keys = [f'LA_{label}_{d}' for d in range(50)]
    df = df.dropna()
    # cleaned_dfs.append(df[keys].apply(reject_outliers, axis=1))
    # cleaned_df = pd.DataFrame(cleaned_df.tolist(), columns=keys)
    cleaned_dfs.append(df[keys])
    ell_max = cleaned_dfs[-1][keys].apply(get_max, axis=1)
    ell_min = cleaned_dfs[-1][keys].apply(get_min, axis=1)
    ell_mean = cleaned_dfs[-1].values.mean(axis=1)
    cleaned_dfs[-1]['sample_id'] = df['sample_id']
    cleaned_dfs[-1][f'LA_{label}_cleaned_max'] = ell_max
    cleaned_dfs[-1][f'LA_{label}_cleaned_min'] = ell_min
    cleaned_dfs[-1]['LA_Biplan_vol_max'] = df['LA_Biplan_vol_max']
    cleaned_dfs[-1]['LA_Biplan_vol_min'] = df['LA_Biplan_vol_min']
    petersen_max = df['LA_Biplan_vol_max'].values
    petersen_min = df['LA_Biplan_vol_min'].values
    petersen_mean = 0.33*df['LA_Biplan_vol_max']+0.67*df['LA_Biplan_vol_min']

    # cleaned_dfs[-1].to_csv('/home/pdiachil/ml/notebooks/mri/all_atria_boundary_cleaned.csv')

    err  = ((ell_max-petersen_max)**2.0) / petersen_max / petersen_max
    idxs = err.argsort()
    idxs_sorted.append(idxs)

    for nidxs in [1000, 2000, 3000, 4000]:
        f, ax = plt.subplots()
        f.set_size_inches(3, 3)
        ax.hexbin(ell_max[idxs_sorted[-1]][:nidxs], petersen_max[idxs_sorted[-1]][:nidxs], extent=(0, 125, 0, 125), mincnt=1, cmap='gray')
        ax.set_xlim([0, 200])
        ax.set_ylim([0, 200])
        ax.set_aspect('equal')
        ax.plot([0, 300], [0, 300], color='k')
        ax.set_xlabel(f'LA 3-D surface volume at max (ml)')
        ax.set_ylabel('LA biplane volume at max (ml)')
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_title(f'n={len(ell_max[idxs_sorted[-1]][:nidxs])}, r={scipy.stats.pearsonr(ell_max[idxs_sorted[-1]][:nidxs], petersen_max[idxs_sorted[-1]][:nidxs])[0]:.2f}')
        plt.tight_layout()
        f.savefig(f'/home/pdiachil/ml/notebooks/mri/{label}_vol_boundary_max_{nidxs}.png', dpi=500)

    for nidxs in [1000, 2000, 3000, 4000]:
        f, ax = plt.subplots()
        f.set_size_inches(3, 3)
        ax.hexbin(ell_min[idxs_sorted[-1]][:nidxs], petersen_min[idxs_sorted[-1]][:nidxs], extent=(0, 125, 0, 125), mincnt=1, cmap='gray')
        ax.set_xlim([0, 150])
        ax.set_ylim([0, 150])
        ax.set_aspect('equal')
        ax.plot([0, 300], [0, 300], color='k')
        ax.set_xlabel(f'LA 3-D surface volume at min (ml)')
        ax.set_ylabel('LA biplane volume at min (ml)')
        ax.set_xticks([0, 50, 100, 150])
        ax.set_yticks([0, 50, 100, 150])
        ax.set_title(f'n={len(ell_max[idxs_sorted[-1]][:nidxs])}, r={scipy.stats.pearsonr(ell_min[idxs_sorted[-1]][:nidxs], petersen_min[idxs_sorted[-1]][:nidxs])[0]:.2f}')

        plt.tight_layout()
        f.savefig(f'/home/pdiachil/ml/notebooks/mri/{label}_vol_boundary_min_{nidxs}.png', dpi=500)


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
