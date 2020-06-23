# %%
import numpy as np
import pandas as pd
import os
# %%
df = pd.concat([pd.read_csv(f'/home/pdiachil/ml/atria/petersen_processed_{d}.csv', sep='\t') for d in range(5000) \
                if os.path.isfile(f'/home/pdiachil/ml/atria/petersen_processed_{d}.csv')])

# %%
import scipy.signal
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    idx = np.arange(len(data))
    return idx[s<m], data[s<m]

def plot_volume(df, idx, ax):
    keys = [f'LA_ellipsoid_{d}' for d in range(50)]
    colors = ['gray', 'gray', 'black' ]
    for i, petersen_keys in enumerate(['LA_Biplan_vol']):
        ax.plot([0, 49], [df[petersen_keys+'_max'].values[idx], df[petersen_keys+'_max'].values[idx]], color=colors[i])
        ax.plot([0, 49], [df[petersen_keys+'_min'].values[idx], df[petersen_keys+'_min'].values[idx]], color=colors[i])
    vol_signal = df[keys].values[idx]/1000.0
    x, vol_signal_filtered = reject_outliers(vol_signal)
    ax.plot(x, vol_signal_filtered)

# %%
import matplotlib.pyplot as plt
f, ax = plt.subplots()
keys = [f'LA_ellipsoid_{d}' for d in range(50)]
ell_max = df[keys].values.max(axis=1)/1000.0
ell_min = df[keys].values.min(axis=1)/1000.0
ell_mean = df[keys].values.mean(axis=1)/1000.0
petersen_max = df['LA_Biplan_vol_max'].values
petersen_min = df['LA_Biplan_vol_min'].values
petersen_mean = 0.33*df['LA_Biplan_vol_max']+0.67*df['LA_Biplan_vol_min']
ax.plot(ell_mean, petersen_mean, 'o')
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300])

# %%
idxs = (np.abs((ell_max-petersen_max)) + \
        np.abs((ell_min-petersen_min))).argsort()

# %%
# for i, idx in enumerate(idxs[2000:]):
#     f, ax = plt.subplots()
#     plot_volume(df, idx, ax)
#     ax.set_title(f'{ell_max[idx]}, {petersen_max[idx]}')
#     if i == 10: break

# %%
sample_idxs = df['sample_id'].iloc[idxs[2000:20110]].values
string_idxs = f'{*sample_idxs,}'.replace('(', '').replace(')', '').replace(',', '')
print(f'for i in {string_idxs}; do gsutil cp gs://ml4cvd/pdiachil/atria/atria_ellipsoid/$i* ./; done')


# %%
