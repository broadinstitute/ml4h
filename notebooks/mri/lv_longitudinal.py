# %%
import numpy as np
import pandas as pd

lv = pd.concat([
    pd.read_csv(f'/home/pdiachil/projects/lvm/LV_processed_{start}_{start+10}.csv') \
                for start in range(0, 5000, 10) if os.path.isfile(f'/home/pdiachil/projects/lvm/LV_processed_{start}_{start+10}.csv')
])
lvw = pd.concat([
    pd.read_csv(f'/home/pdiachil/projects/lvm/LVW_processed_{start}_{start+10}.csv') \
                for start in range(0, 5000, 10) if os.path.isfile(f'/home/pdiachil/projects/lvm/LVW_processed_{start}_{start+10}.csv')
])

lvm_cols = [f'LVM_poisson_{i}' for i in range(50)]
# %%
lv = pd.concat([
    pd.read_csv(f'/home/pdiachil/projects/lvm_all/LV_processed_{start}_{start+50}.csv') \
                for start in range(0, 50000, 50) if os.path.isfile(f'/home/pdiachil/projects/lvm_all/LV_processed_{start}_{start+50}.csv')
])
lvw = pd.concat([
    pd.read_csv(f'/home/pdiachil/projects/lvm_all/LVW_processed_{start}_{start+50}.csv') \
                for start in range(0, 50000, 50) if os.path.isfile(f'/home/pdiachil/projects/lvm_all/LVW_processed_{start}_{start+50}.csv')
])
la = pd.concat([
    pd.read_csv(f'/home/pdiachil/projects/lvm_all/LA_processed_{start}_{start+50}.csv') \
                for start in range(0, 50000, 50) if os.path.isfile(f'/home/pdiachil/projects/lvm_all/LA_processed_{start}_{start+50}.csv')
])

lv = lv[lv['sample_id']>0]
lvw = lvw[lvw['sample_id']>0]
lvm = lvw.merge(lv, on='sample_id')
for i in range(50):
    lvm[f'LVM_poisson_{i}'] = lvm[f'LVW_poisson_{i}'] - lvm[f'LV_poisson_{i}']

# %%
petersen = pd.read_csv('/home/pdiachil/returned_lv_mass.tsv', sep='\t')
petersen = petersen.dropna()
petersen = petersen.merge(lv, on='sample_id')
petersen = petersen.merge(lvw, on='sample_id')
# %%

for i in range(50):
    petersen[f'LVM_poisson_{i}'] = petersen[f'LVW_poisson_{i}'] - petersen[f'LV_poisson_{i}']


# %%
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
lvm_scaled = scaler.fit_transform(petersen[lvm_cols])
pca = PCA(n_components=5)
pca.fit(lvm_scaled)
f, ax = plt.subplots()
ax.bar(range(5), pca.explained_variance_ratio_)
# %%
f, ax = plt.subplots(5, 1)
f.set_size_inches(6, 4)
for i, component in enumerate(pca.components_):
    ax[i].plot(component, linewidth=3, color='black')
    ax[i].set_ylabel(f'PC {i+1}')
    ax[i].set_xticklabels([])
    ax[i].set_xticks(range(10, 51, 10))
    ax[i].set_xlim([0, 50])
ax[i].set_xticklabels(map(str, range(10, 51, 10)))
ax[i].set_xlabel('Frames')
plt.tight_layout()
# %%
import scipy
import scipy.stats

f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(petersen['LVM_poisson_0']*1.05, petersen['LVM'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Poisson) (ml)')
ax.set_ylabel('LVM (Petersen) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(petersen['LVM_poisson_0'], petersen['LVM'])[0]
ax.set_title(f'n={len(petersen)}, r={pearson:.2f}')
plt.tight_layout()
# %%
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(petersen[lvm_cols].min(axis=1)*1.05, petersen['LVM'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Poisson) (g)')
ax.set_ylabel('LVM (Petersen) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(petersen[lvm_cols].min(axis=1)*1.05, petersen['LVM'])[0]
ax.set_title(f'n={len(petersen)}, r={pearson:.2f}')
plt.tight_layout()
# %%
# %%
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(petersen[lvm_cols].mean(axis=1)*1.05, petersen['LVM'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Poisson) (g)')
ax.set_ylabel('LVM (Petersen) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(petersen[lvm_cols].mean(axis=1)*1.05, petersen['LVM'])[0]
ax.set_title(f'n={len(petersen)}, r={pearson:.2f}')
plt.tight_layout()
plt.savefig('mean_lvm_petersen.png', dpi=500)
# %%
# Values from Shaan
df_shaan = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/ecg_lvh/ecg_inference_weighted_all.tsv', sep='\t')
petersen = petersen.merge(df_shaan, on='sample_id')
petersen = petersen.dropna()

# %%
f, ax = plt.subplots()
f.set_size_inches(3, 3)
is_lvh = petersen['lvh_from_indexed_lvm_weighted_left_ventricular_hypertrophy_actual']>0.5
ax.hexbin(
    petersen[is_lvh][lvm_cols].mean(axis=1)*1.05,
    petersen[is_lvh]['adjusted_myocardium_mass_asym_outlier_no_poor_actual'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray',
)
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Poisson) (g)')
ax.set_ylabel('LVM (inlinevf) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(petersen[is_lvh][lvm_cols].mean(axis=1)*1.05, petersen[is_lvh]['adjusted_myocardium_mass_asym_outlier_no_poor_actual'])[0]
ax.set_title(f'n={len(petersen)}, r={pearson:.2f}')
plt.tight_layout()
plt.savefig('lvm_poisson_inlinevf.png', dpi=500)

# %%

f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(
    petersen[is_lvh]['LVM'],
    petersen[is_lvh]['adjusted_myocardium_mass_asym_outlier_no_poor_actual'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray',
)
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Petersen) (g)')
ax.set_ylabel('LVM (inlinevf) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(petersen[is_lvh]['LVM'], petersen[is_lvh]['adjusted_myocardium_mass_asym_outlier_no_poor_actual'])[0]
ax.set_title(f'n={len(petersen)}, r={pearson:.2f}')
plt.tight_layout()
plt.savefig('lvm_petersen_inlinevf.png', dpi=500)

# %%
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(
    petersen[lvm_cols].mean(axis=1)*1.05,
    petersen['adjusted_myocardium_mass_asym_outlier_no_poor_prediction'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray',
)
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Poisson) (g)')
ax.set_ylabel('LVM (inlinevf) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(petersen[lvm_cols].mean(axis=1)*1.05, petersen['adjusted_myocardium_mass_asym_outlier_no_poor_prediction'])[0]
ax.set_title(f'n={len(petersen)}, r={pearson:.2f}')
plt.tight_layout()

# %%
petersen['LVM_Poisson'] = petersen[lvm_cols].mean(axis=1)*1.05
# %%
petersen[['sample_id', 'LVM', 'LVM_Poisson', 'adjusted_myocardium_mass_asym_outlier_no_poor_actual']].to_csv('petersen_shaan_poisson_lvm.csv', index=False)
# %%
# Values from Shaan
df_shaan = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/ecg_lvh/ecg_inference_weighted_all.tsv', sep='\t')
lvm = lvm.merge(df_shaan, on='sample_id')
lvm = lvm.dropna()

# %%
import matplotlib.pyplot as plt
import scipy.stats
f, ax = plt.subplots()
f.set_size_inches(3, 3)
is_lvh = petersen['lvh_from_indexed_lvm_weighted_left_ventricular_hypertrophy_actual']>0.5
ax.hexbin(
    lvm[lvm_cols].mean(axis=1)*1.05,
    lvm['adjusted_myocardium_mass_asym_outlier_no_poor_actual'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray',
)
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Poisson) (g)')
ax.set_ylabel('LVM (inlinevf) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(lvm[lvm_cols].mean(axis=1)*1.05, lvm['adjusted_myocardium_mass_asym_outlier_no_poor_actual'])[0]
ax.set_title(f'n={len(lvm)}, r={pearson:.2f}')
plt.tight_layout()
plt.savefig('lvm_poisson_inlinevf_all.png', dpi=500)

# %%
import matplotlib.pyplot as plt
import scipy.stats
f, ax = plt.subplots()
f.set_size_inches(3, 3)
is_lvh = lvm['lvh_from_indexed_lvm_weighted_left_ventricular_hypertrophy_actual']>0.5
ax.hexbin(
    lvm[is_lvh][lvm_cols].mean(axis=1)*1.05,
    lvm[is_lvh]['adjusted_myocardium_mass_asym_outlier_no_poor_actual'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray',
)
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Poisson) (g)')
ax.set_ylabel('LVM (inlinevf) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(lvm[is_lvh][lvm_cols].mean(axis=1)*1.05, lvm[is_lvh]['adjusted_myocardium_mass_asym_outlier_no_poor_actual'])[0]
ax.set_title(f'n={len(lvm)}, r={pearson:.2f}')
plt.tight_layout()
plt.savefig('lvm_poisson_inlinevf_lvh.png', dpi=500)

# %%
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(
    lvm[lvm_cols].mean(axis=1)*1.05,
    lvm['adjusted_myocardium_mass_asym_outlier_no_poor_prediction'], extent=(00, 300, 00, 300), mincnt=1, cmap='gray',
)
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'LVM (Poisson) (g)')
ax.set_ylabel('LVM (ECG) (g)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])
pearson = scipy.stats.pearsonr(lvm[lvm_cols].mean(axis=1)*1.05, lvm['adjusted_myocardium_mass_asym_outlier_no_poor_prediction'])[0]
ax.set_title(f'n={len(lvm)}, r={pearson:.2f}')
plt.tight_layout()
plt.savefig('lvm_poisson_ecg.png', dpi=500)
# %%
lvm['LVM_Poisson'] = lvm[lvm_cols].mean(axis=1)*1.05
# %%
lvm[['sample_id', 'LVM_Poisson', 'adjusted_myocardium_mass_asym_outlier_no_poor_actual', 'adjusted_myocardium_mass_asym_outlier_no_poor_prediction']].to_csv('all_shaan_poisson_lvm.csv', index=False)
# %%
