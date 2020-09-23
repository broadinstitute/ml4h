# %%
from statsmodels.multivariate.pca import PCA
import pandas as pd

df_james = pd.read_csv('la.combined.tsv', sep='\t')
keep_sids = df_james[df_james['bad']==False][['sid', 'la_max_circle', 'la_min_circle', 'la_max_circle_geom', 'la_min_circle_geom']]

df_petersen = pd.read_csv('returned_lv_mass.tsv', sep='\t')

# %%
df_poi = pd.read_csv("/home/pdiachil/ml/notebooks/mri/all_atria_boundary_v20200905.csv")
df_poi = df_poi.merge(keep_sids, left_on='sample_id', right_on='sid')
df_poi = df_poi.merge(df_petersen[['LA_Biplan_vol_max', 'LA_Biplan_vol_min', 'sample_id']], on='sample_id')
df_poi = df_poi.dropna()
# %%
keys = [f'LA_poisson_{d}' for d in range(50)]
pca = PCA(df_poi[keys], ncomp=10)

pca_projections = pca.projection
pca_projections['sample_id'] = df_poi['sample_id']
pca_projections['la_max_circle'] = df_poi['la_max_circle']
pca_projections['la_min_circle'] = df_poi['la_min_circle']

# %%
projection_petersen = pca_projections.merge(df_petersen, on='sample_id')
projection_petersen['LA_poisson_max'] = projection_petersen[keys].max(axis=1)
projection_petersen['LA_poisson_min'] = projection_petersen[keys].min(axis=1)

# %%
import scipy
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(projection_petersen['LA_poisson_max'], projection_petersen['la_max_circle'], extent=(0, 125, 0, 125), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'Maximum LA volume (3-D surface) (ml)')
ax.set_ylabel('Maximum LA volume (circle) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250])
ax.set_yticks([0, 50, 100, 150, 200, 250])
ax.set_xlim([0, 150])
ax.set_ylim([0, 150])
pearson = scipy.stats.pearsonr(projection_petersen['la_max_circle'], projection_petersen['LA_poisson_max'])[0]
ax.set_title(f'n={len(projection_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('poisson_circle.png', dpi=500)

# %%
import scipy
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(projection_petersen['la_max_circle'], projection_petersen['LA_Biplan_vol_max'], extent=(0, 125, 0, 125), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'Maximum LA volume (circle) (ml)')
ax.set_ylabel('Maximum LA volume (biplane) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250])
ax.set_yticks([0, 50, 100, 150, 200, 250])
ax.set_xlim([0, 150])
ax.set_ylim([0, 150])
pearson = scipy.stats.pearsonr(projection_petersen['la_max_circle'], projection_petersen['LA_Biplan_vol_max'])[0]
ax.set_title(f'n={len(projection_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('circle_petersen.png', dpi=500)

# %%
import scipy
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(projection_petersen['LA_poisson_max'], projection_petersen['LA_Biplan_vol_max'], extent=(0, 125, 0, 125), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 300], [0, 300], color='k')
ax.set_xlabel(f'Maximum LA volume (3-D surface) (ml)')
ax.set_ylabel('Maximum LA volume (biplane) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250])
ax.set_yticks([0, 50, 100, 150, 200, 250])
ax.set_xlim([0, 150])
ax.set_ylim([0, 150])
pearson = scipy.stats.pearsonr(projection_petersen['LA_Biplan_vol_max'], projection_petersen['LA_poisson_max'])[0]
ax.set_title(f'n={len(projection_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('petersen_poisson.png', dpi=500)


# %%
import numpy as np
projection_petersen['poisson_disc_max'] = (projection_petersen['LA_poisson_max'] - projection_petersen['LA_Biplan_vol_max'])/projection_petersen['LA_Biplan_vol_max']
projection_petersen['la_max_circle_disc'] = (projection_petersen['la_max_circle'] - projection_petersen['LA_Biplan_vol_max'])/projection_petersen['LA_Biplan_vol_max']
projection_petersen['poisson_err_max'] = np.abs(projection_petersen['LA_poisson_max'] - projection_petersen['LA_Biplan_vol_max'])/projection_petersen['LA_Biplan_vol_max']
projection_petersen['la_max_circle_err'] = np.abs(projection_petersen['la_max_circle'] - projection_petersen['LA_Biplan_vol_max'])/projection_petersen['LA_Biplan_vol_max']

# %%

import seaborn as sns
f, ax = plt.subplots()
sns.distplot(projection_petersen['poisson_disc_max'], ax=ax, kde=False, label='3-D', color='black', hist_kws={'alpha': 0.9})
sns.distplot(projection_petersen['la_max_circle_disc'], ax=ax, kde=False, label='circle', color='gray', hist_kws={'alpha': 0.9})
ax.set_xlabel('e (3-D, circle)')
ax.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
ax.set_xticklabels(['-50%', '-25%', '0%', '25%', '50%', '75%'])
ax.set_xlim([-0.5, 0.8])
ax.legend()
plt.savefig('distplot.png', dpi=500)
print(np.sum(projection_petersen['poisson_err_max']), np.sum(projection_petersen['la_max_circle_err']))

# %%

arr = np.zeros((2, 2), dtype=np.int64)
thresh = 0.15
arr[0, 0] = len(
    projection_petersen[(projection_petersen['poisson_err_max']<=thresh) & \
                                    (projection_petersen['la_max_circle_err']<=thresh)],
)
arr[1, 1] = len(
    projection_petersen[(projection_petersen['poisson_err_max']>thresh) & \
                                    (projection_petersen['la_max_circle_err']>thresh)],
)
arr[0, 1] = len(
    projection_petersen[(projection_petersen['poisson_err_max']>thresh) & \
                                    (projection_petersen['la_max_circle_err']<=thresh)],
)
arr[1, 0] = len(
    projection_petersen[(projection_petersen['poisson_err_max']<=thresh) & \
                                    (projection_petersen['la_max_circle_err']>thresh)],
)

f, ax = plt.subplots()
f.set_size_inches(4, 3)
sns.heatmap(arr, annot=True, cmap='gray', ax=ax, fmt='d')
ax.set_xticklabels(['low $|e|$ (3-D)', 'high $|e|$ (3-D)'], rotation=45)
ax.set_yticklabels(['low $|e|$ (circle)', 'high $|e|$ (circle)'], rotation='horizontal')
plt.tight_layout()
plt.savefig('confusion.png', dpi=500)

# %%
idx = np.argsort((projection_petersen['la_max_circle'] - projection_petersen['LA_poisson_max'])).iloc[-10]
projection_petersen.iloc[idx][['sample_id', 'la_max_circle', 'LA_poisson_max', 'LA_Biplan_vol_max']]
# %%
import matplotlib.pyplot as plt


select_petersen = projection_petersen[projection_petersen['sample_id'] == 5936590]

i = 0
f, ax = plt.subplots()
ax.plot(select_petersen[keys].iloc[i], 'k', linewidth=3, label='3-D')
# ax.plot(df_poi[df_poi['sample_id']==1000800][keys].iloc[0])
ax.plot([0, 50], [select_petersen['LA_Biplan_vol_max'].iloc[i], select_petersen['LA_Biplan_vol_max'].iloc[i]], color='k', label='biplane')
ax.plot([0, 50], [select_petersen['LA_Biplan_vol_min'].iloc[i], select_petersen['LA_Biplan_vol_min'].iloc[i]], color='k')
ax.plot([0, 50], [select_petersen['la_max_circle'].iloc[i], select_petersen['la_max_circle'].iloc[i]], color='gray', label='circle')
ax.plot([0, 50], [select_petersen['la_min_circle'].iloc[i], select_petersen['la_min_circle'].iloc[i]], color='gray')
ax.set_title(f'{select_petersen["sample_id"].iloc[i]}')
ax.legend()
ax.set_xticks(list(range(5, 51, 5)))
ax.set_xticklabels(list(range(5, 51, 5)))
ax.set_xlim([0, 49])
ax.set_xlabel('Frames')
ax.set_ylabel('LA$_{max}$ (ml)')
f.savefig('1000800.png', dpi=500)
# %%
f, ax = plt.subplots()
f.set_size_inches(4, 3)
maxs = ['LA_poisson_max', 'LA_Biplan_vol_max', 'la_max_circle']
labels = ['3-D', 'Biplane', 'Circle']
sns.heatmap(projection_petersen[maxs].corr(), annot=True, cmap='gray', ax=ax)
ax.set_xticklabels(labels, rotation=45)
ax.set_yticklabels(labels)
plt.tight_layout()
plt.savefig('cross_corr.png', dpi=500)
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_poi = pd.read_csv("/home/pdiachil/ml/notebooks/mri/all_atria_boundary_v20200905.csv")
scaler = StandardScaler()
df_poi_scaled = scaler.fit_transform(df_poi[keys])
pca = PCA(n_components=5)
pca.fit(df_poi[keys])
f, ax = plt.subplots()
ax.bar(range(5), pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))
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
f.savefig('PCs.png', dpi=500)

# %%
f, ax = plt.subplots(5, 1)
f.set_size_inches(3, 4)
trans = pca.transform(df_poi[keys])
for i, component in enumerate(pca.components_):
    sns.distplot(trans[:, i], kde=False, ax = ax[i])
plt.tight_layout()
f.savefig('PC_dists.png', dpi=500)
# %%

for i, component in enumerate(pca.components_):
    df_poi[f'PC_{i}'] = trans[:, i]

df_poi[['sample_id'] + keys + [f'PC_{i}' for i in range(5)]].to_csv('all_atria_boundary_pcs_v20200905.csv', index=False)
# %%
