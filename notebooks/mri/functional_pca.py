# %%
from statsmodels.multivariate.pca import PCA
import pandas as pd

df_james = pd.read_csv('la.combined.tsv', sep='\t')
keep_sids = df_james[df_james['bad']==False][['sid', 'la_max_circle', 'la_min_circle', 'la_max_circle_geom', 'la_min_circle_geom']]

df_petersen = pd.read_csv('returned_lv_mass.tsv', sep='\t')

# %%
df_poi = pd.read_csv("/home/pdiachil/ml/notebooks/mri/all_atria_boundary_v20200901.csv")
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
import numpy as np
projection_petersen['poisson_err_max'] = np.sqrt(projection_petersen['LA_poisson_max'] - projection_petersen['LA_Biplan_vol_max'])**2.0/projection_petersen['LA_Biplan_vol_max']
projection_petersen['la_max_circle_err'] = np.sqrt(projection_petersen['la_max_circle'] - projection_petersen['LA_Biplan_vol_max'])**2.0/projection_petersen['LA_Biplan_vol_max']

# %%
projection_petersen[(projection_petersen['poisson_err_max']<0.01) & \
                    (projection_petersen['la_max_circle_err']<0.01)][['poisson_err_max', 'la_max_circle_err', 'sample_id']]
# %%
import matplotlib.pyplot as plt
select_petersen = projection_petersen[projection_petersen['sample_id'] == 3267768]

i =0
f, ax = plt.subplots()
ax.plot(select_petersen[keys].iloc[i])
ax.plot([0, 50], [select_petersen['LA_Biplan_vol_max'].iloc[i], select_petersen['LA_Biplan_vol_max'].iloc[i]], color='k')
ax.plot([0, 50], [select_petersen['LA_Biplan_vol_min'].iloc[i], select_petersen['LA_Biplan_vol_min'].iloc[i]], color='k')
ax.plot([0, 50], [select_petersen['la_max_circle'].iloc[i], select_petersen['la_max_circle'].iloc[i]], color='red')
ax.plot([0, 50], [select_petersen['la_min_circle'].iloc[i], select_petersen['la_min_circle'].iloc[i]], color='red')
ax.set_title(select_petersen['sample_id'].iloc[i])

# %%
import seaborn as sns
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
