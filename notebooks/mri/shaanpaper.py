# %%
from statsmodels.multivariate.pca import PCA
import pandas as pd

# %%
df_saxrv_20201122 = pd.read_csv('/home/pdiachil/projects/surface_reconstruction/sax-v20201116-lax-v20201122-petersen/all_RV_discs_sax_v20201116_lax_v20201122.csv')
df_saxrv_20201122 = df_saxrv_20201122[df_saxrv_20201122['sample_id']!=-1]

# %%
import seaborn as sns
import numpy as np

sns.distplot(df_saxrv_20201122['depth'], kde=False, bins=np.linspace(8.0, 12.0, 20))
# %%

df_saxrv_20201122[df_saxrv_20201122['depth']<9.9]
# %%
