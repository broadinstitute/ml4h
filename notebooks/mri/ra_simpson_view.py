# %%
import pandas as pd
import glob as glob
import os

all_ra_simpson = pd.read_csv('/home/pdiachil/projects/ra_simpson/all_RA_simpson.csv')

# %%
all_ra_simpson[all_ra_simpson['sample_id']>0].to_csv('/home/pdiachil/projects/ra_simpson/all_cleaned_RA_simpson_fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122.csv', index=False)
# %%
import matplotlib.pyplot as plt
import seaborn as sns
cols = [f'RA_simpson_{t}' for t in range(50)]
f, ax = plt.subplots(1, 2)
f.set_size_inches(4, 2.25)
sns.distplot(all_ra_simpson[all_ra_simpson['sample_id']>0][cols].max(axis=1), ax=ax[0], kde=False, color='gray')
sns.distplot(all_ra_simpson[all_ra_simpson['sample_id']>0][cols].min(axis=1), ax=ax[1], kde=False, color='gray')
ax[0].set_xlabel('RA max (ml)')
ax[1].set_xlabel('RA min (ml)')
ax[0].set_ylabel('counts')
ax[0].set_yticks(range(0, 6000, 1000))
ax[1].set_yticks(range(0, 6000, 1000))
ax[1].set_yticklabels([])
plt.tight_layout()
f.savefig('ra_simpson_dist.png', dpi=500)
# %%
all_cleaned_simpson = pd.read_csv('/home/pdiachil/projects/ra_simpson/all_clean_RA_simpson.csv')
# %%
james_ra = pd.read_csv('/home/pdiachil/projects/ra_simpson/right_left_heart_combined.pheno.tsv', sep='\t')
# %%
all_cleaned_ra = all_cleaned_simpson.merge(james_ra, on=['sample_id', 'instance'])
# %%
cols = [f'RA_simpson_{t}' for t in range(50)]
all_cleaned_ra['RA_simpson_max'] = all_cleaned_ra[cols].max(axis=1)
all_cleaned_ra['RA_simpson_min'] = all_cleaned_ra[cols].min(axis=1)
all_cleaned_ra['max_ra_vol'] = np.sqrt(all_cleaned_ra['max_ra'] / np.pi)**3.0*4.0/3.0*np.pi
all_cleaned_ra['min_ra_vol'] = np.sqrt(all_cleaned_ra['min_ra'] / np.pi)**3.0*4.0/3.0*np.pi
# %%
import numpy as np
all_cleaned_ra['RA_max_diff'] = np.abs((all_cleaned_ra['RA_simpson_max'] - all_cleaned_ra['max_ra_vol'])/all_cleaned_ra['max_ra_vol'])
all_cleaned_ra['RA_min_diff'] = np.abs((all_cleaned_ra['RA_simpson_min'] - all_cleaned_ra['min_ra_vol'])/all_cleaned_ra['min_ra_vol'])
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(all_cleaned_ra['RA_max_diff'].dropna())
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(all_cleaned_ra['RA_min_diff'].dropna())

# %%
all_cleaned_ra[(all_cleaned_ra['RA_min_diff']<all_cleaned_ra['RA_min_diff'].quantile(0.998)+0.01) & \
               (all_cleaned_ra['RA_min_diff']>all_cleaned_ra['RA_min_diff'].quantile(0.998)-0.01)]
# %%
sample_id = 2725878
f, ax = plt.subplots()
ax.plot(all_cleaned_ra[all_cleaned_ra['sample_id']==sample_id][cols].values[0], label='Simpson')
ax.plot([0, 50], [all_cleaned_ra[all_cleaned_ra['sample_id']==sample_id]['min_ra_vol'].values, all_cleaned_ra[all_cleaned_ra['sample_id']==sample_id]['min_ra_vol'].values], label='min_ra_sphere')
ax.plot([0, 50], [all_cleaned_ra[all_cleaned_ra['sample_id']==sample_id]['max_ra_vol'].values, all_cleaned_ra[all_cleaned_ra['sample_id']==sample_id]['max_ra_vol'].values], label='max_ra_sphere')
ax.legend()
f.savefig(f'{sample_id}.png')
# %%
