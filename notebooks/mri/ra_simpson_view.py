# %%
import pandas as pd
import glob as glob
import os

all_ra_simpson_lax_cog = pd.read_csv('/home/pdiachil/projects/ra_simpson/lax_cog/all_RA_lax_cog_simpson.csv')
all_ra_simpson_ortho = pd.read_csv('/home/pdiachil/projects/ra_simpson/all_clean_RA_simpson.csv')

# %%
all_ra_simpson_lax_cog = all_ra_simpson_lax_cog[all_ra_simpson_lax_cog['sample_id']>-1]
all_ra_simpson_lax_cog = all_ra_simpson_lax_cog.drop_duplicates()
all_ra_simpson_lax = all_ra_simpson_lax_cog.merge(all_ra_simpson_ortho, on=['sample_id', 'instance'])
all_ra_simpson_lax = all_ra_simpson_lax_cog
# %%
import numpy as np
cols_simpson = [f'RA_simpson_cog_{i}' for i in range(50)]
cols_lax_ortho = [f'RA_lax_ortho_{i}' for i in range(50)]
cols_lax_cog = [f'RA_lax_cog_{i}' for i in range(50)]
# cols_ortho_simpson = [f'RA_simpson_{i}' for i in range(50)]

for cols in [cols_simpson, cols_lax_ortho, cols_lax_cog, cols_ortho_simpson]:
    col_max = cols[0].replace('_0', '_max')
    col_min = cols[0].replace('_0', '_min')
    vals = all_ra_simpson_lax[cols].values
    vals[vals<0] = np.nan
    all_ra_simpson_lax[col_max] = np.nanmax(vals, axis=1)
    all_ra_simpson_lax[col_min] = np.nanmin(vals, axis=1)
    all_ra_simpson_lax = all_ra_simpson_lax.dropna(subset=[col_max, col_min])

all_ra_simpson_lax = all_ra_simpson_lax[all_ra_simpson_lax['RA_lax_cog_max']>0]
all_ra_simpson_lax = all_ra_simpson_lax[all_ra_simpson_lax['RA_lax_ortho_max']>0]

# %%
# all_ra_simpson_lax = all_ra_simpson_lax_cog.merge(all_ra_simpson_ortho, on=['sample_id', 'instance'])

# %%
import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots()
sns.distplot(all_ra_simpson_lax['RA_lax_cog_max'], ax=ax)
sns.distplot(all_ra_simpson_lax['RA_lax_ortho_max'], ax=ax)
sns.distplot(all_ra_simpson_lax['RA_lax_cog_min'], ax=ax)
sns.distplot(all_ra_simpson_lax['RA_lax_ortho_min'], ax=ax)

f, ax = plt.subplots()
sns.distplot(all_ra_simpson_lax['RA_simpson_max'], ax=ax)
sns.distplot(all_ra_simpson_lax['RA_simpson_cog_max'], ax=ax)
sns.distplot(all_ra_simpson_lax['RA_simpson_min'], ax=ax)
sns.distplot(all_ra_simpson_lax['RA_simpson_cog_min'], ax=ax)

# %%
all_ra_simpson_lax[[col for col in all_ra_simpson_lax if ('_max' in col)]].corr()


# %%
all_ra_simpson[all_ra_simpson['sample_id']>0].to_csv('/home/pdiachil/projects/ra_simpson/all_cleaned_RA_simpson_fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122.csv', index=False)
# %%

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
