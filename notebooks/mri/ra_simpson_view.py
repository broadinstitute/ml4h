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
all_cleaned_simpson = pd.read_csv('/home/pdiachil/projects/ra_simpson/all_cleaned_RA_simpson_fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122.csv')
# %%
all_cleaned_simpson[all_cleaned_simpson[cols].max(axis=1)<1]
# %%
