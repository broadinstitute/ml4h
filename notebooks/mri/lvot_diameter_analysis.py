# %%
import pandas as pd
# %%
cols_start = ['sample_id', 'instance', 'nset', 'frame', 'fname']
cols = []
regions = {'lvot': 1, 'aortic_root': 1, 'ascending_aorta': 20}
for region, npts in regions.items():
        for p in range(npts):
            for px in [0, 1]:
                for d in ['r', 'c']:
                    cols.append(f'{region}_{p}_{px}_{d}')

# %%
all = pd.read_csv('/home/pdiachil/projects/lvot/1cm_csv/all_lvot_diameter.csv', usecols=cols_start+cols)
all_height = pd.read_csv('/home/pdiachil/projects/lvot/1cm_height_csv/all_lvot_diameter.csv', usecols=cols_start+cols)

# %%
all = all.drop_duplicates()
all_height = all_height.drop_duplicates()

# %%
all.to_csv('/home/pdiachil/projects/lvot/1cm_csv/all_cleaned_lvot_diameter.csv', index=False)
all_height.to_csv('/home/pdiachil/projects/lvot/1cm_height_csv/all_cleaned_lvot_diameter.csv', index=False)
# %%
howmany = {'sample_id': [], 'instance': [], 'frame': [], 'asc_aorta_npts': []}
howmany['sample_id'].extend(all['sample_id'].values.tolist())
howmany['instance'].extend(all['instance'].values.tolist())
howmany['frame'].extend(all['frame'].values.tolist())
howmany['asc_aorta_npts'].extend(((all[cols]>0).sum(axis=1).values//4-2).tolist())

# %%
howmany_height = {'sample_id': [], 'instance': [], 'frame': [], 'asc_aorta_npts': []}
howmany_height['sample_id'].extend(all_height['sample_id'].values.tolist())
howmany_height['instance'].extend(all_height['instance'].values.tolist())
howmany_height['frame'].extend(all_height['frame'].values.tolist())
howmany_height['asc_aorta_npts'].extend(((all_height[cols]>0).sum(axis=1).values//4-2).tolist())


# %%
howmany_df = pd.DataFrame(howmany)
howmany_height_df = pd.DataFrame(howmany_height)
# %%
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
howmany_npts = howmany_df.groupby(['sample_id', 'instance']).mean()
howmany_height_npts = howmany_height_df.groupby(['sample_id', 'instance']).mean()

f, ax = plt.subplots()
sns.distplot(howmany_npts['asc_aorta_npts'], label='1 cm', kde=False, bins=range(0, 15), ax=ax)
sns.distplot(howmany_height_npts['asc_aorta_npts'], label='1 cm / height', kde=False, bins=range(0, 15), ax=ax)
ax.legend()
f.savefig('asc_aorta_npts.png', dpi=500)
# %%
all6 = pd.read_csv('/home/pdiachil/projects/lvot_6points/all_lvot_diameter_6points.csv')
# %%
import numpy as np
unique = set(np.unique(all['sample_id']))
unique6 = set(np.unique(all6['sample_id']))

# %%
len(unique - unique6)
# %%
import glob
import os

all_csv = set(map(os.path.basename, glob.glob('/home/pdiachil/projects/lvot/lvot*.csv')))
all6_csv = set(map(os.path.basename, glob.glob('/home/pdiachil/projects/lvot_6points/lvot*.csv')))
# %%
remaining = all_csv - all6_csv

def return_index(fname):
    idx = int(fname.split('_')[2]) // 46
    return idx

indices_remaining = list(map(return_index, remaining))
# %%
len(remaining)
# %%
