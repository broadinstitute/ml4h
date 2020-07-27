# %%
import pandas as pd
df = pd.read_csv('/home/paolo/mgh_mrns_to_extract/waveform_analysis/pretrain_onelead/tensors_all_intersect.csv')


# %%
df

# %%
import h5py
ff = h5py.File('/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/40643.hd5', 'r')

# %%
ff['partners_ecg_rest/2004-02-08T15:56:00'].keys()

# %%
