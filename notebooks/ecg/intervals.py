# %%
import pandas as pd

metadata = pd.read_parquet('/home/pdiachil/ml/notebooks/ecg/joint_meta_collected_tabular.pq')
# %%
sel_cols = []
for col in metadata.columns:
    if 'restingecgmeasurement' in col.lower():
        sel_cols.append(col)
# %%
metadata[sel_cols]
# %%
for col in metadata.columns:
    print(col)
# %%
metadata['source_file'].iloc[0]
# %%
metadata['sample_id'] = metadata['source_file'].str.split('/').str[-1].str.split('_').str[0].apply(int)
metadata['instance'] = metadata['source_file'].str.split('/').str[-1].str.split('_').str[-2].apply(int)
# %%
metadata[['sample_id', 'instance'] + sel_cols].to_csv('resting_ecg_measurements.csv', index=False)
# %%
