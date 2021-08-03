# %%
import pandas as pd

# %%
evaluate_batch_dev = pd.read_csv(f'/mnt/disks/pdiachil-t1map/predictions/evaluate_batch_dev.tsv', sep='\t')
evaluate_batch_holdout = pd.read_csv(f'/mnt/disks/pdiachil-t1map/predictions/evaluate_batch_holdout.tsv', sep='\t')
evaluate_batch = pd.read_csv(f'/mnt/disks/pdiachil-t1map/predictions/evaluate_batch.tsv', sep='\t')
# %%
import pandas as pd
import h5py
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('/home/pdiachil/projects/t1map/inference/all_t1map_inference.csv')
df_request = pd.read_csv('/home/pdiachil/ml/notebooks/mri/remaining_rois.csv')
hd5 = h5py.File('/mnt/disks/pdiachil-t1map/predictions3/ML4H_mdrk_ukb__cardiac_t1_weighted__predictions__5e806c4c75fa47d59f3270711fc35106.h5', 'r')
df_request['sample_id'] = df_request['diff'].str.split('_').str[0].apply(int)
df_request['instance'] = df_request['diff'].str.split('_').str[1].apply(int)

df2 = df[df['instance']==2]
df_request2 = df_request[df_request['instance']==2]

# %%
common_outer = df_request2.merge(df2, on=['sample_id', 'instance'], how='outer', indicator=True)

# %%
df_request2.to_csv('/home/pdiachil/ml/notebooks/mri/remaining_rois2.csv')
# %%
f, ax = plt.subplots(2, 1)

ax[0].hist(df['LV Free Wall_model_iqr'], bins=range(0, 2000), color='blue', label='LV FW IQR')
ax[0].hist(df['Interventricular Septum_model_iqr'], bins=range(0, 2000), color='orange', label='IVS IQR', alpha=0.5)
ax[0].set_xlim([0, 2000])
ax[0].legend()

ax[1].hist(df['LV Cavity_model_iqr'], bins=range(0, 2000), color='blue', label='LV BP IQR')
ax[1].hist(df['RV Cavity_model_iqr'], bins=range(0, 2000), color='orange', label='RV BP IQR', alpha=0.5)
ax[1].set_xlim([0, 2000])
ax[1].legend()

f.savefig('distributions_t1map_myok.png', dpi=500)

# %%
f, ax = plt.subplots(2, 1)

ax[0].hist(df['LV Free Wall_model'], bins=range(0, 2000), color='blue', label='LV FW')
ax[0].hist(df['Interventricular Septum_model'], bins=range(0, 2000), color='orange', label='IVS', alpha=0.5)
ax[0].set_xlim([0, 2000])
ax[0].legend()


ax[1].hist(df['LV Cavity_model'], bins=range(0, 2000), color='blue', label='LV BP')
ax[1].hist(df['RV Cavity_model'], bins=range(0, 2000), color='orange', label='RV BP', alpha=0.5)
ax[1].set_xlim([0, 2000])
ax[1].legend()

f.savefig('distributions_t1map_bp.png', dpi=500)
# %%
cols_iqr = [
    'LV Cavity_model_iqr', 'RV Cavity_model_iqr',
    #'LV Cavity_model', 'RV Cavity_model',
    'LV Free Wall_model_iqr', 'Interventricular Septum_model_iqr',
    #'LV Free Wall_model', 'Interventricular Septum_model'
]

cols = [
    # 'LV Cavity_model_iqr', 'RV Cavity_model_iqr',
    'LV Cavity_model', 'RV Cavity_model',
    # 'LV Free Wall_model_iqr', 'Interventricular Septum_model_iqr',
    'LV Free Wall_model', 'Interventricular Septum_model'
]

# %%
import seaborn as sns
labels = ['LV BP', 'RV BP', 'LV FW', 'IVS']
f, ax = plt.subplots()
sns.heatmap(df[cols_iqr].corr(), cmap='gray', vmin=0.3, vmax=1.0, ax=ax, annot=True, fmt='.2f')
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
f.savefig('iqr_heatmap.png', dpi=500)

# %%
labels = ['LV BP', 'RV BP', 'LV FW', 'IVS']
f, ax = plt.subplots()
sns.heatmap(df[cols].corr(), cmap='gray', vmin=0.3, vmax=1.0, ax=ax, annot=True, fmt='.2f')
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
f.savefig('model_heatmap.png', dpi=500)
# %%
import scipy.stats
f, ax = plt.subplots(1, 2)
x = df['LV Free Wall_model_iqr'].values
y = df['Interventricular Septum_model_iqr'].values
ax[0].plot(df['LV Free Wall_model_iqr'], df['Interventricular Septum_model_iqr'], 'ko', alpha=0.01)
ax[0].set_aspect('equal')
ax[0].plot([0, 2000], [0, 2000], 'k-')
ax[0].set_xlim([0, 2000])
ax[0].set_ylim([0, 2000])
ax[0].set_title(f'IQR, r={scipy.stats.pearsonr(x, y)[0]:.2f}, s={scipy.stats.spearmanr(x, y)[0]:.2f}')
x = df['LV Free Wall_model'].values
y = df['Interventricular Septum_model'].values
ax[1].plot(df['LV Free Wall_model'], df['Interventricular Septum_model'], 'ko', alpha=0.01)
ax[1].set_aspect('equal')
ax[1].plot([0, 2000], [0, 2000], 'k-')
ax[1].set_xlim([0, 2000])
ax[1].set_ylim([0, 2000])
ax[1].set_yticklabels([])
ax[1].set_title(f'Median, r={scipy.stats.pearsonr(x, y)[0]:.2f}, s={scipy.stats.spearmanr(x, y)[0]:.2f}')
ax[1].set_xlabel('T1 LV FW (ms)')
ax[0].set_xlabel('T1 LV FW (ms)')
ax[0].set_ylabel('T1 IVS (ms)')
f.savefig('FWvsIVS.png', dpi=500)
# %%
import pandas as pd

bad_cases = pd.read_csv('/home/pdiachil/projects/t1map/list_ids_off_axis_low_intensity.csv')
inference = pd.read_csv('/home/pdiachil/projects/t1map/inference/all_inference_instance2_3px_iqr.csv')
not_inferred = pd.read_csv('/home/pdiachil/projects/t1map/list_ids_in_qc_dataset_but_wo_inference.csv')

common = not_inferred.merge(inference, left_on=['sample_id', 'instance_number'], right_on=['sample_id', 'instance'])
# %%
cleaned_inference = inference[~((inference['sample_id'].isin(common['sample_id']))&(inference['instance'].isin(common['instance'])))]
# %%
cleaned_inference[cleaned_inference['Interventricular Septum_model']<600]
# %%
common
# %%
common_inference = not_inferred.merge(inference, left_on=['sample_id', 'instance_number'], right_on=['sample_id', 'instance'])
common_bad_cases = not_inferred.merge(bad_cases, left_on=['sample_id', 'instance_number'], right_on=['sample_id', 'instance_number'])
lack_inference = not_inferred[~((not_inferred['sample_id'].isin(common_inference['sample_id']))&(not_inferred['instance_number'].isin(common_inference['instance_number'])))]
lack_inference = not_inferred[~((not_inferred['sample_id'].isin(common_bad_cases['sample_id']))&(not_inferred['instance_number'].isin(common_bad_cases['instance_number'])))]
# %%
hd51 = h5py.File('/mnt/disks/pdiachil-t1map/predictions3/ML4H_mdrk_ukb__cardiac_t1_weighted__predictions__5e806c4c75fa47d59f3270711fc35106.h5', 'r')

remaining = pd.DataFrame({'diff': list(hd51.keys())})
remaining.to_csv('/home/pdiachil/ml/notebooks/mri/remaining_rois.csv', index=False)
# %%
