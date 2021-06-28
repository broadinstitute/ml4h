# %%
import pandas as pd

df = pd.read_parquet('/mnt/disks/pdiachil-t1map/cardiac_t1_data/BROAD_ml4h_mdrk_ukb_cardiac_t1__meta__5cc27b04b5784b4d96170652bf9026a9.pq')
# %%
import h5py

for i, row in df.iterrows():
    hd5 = h5py.File(f'/mnt/disks/pdiachil-t1map/cardiac_t1_data/{row["files"]}', 'r')
    break    
# %%
import matplotlib.pyplot as plt
f, ax = plt.subplots(1, 2)
ax[0].imshow(hd52['segmentation'][()].argmax(axis=2))
ax[1].imshow(hd51['pred_class'][()][0])
# %%
hd5 = h5py.File('/home/pdiachil/projects/t1map/cardiac_t1_data/1003150.h5', 'r')
# %%
hd51 = h5py.File('/home/pdiachil/projects/t1map/segmentations/old_model/ML4H_mdrk_ukb__cardiac_t1_weighted__6022194__a3dbc8fa1224436099900f10e2b15c89.h5', 'r')
hd52 = h5py.File('/home/pdiachil/projects/t1map/cardiac_t1_data/6022194.h5', 'r')
# %%
df
# %%
masks = h5py.File('/mnt/disks/pdiachil-t1map/predictions/holdout_results.h5py', 'r')
# %%
holdout = pd.read_csv('/mnt/disks/pdiachil-t1map/predictions/evaluate_batch_holdout.tsv', sep='\t')

# %%
import matplotlib.pyplot as plt
sample_id = '1112229'
image_file = f'/mnt/disks/pdiachil-t1map/cardiac_t1_data/{df.loc[sample_id].files}'
prediction_file = f'/mnt/disks/pdiachil-t1map/predictions/dev_results.h5py'
pred_manifest = pd.read_csv(f'/mnt/disks/pdiachil-t1map/predictions/evaluate_batch_dev.tsv', sep='\t')
index = pred_manifest[pred_manifest['ukbid']==int(sample_id)].index[0]
with h5py.File(image_file) as hd5:
    with h5py.File(prediction_file) as hd5_pred:
        image = hd5['image'][()]
        segmentation = hd5['segmentation'][()].argmax(axis=2)
        prediction = hd5_pred['predictions_argmax'][()][index]
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].imshow(segmentation, alpha=0.5)
        ax[1].imshow(image)
        ax[1].imshow(prediction, alpha=0.5)


# %%
hd5_pred = h5py.File(prediction_file)
# %%
pred_manifest
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
hd5 = h5py.File('/mnt/disks/pdiachil-t1map/predictions3/ML4H_mdrk_ukb__cardiac_t1_weighted__predictions__5e806c4c75fa47d59f3270711fc35106.h5', 'r')

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
