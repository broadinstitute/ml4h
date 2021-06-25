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
# %%
df = pd.read_csv('/home/pdiachil/projects/t1map/inference/all_t1map_inference.csv')
hd5 = h5py.File('/mnt/disks/pdiachil-t1map/predictions3/ML4H_mdrk_ukb__cardiac_t1_weighted__predictions__5e806c4c75fa47d59f3270711fc35106.h5', 'r')
# %%
f, ax = plt.subplots()

ax.hist(df['LV Cavity_model_iqr'], bins=range(0, 3000), color='blue', label='LV')
ax.hist(df['RV Cavity_model_iqr'], bins=range(0, 3000), color='orange', label='RV')
ax.set_xlim([0, 3000])
ax.legend()

# %%
f, ax = plt.subplots()

ax.hist(df['LV Cavity_model'], bins=range(0, 3000), color='blue', label='LV')
ax.hist(df['RV Cavity_model'], bins=range(0, 3000), color='orange', label='RV')
ax.set_xlim([0, 3000])
ax.legend()

# %%
f, ax = plt.subplots(2, 1)

ax[0].hist(df['LV Free Wall_model_iqr'], bins=range(0, 2000), color='blue', label='LV FW IQR')
ax[0].hist(df['Interventricular Septum_model_iqr'], bins=range(0, 2000), color='orange', label='IVS IQR')
ax[0].set_xlim([0, 2000])
ax[0].legend()


ax[1].hist(df['LV Free Wall_model'], bins=range(0, 2000), color='blue', label='LV FW')
ax[1].hist(df['Interventricular Septum_model'], bins=range(0, 2000), color='orange', label='IVS')
ax[1].set_xlim([0, 2000])
ax[1].legend()
# %%
