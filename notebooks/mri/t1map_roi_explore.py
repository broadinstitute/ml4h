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
