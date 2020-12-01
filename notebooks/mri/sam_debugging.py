# %%
import h5py
import pandas as pd

# %%
import matplotlib.pyplot as plt
import numpy as np


# %%
df_sax_4ch_petersen = pd.read_csv('/home/pdiachil/projects/manifests/manifest_sax.tsv', sep='\t', 
                                  usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
df_sax_4ch_petersen = df_sax_4ch_petersen[df_sax_4ch_petersen['instance']==2]
# %%
import numpy as np
from ml4h.tensorize.tensor_writer_ukbb import tensor_path, first_dataset_at_path, create_tensor_in_hd5
sample_ids = [2865182]
predictions = ['results3']

for sample_id, prediction in zip(sample_ids, predictions):
    with h5py.File(f'/home/pdiachil/projects/chambers/{sample_id}.hd5', 'a') as hd5_ff:
        with h5py.File(f'/home/pdiachil/projects/chambers/{prediction}.h5', 'r') as hd5_prediction:
            df_hd5 = df_sax_4ch_petersen[df_sax_4ch_petersen['sample_id']==sample_id]
            for nrow, dcm in df_hd5.iterrows():
                print(dcm.series, dcm.instance_number)
                x = 256
                y = 256
                path_prefix='ukb_cardiac_mri'
                full_tensor = np.zeros((x, y), dtype=np.float32)
                series = dcm.series.lower()
                series_idx = int(series.split('_b')[-1])
                tensor_name = series + '_annotated_' + str(dcm.instance_number)
                png = np.argmax(hd5_prediction['predictions'][()][series_idx-1, dcm.instance_number-1, ...], axis=-1)
                full_tensor[:png.shape[0], :png.shape[1]] = png[:, :]
                tensor_name = series + '_annotated_' + str(dcm.instance_number)
                tp = tensor_path(path_prefix, tensor_name)
                if tp in hd5_ff:
                    tensor = first_dataset_at_path(hd5_ff, tp)
                    tensor[:] = full_tensor
                else:
                    create_tensor_in_hd5(hd5_ff, path_prefix, tensor_name, full_tensor)


# %%
hd5 = h5py.File('/home/pdiachil/projects/chambers/2865182.hd5', 'r')

f, ax = plt.subplots()
ax.imshow(hd5['ukb_cardiac_mri/cine_segmented_sax_b5/instance_0'][()][:, :, 0])
ax.imshow(hd5['ukb_cardiac_mri/cine_segmented_sax_b5_annotated_1/instance_0'][()]==5, alpha=0.5)
# png = hd5['ukb_cardiac_mri/cine_segmented_sax_b1/2'][()]

hd5.close()

# %%
hd5 = h5py.File('/home/pdiachil/projects/chambers/2865182.hd5', 'r')
hd5_pred = h5py.File('/home/pdiachil/projects/chambers/results3.h5', 'r')
hd5_2 = h5py.File('/home/pdiachil/projects/chambers/2865182_copy.hd5', 'r')

f, ax = plt.subplots()
ax.imshow(hd5['ukb_cardiac_mri/cine_segmented_sax_b5/instance_0'][()][:, :, 0])
# ax[1].imshow(hd5_2['ukb_cardiac_mri/cine_segmented_sax_b5/instance_0'][()][:, :, 0])
# ax[2].imshow(np.argmax(hd5_pred['predictions'][()][4, 0, :, :], axis=-1))
# ax[1].imshow(np.argmax(hd5_pred['predictions'][()][4, 0, :, :], axis=-1)==5, alpha=0.5)
ax.imshow(np.argmax(hd5_pred['predictions'][()][4, 0, :, :], axis=-1)==5, alpha=0.5)
# png = hd5['ukb_cardiac_mri/cine_segmented_sax_b1/2'][()]

hd5.close()
hd5_pred.close()
# %%
