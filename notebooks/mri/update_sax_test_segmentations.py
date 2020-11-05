# %%
import h5py
import glob
import time
import sys
import pandas as pd
from google.cloud import storage
import logging

import pandas as pd
view='sax'
version='v20201102'
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')
# blob = bucket.blob(f'jamesp/annotation/{view}/{version}/manifest.tsv')
# blob.download_to_filename('/home/pdiachil/manifest.tsv')

df_sax = pd.read_csv('/home/pdiachil/manifest.tsv', sep='\t', usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
#df_sax['png_file'] = df_sax['dicom_file'].str.replace('.dcm', '.dcm.png.mask.png')

# df_sax_pngs = df_sax.merge(df_pngs, on ='png_file')
# df_sax_pngs = df_sax_pngs[df_sax_pngs['instance']==2]
# df_sax_pngs = df_sax_pngs.sort_values(by='sample_id')
# %%

import imageio
import shutil
from ml4h.tensorize.tensor_writer_ukbb import tensor_path, first_dataset_at_path, create_tensor_in_hd5
import numpy as np

start = int(sys.argv[1])
end = int(sys.argv[2])
# start = 1
# end = 2
df_sax = df_sax[df_sax['instance']==2]

start_time = time.time()
for i, (sample_id, df_hd5) in enumerate(df_sax.groupby('sample_id')):
    if i < start:
        continue
    if i == end:
        break

    shutil.copyfile(f'/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/{sample_id}.hd5', f'/home/pdiachil/{sample_id}.hd5')
    try:
        with h5py.File(f'/home/pdiachil/{sample_id}.hd5', 'a') as hd5_ff:
            for nrow, dcm in df_hd5.iterrows():
                segmented_path = f'jamesp/annotation/{view}/{version}/apply/output/output_pngs/{dcm.dicom_file}.png.mask.png'
                blob = bucket.blob(segmented_path)
                blob.download_to_filename('tmp.png')
                png = imageio.imread('tmp.png')
                x = 256
                y = 256
                series = dcm.series.lower()
                path_prefix='ukb_cardiac_mri'
                full_tensor = np.zeros((x, y), dtype=np.float32)
                full_tensor[:png.shape[0], :png.shape[1]] = png
                tensor_name = series + '_annotated_' + str(dcm.instance_number)
                tp = tensor_path(path_prefix, tensor_name)
                if tp in hd5_ff:
                    tensor = first_dataset_at_path(hd5_ff, tp)
                    tensor[:] = full_tensor
                else:
                    create_tensor_in_hd5(hd5_ff, path_prefix, tensor_name, full_tensor)
    except Exception as e:
        logging.info(f'Caught exception at {sample_id}: {e}')
        continue
end_time = time.time()
print(end_time-start_time)

# %%
