# %%
import h5py
import glob
import time
import sys
import pandas as pd

hd5s = glob.glob('/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/*.hd5')
hd5s = sorted(hd5s)
pngs = glob.glob('/mnt/disks/segmented-sax-lax-v20200901/2020-10-26b/pngs/*.png')
df_pngs = pd.DataFrame({'png_file': pngs})
df_pngs['png_file'] = df_pngs['png_file'].str.split('/').str[-1]
# %%
import pandas as pd
df_sax = pd.read_csv('/home/pdiachil/manifest.tsv', sep='\t')
df_sax['png_file'] = df_sax['dicom_file'].str.replace('.dcm', '.dcm.png.mask.png')

df_sax_pngs = df_sax.merge(df_pngs, on ='png_file')
df_sax_pngs = df_sax_pngs[df_sax_pngs['instance']==2]
df_sax_pngs = df_sax_pngs.sort_values(by='sample_id')
# %%
from google.cloud import storage
import imageio
import shutil
from ml4h.tensorize.tensor_writer_ukbb import tensor_path, first_dataset_at_path, create_tensor_in_hd5
import numpy as np

start = int(sys.argv[1])
end = int(sys.argv[2])
# start = 1
# end = 2
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')
start_time = time.time()
for i, (sample_id, df_hd5) in enumerate(df_sax_pngs.groupby('sample_id')):
    if i < start:
        continue
    if i == end:
        break
    view='sax'
    version='v20201026b'
    shutil.copyfile(f'/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/{sample_id}.hd5', f'{sample_id}.hd5')
    try:
        with h5py.File(f'{sample_id}.hd5', 'a') as hd5_ff:        
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
        upload_path = f'pdiachil/segmented_sax_lax_{version}/{sample_id}.hd5'
        upload_blob = bucket.blob(upload_path)
        upload_blob.upload_from_filename(f'{sample_id}.hd5')
    except NotImplementedError:
        continue
end_time = time.time()
print(end_time-start_time)
# %%
hd5_ff= h5py.File(f'{sample_id}.hd5')
# %%
