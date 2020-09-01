# %%
import h5py
import glob
import time

ff = h5py.File('/mnt/disks/segmented-sax-lax/2020-07-07/1036684.hd5', 'r')
hd5s = glob.glob('/mnt/disks/segmented-sax-lax/2020-07-07/*.hd5')
hd5s = sorted(hd5s)
# %%
import pandas as pd
df_2ch = pd.read_csv('/home/pdiachil/manifest_2ch.tsv', sep='\t')
df_3ch = pd.read_csv('/home/pdiachil/manifest_3ch.tsv', sep='\t')
df_4ch = pd.read_csv('/home/pdiachil/manifest_4ch.tsv', sep='\t')

# %%
from google.cloud import storage
import imageio
import shutil
from ml4cvd.tensor_writer_ukbb import tensor_path, first_dataset_at_path
import numpy as np

# start = int(sys.argv[1])
# end = int(sys.argv[2])
start = 1
end = 2
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')
start_time = time.time()
for i, hd5 in enumerate(hd5s):
    if i < start:
        continue
    if i == end:
        break
    sample_id = int(hd5.split('/')[-1].replace('.hd5', ''))
    shutil.copyfile(hd5, f'{sample_id}.hd5')
    with h5py.File(f'{sample_id}.hd5', 'a') as hd5_ff:
        for df, view, version in zip([df_2ch, df_3ch, df_4ch],
                                    ['2ch', '3ch', '4ch'],
                                    ['v20200809', 'v20200603', 'v20200816']):
            print(view)
            df_patient = df[df.sample_id==sample_id]
            for nrow, dcm in df_patient.iterrows():
                segmented_path = f'jamesp/annotation/{view}/{version}/apply/output/output_pngs/{dcm.dicom_file}.png.mask.png'
                blob = bucket.blob(segmented_path)
                blob.download_to_filename('tmp.png')
                png = imageio.imread('tmp.png')
                x = 256
                y = 256
                series = f'cine_segmented_lax_{view}'
                path_prefix='ukb_cardiac_mri'
                full_tensor = np.zeros((x, y), dtype=np.float32)
                full_tensor[:png.shape[0], :png.shape[1]] = png            
                tensor_name = series + '_annotated_' + str(dcm.instance_number)
                tp = tensor_path(path_prefix, tensor_name)
                if tp in hd5_ff:
                    tensor = first_dataset_at_path(hd5_ff, tp)
                    tensor[:] = full_tensor
                else:
                    create_tensor_in_hd5(hd5_ff, path_prefix, tensor_name, full_tensor, stats)
end_time = time.time()
print(end_time-start_time) 
# %%
