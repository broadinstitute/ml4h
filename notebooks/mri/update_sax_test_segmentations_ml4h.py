# %%
import h5py
import glob
import time
import sys
import pandas as pd
from google.cloud import storage
import logging
import numpy as np
import blosc
import pandas as pd

sax_version='v202012'
lax_version='v20201122'
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')

df_sax_4ch_petersen = pd.read_csv('/home/pdiachil/projects/manifests/manifest_sax_4ch_petersen.csv', 
                                  usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])

# %%
import imageio
import shutil
from ml4h.tensorize.tensor_writer_ukbb import tensor_path, first_dataset_at_path, create_tensor_in_hd5
import numpy as np

start = int(sys.argv[1])
end = int(sys.argv[2])
# start = 0
# end = 10
df_sax_4ch_petersen = df_sax_4ch_petersen[df_sax_4ch_petersen['instance']==2]

start_time = time.time()
for i, (sample_id, df_hd5) in enumerate(df_sax_4ch_petersen.groupby('sample_id')):
    if i < start:
        continue
    if i == end:
        break

    shutil.copyfile(f'/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/{sample_id}.hd5', f'/home/pdiachil/{sample_id}.hd5')
    try:
        with h5py.File(f'/home/pdiachil/{sample_id}.hd5', 'r+') as hd5_ff:
            segmented_path = f'mdrk/sax_segmentations/sax_slices_jamesp_4b_hyperopted_dropout_v620_converge/{sample_id}_inference__argmax.h5'
            blob = bucket.blob(segmented_path)
            blob.download_to_filename('tmp.h5')
            with h5py.File('tmp.h5', 'r') as segmentation_hd5:
                segmentations = segmentation_hd5['argmax'][()]
            segmentations = np.frombuffer(blosc.decompress(segmentations), dtype=np.uint8).reshape(segmentation_hd5.attrs['shape'][:-1])
            for nrow, dcm in df_hd5.iterrows():
                view = '4ch' if ('LAX_4Ch' in dcm['series']) else 'sax'
                version = lax_version if ('LAX_4Ch' in dcm['series']) else sax_version
                apply_string = 'apply' if ('LAX_4Ch' in dcm['series']) else 'apply'
                segmented_path = f'jamesp/annotation/{view}/{version}/{apply_string}/output/output_pngs/{dcm.dicom_file}.png.mask.png'
                # segmented_path = f'mdrk/sax_segmentations/sax_slice_both_dropout_folders/2020-11-06/{sample_id}/{dcm.dicom_file}.png.mask.png'
                x = 256
                y = 256
                series = dcm.series.lower()
                if view == '4ch':
                    blob = bucket.blob(segmented_path)
                    blob.download_to_filename('tmp.png')
                    png = imageio.imread('tmp.png')
                elif view == 'sax':
                    b_idx = int(series.split('cine_segmented_sax_b')[-1]) - 1
                    t_idx = dcm.instance_number - 1 
                    png = segmentations[b_idx, t_idx, :, :]
                
                path_prefix='ukb_cardiac_mri'
                full_tensor = np.zeros((x, y), dtype=np.float32)
                if len(png.shape) == 3:
                    full_tensor[:png.shape[0], :png.shape[1]] = png[:, :, 0]
                else:
                    full_tensor[:png.shape[0], :png.shape[1]] = png[:, :]
                tensor_name = series + '_annotated_' + str(dcm.instance_number)
                tp = tensor_path(path_prefix, tensor_name)
                if tp in hd5_ff:
                    tensor = first_dataset_at_path(hd5_ff, tp)
                    tensor[:] = full_tensor
                else:
                    create_tensor_in_hd5(hd5_ff, path_prefix, tensor_name, full_tensor)
    except FutureWarning as e:
        logging.warning(f'Caught exception at {sample_id}: {e}')
        continue
end_time = time.time()
print(end_time-start_time)

# %%
