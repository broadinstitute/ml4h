# %%
import h5py
import glob
import time
import sys
import pandas as pd
import pydicom
from google.cloud import storage
import logging
import tarfile, zipfile
from ml4h.defines import MRI_PIXEL_WIDTH, MRI_PIXEL_HEIGHT, MRI_SLICE_THICKNESS, MRI_PATIENT_ORIENTATION, MRI_PATIENT_POSITION, HD5_GROUP_CHAR

import pandas as pd

sax_version='v20201202'
lax_version='v20201122'
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')
logging.getLogger().setLevel('INFO')

df_sax_4ch_petersen = pd.read_csv('/home/pdiachil/projects/manifests/manifest_sax_4ch_tar.csv',
                                  usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
# %%

def _save_pixel_dimensions_if_missing(slicer, series, instance, hd5):
    if f'{MRI_PIXEL_WIDTH}_{series}/{instance}' not in hd5:
        if isinstance(hd5[f'{MRI_PIXEL_WIDTH}_{series}'], h5py.Dataset):
            del hd5[f'{MRI_PIXEL_WIDTH}_{series}']
        hd5.create_dataset(f'{MRI_PIXEL_WIDTH}_{series}/{instance}', data=float(slicer.PixelSpacing[0]))
    if f'{MRI_PIXEL_HEIGHT}_{series}/{instance}' not in hd5:
        if isinstance(hd5[f'{MRI_PIXEL_HEIGHT}_{series}'], h5py.Dataset):
            del hd5[f'{MRI_PIXEL_HEIGHT}_{series}']
        hd5.create_dataset(f'{MRI_PIXEL_HEIGHT}_{series}/{instance}', data=float(slicer.PixelSpacing[1]))


def _save_slice_thickness_if_missing(slicer, series, instance, hd5):
    if isinstance(hd5[f'{MRI_SLICE_THICKNESS}_{series}'], h5py.Dataset):
        del hd5[f'{MRI_SLICE_THICKNESS}_{series}']
    if f'{MRI_SLICE_THICKNESS}_{series}/{instance}' not in hd5:
        hd5.create_dataset(f'{MRI_SLICE_THICKNESS}_{series}/{instance}', data=float(slicer.SliceThickness))


def _save_series_orientation_and_position_if_missing(slicer, series, instance, hd5):
    if isinstance(hd5[f'{MRI_PATIENT_ORIENTATION}_{series}'], h5py.Dataset):
        del hd5[f'{MRI_PATIENT_ORIENTATION}_{series}']
    if isinstance(hd5[f'{MRI_PATIENT_POSITION}_{series}'], h5py.Dataset):
        del hd5[f'{MRI_PATIENT_POSITION}_{series}']
    orientation_ds_name = f'{MRI_PATIENT_ORIENTATION}_{series}/{instance}'
    position_ds_name = f'{MRI_PATIENT_POSITION}_{series}/{instance}'
    if orientation_ds_name not in hd5:
        hd5.create_dataset(orientation_ds_name, data=[float(x) for x in slicer.ImageOrientationPatient])
    if position_ds_name not in hd5:
        hd5.create_dataset(position_ds_name, data=[float(x) for x in slicer.ImagePositionPatient])

# %%

import imageio
import shutil
from ml4h.tensorize.tensor_writer_ukbb import tensor_path, first_dataset_at_path, create_tensor_in_hd5
import numpy as np

start = int(sys.argv[1])
end = int(sys.argv[2])
# start = 0
# end = 10

start_time = time.time()
for i, (sample_id, df_sample_id) in enumerate(df_sax_4ch_petersen.groupby('sample_id')):
    if i < start:
        continue
    if i == end:
        break
    shutil.copyfile(f'/mnt/disks/annotated-cardiac-tensors-45k/2020-09-21/{sample_id}.hd5', f'/home/pdiachil/{sample_id}.hd5')
    for j, (instance, df_hd5) in enumerate(df_sample_id.groupby('instance')):        
        try:
            logging.info(f'Processing {sample_id}, instance {instance}')
            segmented_path = f'jamesp/annotation/sax/{sax_version}/apply/try3-images/output/overlay/{sample_id}_20209_{instance}_0.overlay.tar.gz'
            blob = bucket.blob(segmented_path)
            blob.download_to_filename('tmp.tar.gz')
            raw_path_sax = f'/mnt/disks/cardiac-raw/zips/{sample_id}_20209_{instance}_0.zip'
            raw_path_lax = f'/mnt/disks/cardiac-raw/zips/{sample_id}_20208_{instance}_0.zip'
            with h5py.File(f'/home/pdiachil/{sample_id}.hd5', 'a') as hd5_ff:
                for nrow, dcm in df_hd5.iterrows():
                    view = '4ch' if ('LAX_4Ch' in dcm['series']) else 'sax'
                    version = lax_version if ('LAX_4Ch' in dcm['series']) else sax_version
                    apply_string = 'apply' if ('LAX_4Ch' in dcm['series']) else 'apply'
                    if view == '4ch':
                        segmented_path = f'jamesp/annotation/{view}/{version}/{apply_string}/output/output_pngs/{dcm.dicom_file}.png.mask.png'
                        blob = bucket.blob(segmented_path)
                        blob.download_to_filename('tmp.png')
                        png = imageio.imread('tmp.png')                
                    else:  
                        with tarfile.open('tmp.tar.gz', 'r:gz') as tar:
                            png_file = tar.extractfile(f'{dcm["dicom_file"]}.png.mask.png')
                            png = imageio.imread(png_file)
                    raw_path = raw_path_lax if view == '4ch' else raw_path_sax
                    series = dcm.series.lower()
                    with zipfile.ZipFile(raw_path, 'r') as zip_file:
                        dcm_file = zip_file.extract(dcm['dicom_file'])
                        slicer = pydicom.read_file(dcm_file)
                        _save_pixel_dimensions_if_missing(slicer, series, instance, hd5_ff)
                        _save_slice_thickness_if_missing(slicer, series, instance, hd5_ff)
                        _save_series_orientation_and_position_if_missing(slicer, series, instance, hd5_ff)
                    x = 256
                    y = 256
                    path_prefix='ukb_cardiac_mri'
                    full_tensor = np.zeros((x, y), dtype=np.float32)
                    if len(png.shape) == 3:
                        full_tensor[:png.shape[0], :png.shape[1]] = png[:, :, 0]
                    else:
                        full_tensor[:png.shape[0], :png.shape[1]] = png[:, :]
                    tensor_name = f'{series}_annotated_{dcm.instance_number}/{instance}'
                    tp = tensor_path(path_prefix, tensor_name)
                    if tp in hd5_ff:
                        tensor = first_dataset_at_path(hd5_ff, tp)
                        tensor[:] = full_tensor
                    else:
                        create_tensor_in_hd5(hd5_ff, path_prefix, tensor_name, full_tensor)
        except Exception as e:
            logging.warning(f'Caught exception at {sample_id}: {e}')
            continue
end_time = time.time()
print(end_time-start_time)

# %%
