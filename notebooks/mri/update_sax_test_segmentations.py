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
lax_2ch_version='v20200809'
lax_3ch_version='v20200603'
lax_4ch_version='v20201122'
storage_client = storage.Client('broad-ml4cvd')
logging.getLogger().setLevel('INFO')

df_sax = pd.read_csv('/home/pdiachil/projects/manifests/manifest_sax.tsv', sep='\t',
                     usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
df_2ch = pd.read_csv('/home/pdiachil/projects/manifests/manifest_2ch.tsv', sep='\t',
                     usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
df_3ch = pd.read_csv('/home/pdiachil/projects/manifests/manifest_3ch.tsv', sep='\t',
                     usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
df_4ch = pd.read_csv('/home/pdiachil/projects/manifests/manifest_4ch.tsv', sep='\t',
                     usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])

# %%
df_manifest = pd.concat([df_sax, df_2ch, df_3ch, df_4ch])
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
for i, (sample_id, df_sample_id) in enumerate(df_manifest.groupby('sample_id')):
    if i < start:
        continue
    if i == end:
        break
    shutil.copyfile(f'/mnt/disks/annotated-cardiac-tensors-45k/2020-09-21/{sample_id}.hd5', f'/home/pdiachil/{sample_id}.hd5')
    for j, (instance, df_hd5) in enumerate(df_sample_id.groupby('instance')):        
        try:
            bucket = storage_client.get_bucket('ml4cvd')
            logging.info(f'Processing {sample_id}, instance {instance}')
            segmented_sax = f'jamesp/annotation/sax/{sax_version}/apply/try3-images/output/overlay/{sample_id}_20209_{instance}_0.overlay.tar.gz'
            blob = bucket.blob(segmented_sax)
            blob.download_to_filename('sax.tar.gz')
            segmented_2ch = f'jamesp/annotation/2ch/{lax_2ch_version}/apply/try1/output/overlay/{sample_id}_20208_{instance}_0.overlay.tar.gz'
            blob = bucket.blob(segmented_2ch)
            blob.download_to_filename('2ch.tar.gz')
            segmented_3ch = f'jamesp/annotation/3ch/{lax_3ch_version}/apply/try1/output/overlay/{sample_id}_20208_{instance}_0.overlay.tar.gz'
            blob = bucket.blob(segmented_3ch)
            blob.download_to_filename('3ch.tar.gz')
            segmented_4ch = f'jamesp/annotation/4ch/{lax_4ch_version}/apply/try1/output/overlay/{sample_id}_20208_{instance}_0.overlay.tar.gz'
            blob = bucket.blob(segmented_4ch)
            blob.download_to_filename('4ch.tar.gz')

            bucket = storage_client.get_bucket('bulkml4cvd')
            raw_sax = f'cardiacmri/all/raw/{sample_id}_20209_{instance}_0.zip'
            blob = bucket.blob(raw_sax)
            blob.download_to_filename('raw_sax.zip')
            raw_lax = f'cardiacmri/all/raw/{sample_id}_20208_{instance}_0.zip'
            blob = bucket.blob(raw_lax)
            blob.download_to_filename('raw_lax.zip')
            with h5py.File(f'/home/pdiachil/{sample_id}.hd5', 'a') as hd5_ff:
                for nrow, dcm in df_hd5.iterrows():
                    rawpath = 'raw_lax.zip'
                    if 'LAX_4Ch' in dcm['series']:
                        view = '4ch'
                        version = lax_4ch_version
                        tarpath = '4ch.tar.gz'
                    elif 'LAX_3Ch' in dcm['series']:
                        view = '3ch'
                        version = lax_3ch_version
                        tarpath = '3ch.tar.gz'
                    elif 'LAX_2Ch' in dcm['series']:
                        view = '2ch'
                        version = lax_2ch_version
                        tarpath = '2ch.tar.gz'
                    else:
                        view = 'sax'
                        version = sax_version
                        tarpath = 'sax.tar.gz'
                        rawpath = 'raw_sax.zip'
                    with tarfile.open(tarpath, 'r:gz') as tar:
                        png_file = tar.extractfile(f'{dcm["dicom_file"]}.png.mask.png')
                        png = imageio.imread(png_file)
                    series = dcm.series.lower()
                    with zipfile.ZipFile(rawpath, 'r') as zip_file:
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
