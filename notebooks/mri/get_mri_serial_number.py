# %%
import h5py
import glob
import time
import sys
import os
import pandas as pd
import pydicom
from google.cloud import storage
import logging
import tarfile, zipfile
from ml4h.defines import MRI_PIXEL_WIDTH, MRI_PIXEL_HEIGHT, MRI_SLICE_THICKNESS, MRI_PATIENT_ORIENTATION, MRI_PATIENT_POSITION, HD5_GROUP_CHAR

import pandas as pd
from collections import defaultdict

sax_version='v20201202'
lax_2ch_version='v20200809'
lax_3ch_version='v20200603'
lax_4ch_version='v20201122'
storage_client = storage.Client('broad-ml4cvd')
logging.getLogger().setLevel('INFO')

# df_sax = pd.read_csv('/home/pdiachil/projects/manifests/manifest_sax.tsv', sep='\t',
#                      usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
# df_2ch = pd.read_csv('/home/pdiachil/projects/manifests/manifest_2ch.tsv', sep='\t',
#                      usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
# df_3ch = pd.read_csv('/home/pdiachil/projects/manifests/manifest_3ch.tsv', sep='\t',
#                      usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'])
df_4ch = pd.read_csv(
    '/home/pdiachil/projects/manifests/manifest_4ch.tsv', sep='\t',
    usecols=['sample_id', 'instance', 'dicom_file', 'series', 'instance_number'],
)

# df_t1 = pd.read_csv('/home/pdiachil/segmentation_t1_id_list.csv', usecols=['ID'])
# df_4ch = df_4ch.merge(df_t1, left_on='sample_id', right_on='ID')

# %%
df_manifest = pd.concat([df_4ch])
# %%

def _save_pixel_dimensions_if_missing(slicer, series, instance, hd5):
    if f'{MRI_PIXEL_WIDTH}_{series}/{instance}' not in hd5:
        if (f'{MRI_PIXEL_WIDTH}_{series}' in hd5) and isinstance(hd5[f'{MRI_PIXEL_WIDTH}_{series}'], h5py.Dataset):
            del hd5[f'{MRI_PIXEL_WIDTH}_{series}']
        hd5.create_dataset(f'{MRI_PIXEL_WIDTH}_{series}/{instance}', data=float(slicer.PixelSpacing[0]))
    if f'{MRI_PIXEL_HEIGHT}_{series}/{instance}' not in hd5:
        if (f'{MRI_PIXEL_HEIGHT}_{series}' in hd5) and isinstance(hd5[f'{MRI_PIXEL_HEIGHT}_{series}'], h5py.Dataset):
            del hd5[f'{MRI_PIXEL_HEIGHT}_{series}']
        hd5.create_dataset(f'{MRI_PIXEL_HEIGHT}_{series}/{instance}', data=float(slicer.PixelSpacing[1]))


def _save_slice_thickness_if_missing(slicer, series, instance, hd5):
    if (f'{MRI_SLICE_THICKNESS}_{series}' in hd5) and isinstance(hd5[f'{MRI_SLICE_THICKNESS}_{series}'], h5py.Dataset):
        del hd5[f'{MRI_SLICE_THICKNESS}_{series}']
    if f'{MRI_SLICE_THICKNESS}_{series}/{instance}' not in hd5:
        hd5.create_dataset(f'{MRI_SLICE_THICKNESS}_{series}/{instance}', data=float(slicer.SliceThickness))


def _save_series_orientation_and_position_if_missing(slicer, series, instance, hd5):
    if (f'{MRI_PATIENT_ORIENTATION}_{series}' in hd5) and isinstance(hd5[f'{MRI_PATIENT_ORIENTATION}_{series}'], h5py.Dataset):
        del hd5[f'{MRI_PATIENT_ORIENTATION}_{series}']
    if (f'{MRI_PATIENT_POSITION}_{series}' in hd5) and isinstance(hd5[f'{MRI_PATIENT_POSITION}_{series}'], h5py.Dataset):
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

# start = int(sys.argv[1])
# end = int(sys.argv[2])
start = 0
end = 50000
bucket = storage_client.get_bucket('ml4cvd')
bulk_bucket = storage_client.get_bucket('bulkml4cvd')
start_time = time.time()
results_dic = {'sample_id': [], 'instance': [], 'serial_number': []}
for i, (sample_id, df_sample_id) in enumerate(df_manifest.groupby('sample_id')):
    if i < start:
        continue
    if i == end:
        break
    
    for j, (instance, df_hd5) in enumerate(df_sample_id.groupby('instance')):
        try:
            raw_t1 = f'cardiacmri/all/raw/{sample_id}_20214_{instance}_0.zip'
            blob = bulk_bucket.blob(raw_t1)
            blob.download_to_filename('raw_t1.zip')
            with zipfile.ZipFile('raw_t1.zip', 'r') as zip_file:
                files_in_zip = zip_file.namelist()
                manifest_zip_name = 'manifest.csv' if 'manifest.csv' in files_in_zip else 'manifest.cvs'
                manifest_zip = zip_file.extract(manifest_zip_name)
                df_zip = pd.read_csv(manifest_zip)
                views = defaultdict(list)
                dcm_fnames = defaultdict(list)
                filename_col = 'name' if 'name' in df_zip.columns else 'filename'
                for nrow, dcm in df_zip.iterrows():
                    fname =  dcm[filename_col] if '.dcm' in dcm[filename_col] else dcm.name
                    dcm_file = zip_file.extract(fname)
                    slicer = pydicom.read_file(dcm_file)
                    break
            results_dic['sample_id'].append(sample_id)
            results_dic['instance'].append(instance)
            results_dic['serial_number'].append(slicer.DeviceSerialNumber)
            os.remove(fname)
        except Exception as e:
            logging.warning(f'Caught exception at {sample_id}: {e}')
            continue
end_time = time.time()
print(end_time-start_time)

# # %%
# from ml4h.tensormap.ukb.mri_vtk import _mri_hd5_to_structured_grids
# from notebooks.mri.mri_atria import to_xdmf

# sample_id = 1000107
# fibrosis_views = ['shmolli_192i_sax_b2s_sax_b2s_sax_b2s_t1map']
# views = [f'sax_b{d}' for d in range(1, 10)]

# # views = [f'sax_b{d}' for d in range(1, 10)]
# view_format_string = 'cine_segmented_{view}'
# annot_format_string = 'cine_segmented_{view}_annotated'
# annot_time_format_string = 'cine_segmented_{view}_annotated_{t}'
# annot_time_instance_format_string = 'cine_segmented_{view}_annotated_{t}/{instance}'
# orig_datasets = []
# annot_datasets = []
# with h5py.File(f'{sample_id}.hd5', 'r') as hd5:
#     for view in fibrosis_views:
#         orig_datasets.append(
#             _mri_hd5_to_structured_grids(
#                 hd5, view,
#                 view_name=view,
#                 instance=2,
#                 concatenate=False, annotation=False,
#                 save_path=None, order='F',
#                 mri_frames=2
#             )[0],
#         )
#         to_xdmf(orig_datasets[-1], f'{sample_id}_{view}_original', squash=True, mri_frames=2)
#     for iv, view in enumerate(views):
#         annot_datasets.append(
#             _mri_hd5_to_structured_grids(
#                 hd5, annot_format_string.format(view=view),
#                 view_name=view_format_string.format(view=view),
#                 instance=instance,
#                 concatenate=True, annotation=True,
#                 save_path=None, order='F',
#             )[0],
#         )
#         orig_datasets.append(
#             _mri_hd5_to_structured_grids(
#             hd5, view_format_string.format(view=view),
#             view_name=view_format_string.format(view=view),
#             instance=instance,
#             concatenate=False, annotation=False,
#             save_path=None, order='F',
#             )[0],
#         )
#         to_xdmf(annot_datasets[-1], f'{sample_id}_{view}_annotated', squash=True)
#         to_xdmf(orig_datasets[-1], f'{sample_id}_{view}_original', squash=True)


# # # %%
# # import h5py

# # hd5 = h5py.File('/home/pdiachil/ml/1000800.hd5', 'r')


# # # %%
# # hd5['ukb_mri'].keys()
# # # %%

# %%
