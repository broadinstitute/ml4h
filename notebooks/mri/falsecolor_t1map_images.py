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
phenotypes = pd.read_csv('/home/pdiachil/projects/t1map/pheno.txt', sep='\t')
phenotypes['sample_id'] = phenotypes['IID']
phenotypes['instance'] = 2
covariates = pd.read_csv('/home/pdiachil/projects/t1map/t1map_covariate_disease_serial_mri.csv')
phenotypes_covariates = phenotypes.merge(covariates, on=['sample_id', 'instance'], suffixes=('', '_dup'))
phenotypes_covariates['ivs_lvbp'] = phenotypes_covariates['ivs'] / phenotypes_covariates['lvbp']
outcome_analysis = pd.read_csv('outcome_analysis_file.csv')
# df_t1 = pd.read_csv('/home/pdiachil/segmentation_t1_id_list.csv', usecols=['ID'])
# df_4ch = df_4ch.merge(df_t1, left_on='sample_id', right_on='ID')

# %%
cols = [col for col in outcome_analysis.columns if 'incident' in col]
interesting_outcomes = outcome_analysis[outcome_analysis['sex']==1].sort_values('ivs')[['sample_id', 'ivs', 'lvbp', 'rvbp'] + cols].dropna()

# Healthy 4293714
interesting_outcomes[interesting_outcomes[cols].max(axis=1)<0.5].iloc[50:100]

# %%
# HCM 2757672
interesting_outcomes[interesting_outcomes['hcm_prevalent']>0.5]

# %%
# DCM 3613237
interesting_outcomes[interesting_outcomes['dcm_prevalent']>0.5]

# %%
# HF 4651326
interesting_outcomes[(interesting_outcomes['hf_prevalent']>0.5)][['sample_id', 'hcm_prevalent', 'dcm_prevalent', 'ivs', 'lvbp', 'rvbp']].iloc[-50:]

# %%
# AF 1781863
interesting_outcomes[(interesting_outcomes['sample_id']>0) & (interesting_outcomes['afib_prevalent']>0.5)][['sample_id', 'hcm_prevalent', 'dcm_prevalent', 'ivs', 'lvbp', 'rvbp']].iloc[-50:]

# %%
# AVDCD 2272348
interesting_outcomes[(interesting_outcomes['sample_id']>0) & (interesting_outcomes['av_dcd_prevalent']>0.5)][['sample_id', 'hcm_prevalent', 'dcm_prevalent', 'ivs', 'lvbp', 'rvbp']].iloc[-50:]

# %%
# diabetes 4575767
interesting_outcomes[(interesting_outcomes['sample_id']==4575767) & (interesting_outcomes['dm2_prevalent']>0.5)][['sample_id', 'hcm_prevalent', 'dcm_prevalent', 'ivs', 'lvbp', 'rvbp']].iloc[-50:]

# %%
# htn 1656898
interesting_outcomes[(interesting_outcomes['sample_id']>0) & (interesting_outcomes['htn_prevalent']>0.5)][['sample_id', 'hcm_prevalent', 'dcm_prevalent', 'ivs', 'lvbp', 'rvbp']].iloc[100:150]

# %%
# ckd 3590492
interesting_outcomes[(interesting_outcomes['sample_id']>0) & (interesting_outcomes['ckd_prevalent']>0.5)][['sample_id', 'hcm_prevalent', 'dcm_prevalent', 'ivs', 'lvbp', 'rvbp']].iloc[-50:]

# %%
# as 1485081
interesting_outcomes[(interesting_outcomes['sample_id']>0) & (interesting_outcomes['as_prevalent']>0.5)][['sample_id', 'hcm_prevalent', 'dcm_prevalent', 'ivs', 'lvbp', 'rvbp']].iloc[-50:]

# %%
# ra 5704649
interesting_outcomes[(interesting_outcomes['sample_id']>0) & (interesting_outcomes['ra_prevalent']>0.5)][['sample_id', 'hcm_prevalent', 'dcm_prevalent', 'ivs', 'lvbp', 'rvbp']].iloc[-50:]


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
my_images = {}
start = 0
end = 100
bucket = storage_client.get_bucket('ml4cvd')
bulk_bucket = storage_client.get_bucket('bulkml4cvd')
start_time = time.time()
results_dic = {'sample_id': [], 'instance': [], 'max_pixel': [], 'min_pixel': [], 'mean_pixel': []}
diseases = {'healthy': 4293714, 'hcm': 2757672, 'as': 1485081, 'hf': 1509268, 'av_dcd': 2272348,'diabetes': 4575767, 'af': 1781863, 'htn': 1656898, 'ckd': 3590492, 'ra': 5704649}
df_manifest = df_manifest[df_manifest['sample_id'].isin(diseases.values())]
for i, (sample_id, df_sample_id) in enumerate(df_manifest.groupby('sample_id')):
    if i < start:
        continue
    if i == end:
        break
    print(sample_id)
    hd5_path = f'pdiachil/segmented-sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122/{sample_id}.hd5'
    blob = bucket.blob(hd5_path)
    try:
        blob.download_to_filename(f'/home/pdiachil/ml/notebooks/mri/{sample_id}.hd5')
    except:
        logging.info('Creating new HD5 because it''s missing')
        hd5_ff = h5py.File(f'/home/pdiachil/ml/notebooks/mri/{sample_id}.hd5', 'w')
        hd5_ff.create_group('ukb_cardiac_mri')
        hd5_ff.close()
    for j, (instance, df_hd5) in enumerate(df_sample_id.groupby('instance')):
        try:
            raw_t1 = f'cardiacmri/all/raw/{sample_id}_20214_{instance}_0.zip'
            blob = bulk_bucket.blob(raw_t1)
            blob.download_to_filename('raw_t1.zip')
            with h5py.File(f'/home/pdiachil/ml/notebooks/mri/{sample_id}.hd5', 'a') as hd5_ff:
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
                        series = slicer.SeriesDescription.lower().replace(' ', '_')
                        views[series].append(slicer)
                    best_mean = 0
                    for v in views:
                        mri_shape = (views[v][0].Rows, views[v][0].Columns, len(views[v]))
                        mri_group = 'ukb_cardiac_mri'
                        mri_data = np.zeros((views[v][0].Rows, views[v][0].Columns, 1), dtype=np.float32)
                        for slicer in views[v]:
                            _save_pixel_dimensions_if_missing(slicer, v, instance, hd5_ff)
                            _save_slice_thickness_if_missing(slicer, v, instance, hd5_ff)
                            _save_series_orientation_and_position_if_missing(slicer, v, instance, hd5_ff)
                            slice_index = slicer.InstanceNumber - 1                            
                            if ('t1map' in v) and ('sax' in v):
                                
                            # if ('t1map' in v):
                                cur_mean = np.mean(slicer.pixel_array.astype(np.float32))
                                if cur_mean > best_mean:
                                    print(v)
                                    best_mean = cur_mean
                                    my_image = slicer.pixel_array.astype(np.float32)
                                    if instance == 2:
                                        my_images[sample_id] = slicer.pixel_array.astype(np.float32)
                                    imageio.imwrite(f'/home/pdiachil/ml/notebooks/mri/{sample_id}_{instance}_0.png', slicer.pixel_array.astype(np.float32))
                                    mri_data[:, :, 0] = slicer.pixel_array.astype(np.float32)
                                    if f'{v}' not in hd5_ff[mri_group]:
                                        print('create', mri_data.shape)
                                        create_tensor_in_hd5(hd5_ff, mri_group, f'{v}/{instance}', mri_data)
                                        print(hd5_ff[f'{mri_group}/{v}/{instance}/instance_0'].shape)
                                    else:
                                        print('overwrite')
                                        hd5_ff[f'{mri_group}/{v}/{instance}/instance_0'][...] = mri_data
                        
                # os.remove('raw_t1.zip')
        except Exception as e:
            logging.warning(f'Caught exception at {sample_id}: {e}')
            continue
end_time = time.time()
print(end_time-start_time)


# %%
import imageio
import rawpy
from PIL import Image
images = {}
masks = {}



for disease, sample_id in diseases.items():
    images[disease] = my_images[sample_id]
    masks[disease] = imageio.imread(f'{disease}_mask.png')
    # masks[disease] = Image.fromarray(masks[disease]).resize([images[disease].shape[1], images[disease].shape[0]])
    # masks[disease] = np.array(masks[disease])

# # %%
# %matplotlib inline
# import matplotlib.pyplot as plt
# f, ax = plt.subplots(1, 2)

# mask_healthy = imageio.imread('healthy_mask.png')>0.5
# mask_healthy[:, -15:] = True

# mask_prevalent = imageio.imread('prevalent_mask.png')>0.5
# mask_prevalent[:, -15:] = True


# ax[0].imshow(np.ma.masked_array(images['healthy'], mask=mask_healthy), cmap='gray', vmin=600, vmax=1500)
# ax[0].imshow(np.ma.masked_array(images['healthy'], mask=~mask_healthy), cmap='jet', vmin=600, vmax=1500)
# ax[1].imshow(np.ma.masked_array(images['prevalent'], mask=mask_prevalent), cmap='gray', vmin=600, vmax=1500)
# ax[1].imshow(np.ma.masked_array(images['prevalent'], mask=~mask_prevalent), cmap='jet', vmin=600, vmax=1500)

# ax[0].set_xticks([])
# ax[1].set_xticks([])
# ax[0].set_yticks([])
# ax[1].set_yticks([])
# plt.show()
# f.savefig('HCM_evolution.png', dpi=500)

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import rawpy

for disease in diseases:
    print(disease)
    image = images[disease]
    mask = masks[disease]

    f, ax = plt.subplots()

    mask = mask>0.5
    mask[:, -15:] = True


    ax.imshow(np.ma.masked_array(image, mask=mask), cmap='gray', vmin=600, vmax=1500)
    ax.imshow(np.ma.masked_array(image, mask=~mask), cmap='jet', vmin=600, vmax=1500)
    # ax[1].imshow(np.ma.masked_array(images['prevalent'], mask=mask_prevalent), cmap='gray', vmin=600, vmax=1500)
    # ax[1].imshow(np.ma.masked_array(images['prevalent'], mask=~mask_prevalent), cmap='jet', vmin=600, vmax=1500)

    ax.set_xticks([])
    ax.set_yticks([])
    # ax[0].set_yticks([])
    # ax[1].set_yticks([])
    plt.show()
    f.savefig(f'{disease}_color.png', dpi=500)


# %%
import matplotlib.pyplot as plt
f, ax = plt.subplots(1, 2)

ax[0].imshow(images['healthy'], cmap='gray')
ax[1].imshow(images['prevalent'], cmap='gray')
ax[0].set_xticks([])
ax[1].set_xticks([])
ax[0].set_yticks([])
ax[1].set_yticks([])
plt.show()
f.savefig('HCM_evolution_gray.png', dpi=500)
# %%
imageio.imwrite('prevalent.png', images['prevalent'])
# %%
