# %%
import h5py
import glob
import time
import sys
import shutil
import pandas as pd

# %%
labels = {
    "Background": { "id": 0, "color": "" },
    "RV Free Wall": { "id": 1, "color": "#f996f1" },
    "Interventricular Septum": { "id": 2, "color": "#bfdad4" },
    "LV Free Wall": { "id": 3, "color": "#f9424b" },
    "LV Pap": { "id": 4, "color": "#3eeaef" },
    "LV Cavity": { "id": 5, "color": "#256676" },
    "RV Cavity": { "id": 6, "color": "#0ba47e" },
    "Thoracic Cavity": { "id": 7, "color": "#984800" },
    "Liver": { "id": 8, "color": "#ffff00" },
    "Stomach": { "id": 9, "color": "#fca283" },
    "Spleen": { "id": 10, "color": "#f82387" },
    "Kidney": { "id": 12, "color": "#ffa500", "sort_order": 10.5 },
    "Body" : { "id": 11, "color": "#4169e1" }
  }

def hex_to_grayscale(png):
    for label in labels:
        pass


# %%
pngs = glob.glob('/home/pdiachil/projects/annotation/jamesp/sax/labels/*.png')
manifest = pd.read_csv('/home/pdiachil/projects/annotation/jamesp/sax/manifest.tsv', sep='\t')

annotated_df = pd.DataFrame({'fpath': pngs})
annotated_df['dicom_file'] = annotated_df['fpath'].str.split('/').str[-1].str.replace('.png.mask.png', '')

annotated_df = annotated_df.merge(manifest, on='dicom_file')

# %%
for i, row in annotated_df.iterrows():
    shutil.copyfile(f'/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/{row["sample_id"]}.hd5', f'/home/pdiachil/projects/sax/{row["sample_id"]}.hd5')
# %%
hd5s = glob.glob('/home/pdiachil/projects/sax/*.hd5')
hd5s = sorted(hd5s)

# %%
%matplotlib inline
import imageio
import shutil
from ml4h.tensorize.tensor_writer_ukbb import tensor_path, first_dataset_at_path, create_tensor_in_hd5
import numpy as np
import matplotlib.pyplot as plt
import PIL

#start = int(sys.argv[1])
#end = int(sys.argv[2])
start = 1
end = 60

start_time = time.time()
for i, hd5 in enumerate(hd5s):
    if i < start:
        continue
    if i == end:
        break
    sample_id = int(hd5.split('/')[-1].replace('.hd5', ''))
    shutil.copyfile(hd5, f'{sample_id}.hd5')
    try:
        with h5py.File(f'{sample_id}.hd5', 'a') as hd5_ff:
            for df, view, version in zip([annotated_df],
                                        ['sax', '3ch', '4ch'],
                                        ['v20200809', 'v20200603', 'v20200816']):

                df_patient = df[df.sample_id==sample_id]
                for nrow, dcm in df_patient.iterrows():
                    segmented_path = f'/home/pdiachil/projects/annotation/jamesp/sax/labels/{dcm.dicom_file}.png.mask.png'
                    png = imageio.imread(segmented_path)
                    png = png[:, :, 0]
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
                        create_tensor_in_hd5(hd5_ff, path_prefix, tensor_name, full_tensor, None)
                    for instance in range(2, 51):
                        tensor_name = series + '_annotated_' + str(instance)
                        tp = tensor_path(path_prefix, tensor_name)
                        if tp in hd5_ff:
                            tensor = first_dataset_at_path(hd5_ff, tp)
                            tensor[:] = full_tensor
                        else:
                            create_tensor_in_hd5(hd5_ff, path_prefix, tensor_name, full_tensor, None)
                    break
    except RecursionError:
        continue
end_time = time.time()
print(end_time-start_time)
# %%
from ml4h.tensormap.ukb.mri_vtk import _mri_hd5_to_structured_grids, _mri_tensor_4d
from notebooks.mri.mri_atria import to_xdmf
import vtk
from parameterize_segmentation import annotation_to_poisson
from ml4h.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP

hd5s = glob.glob('/home/pdiachil/projects/sax/*.hd5')
hd5s = sorted(hd5s)

for hd5_file in hd5s:
    hd5_ff = hd5_file.split('/')[-1]
    if not os.path.isfile(hd5_ff):
        continue
    sample_id = int(hd5_ff.replace('.hd5', ''))
    with h5py.File(hd5_ff) as ff_trad:
        ds_4ch = _mri_hd5_to_structured_grids(
                    ff_trad, 'cine_segmented_lax_4ch_annotated',
                    view_name='cine_segmented_lax_4ch',
                    concatenate=True, annotation=True,
                    save_path=None, order='F',
        )[0]

        ds_4ch_raw = _mri_hd5_to_structured_grids(
                    ff_trad, 'cine_segmented_lax_4ch',
                    view_name='cine_segmented_lax_4ch',
                    concatenate=False, annotation=False,
                    save_path=None, order='F',
        )[0]

        ds_3ch = _mri_hd5_to_structured_grids(
                    ff_trad, 'cine_segmented_lax_3ch_annotated',
                    view_name='cine_segmented_lax_3ch',
                    concatenate=True, annotation=True,
                    save_path=None, order='F',
        )[0]

        ds_2ch = _mri_hd5_to_structured_grids(
                    ff_trad, 'cine_segmented_lax_2ch_annotated',
                    view_name='cine_segmented_lax_2ch',
                    concatenate=True, annotation=True,
                    save_path=None, order='F',
        )[0]

        ds_sax = _mri_hd5_to_structured_grids(
                    ff_trad, 'cine_segmented_sax_b6_annotated',
                    view_name='cine_segmented_sax_b6',
                    concatenate=True, annotation=True,
                    save_path=None, order='F',
        )[0]

        ds_sax_raw = _mri_hd5_to_structured_grids(
                    ff_trad, 'cine_segmented_sax_b6',
                    view_name='cine_segmented_sax_b6',
                    concatenate=False, annotation=False,
                    save_path=None, order='F',
        )[0]

        to_xdmf(ds_4ch, f'{sample_id}_lax_4ch_annotated')
        to_xdmf(ds_4ch_raw, f'{sample_id}_lax_4ch')
        to_xdmf(ds_2ch, f'{sample_id}_lax_2ch_annotated')
        to_xdmf(ds_3ch, f'{sample_id}_lax_3ch_annotated')
        to_xdmf(ds_sax, f'{sample_id}_sax_b6_annotated')
        to_xdmf(ds_sax_raw, f'{sample_id}_sax_b6')

    right_ventricles, volumes = annotation_to_poisson(datasets=[ds_4ch, ds_sax],
                                                    channels=[MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['RV_cavity'], 6],
                                                    views=['lax_4ch', 'sax_b6'],
                                                    format_view='cine_segmented_{view}_annotated_{t}',
                                                    times=range(1))

    left_ventricles, volumes = annotation_to_poisson(datasets=[ds_4ch, ds_3ch, ds_2ch, ds_sax],
                                                    channels=[MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LV_cavity'],
                                                            MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_Cavity'],
                                                            MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LV_cavity'],
                                                            5],
                                                    views=['lax_4ch', 'lax_3ch', 'lax_2ch', 'sax_b6'],
                                                    format_view='cine_segmented_{view}_annotated_{t}',
                                                    times=range(1))

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(right_ventricles[0])
    writer.SetFileName(f'{sample_id}_right_ventricle.vtp')
    writer.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(left_ventricles[0])
    writer.SetFileName(f'{sample_id}_left_ventricle.vtp')
    writer.Update()
# %%
