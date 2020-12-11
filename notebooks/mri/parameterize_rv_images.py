# %%
import vtk
import h5py
import time
import glob
import sys
import pandas as pd
import numpy as np
from vtk.util import numpy_support as ns
from sklearn import svm
from notebooks.mri.mri_atria import to_xdmf
from parameterize_segmentation import annotation_to_poisson, clip_by_separation_plane
from ml4h.tensormap.ukb.mri_vtk import _mri_hd5_to_structured_grids, _mri_tensor_4d
from ml4h.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES


# %%
import logging
logging.getLogger().setLevel('INFO')
hd5s = glob.glob('/mnt/disks/segmented-sax-v20201124-lax-v20201122/2020-11-24/*.hd5')
# hd5s = glob.glob('/home/pdiachil/projects/chambers/sam_debugging/*.hd5')
# %%
start = int(sys.argv[1])
end = int(sys.argv[2])

# start = 4
# end = start+1
version='sep_v20201124_v20201122'
# hd5s = ['/mnt/disks/segmented-sax-lax-v20200901/2020-11-02/2032446.hd5']

# %%
views = ['lax_4ch']
views += [f'sax_b{d}' for d in range(1, 13)]
# views = [f'sax_b{d}' for d in range(1, 10)]
view_format_string = 'cine_segmented_{view}'
annot_format_string = 'cine_segmented_{view}_annotated'
annot_time_format_string = 'cine_segmented_{view}_annotated_{t}'

MRI_SAX_SEGMENTED_CHANNEL_MAP = {'RV_cavity': 6, 'LV_cavity': 5, 'LV_free_wall': 3, 'interventricular_septum': 2, 'RA_cavity': 14}
# MRI_SAX_SEGMENTED_CHANNEL_MAP = {'RV_cavity': 5, 'LV_cavity': 4, 'LV_free_wall': 3, 'interventricular_septum': 2}


channels = [
    [
        MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['RV_cavity'],
    ],
]

channels[0] += [[MRI_SAX_SEGMENTED_CHANNEL_MAP['RV_cavity'], MRI_SAX_SEGMENTED_CHANNEL_MAP['RA_cavity']] for d in range(1, 13)]
# channels = [[MRI_SAX_SEGMENTED_CHANNEL_MAP['RV_cavity'] for d in range(1, 10)]]

chambers = ['RV']

#petersen = pd.read_csv('/home/pdiachil/returned_lv_mass.tsv', sep='\t')
#petersen = petersen.dropna()
# hd5s = petersen.sample_id

# %%
from typing import List
import cv2
from parameterize_segmentation import points_normals_to_poisson, polydata_to_volume
from ml4h.tensormap.ukb.mri_vtk import _project_structured_grids
import matplotlib.pyplot as plt
from parameterize_segmentation import intersection_over_union, align_datasets
import scipy


results = []
for chamber in chambers:
    results.append({'sample_id': [-1]*(end-start)})
    results[-1]['dx_0'] = [-1]*(end-start)
    results[-1]['dx_1'] = [-1]*(end-start)
    for t in range(MRI_FRAMES):
        results[-1][f'{chamber}_poisson_{t}'] = [-1]*(end-start)

start_time = time.time()
for i, hd5 in enumerate(sorted(hd5s)):
    # i = start
    # hd5 = f'/mnt/disks/segmented-sax-v20201116-lax-v20201119-petersen/2020-11-20/5362506.hd5'
    # hd5 = f'/mnt/disks/segmented-sax-lax-v20201102/2020-11-02/5362506.hd5'
    hd5 = f'/mnt/disks/segmented-sax-v20201124-lax-v20201122/2020-11-24/3463087.hd5'
    sample_id = hd5.split('/')[-1].replace('.hd5', '')
    if i < start:
        continue
    if i == end:
        break
    print(hd5)
    annot_datasets = []
    orig_datasets = []
    try:
        with h5py.File(hd5) as ff_trad:
            for iv, view in enumerate(views):
                if view_format_string.format(view=view) not in ff_trad['ukb_cardiac_mri']:
                    views = views[:iv]
                    channels[0] = channels[0][:iv]
                    break
                annot_datasets.append(
                    _mri_hd5_to_structured_grids(
                        ff_trad, annot_format_string.format(view=view),
                        view_name=view_format_string.format(view=view),
                        concatenate=True, annotation=True,
                        save_path=None, order='F',
                    )[0],
                )
                orig_datasets.append(
                    _mri_hd5_to_structured_grids(
                        ff_trad, view_format_string.format(view=view),
                        view_name=view_format_string.format(view=view),
                        concatenate=False, annotation=False,
                        save_path=None, order='F',
                    )[0],
                )
                to_xdmf(annot_datasets[-1], f'{sample_id}_{view}_annotated', squash=True)
                to_xdmf(orig_datasets[-1], f'{sample_id}_{view}_original', squash=True)
    except Exception as e:
        logging.info(f'Caught exception at {sample_id}: {e}')
        continue
end_time = time.time()
logging.info(f'Job for {sample_id} completed. Time elapsed: {end_time-start_time:.0f} s')

