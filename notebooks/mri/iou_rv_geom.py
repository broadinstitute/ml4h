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
from parameterize_segmentation import annotation_to_poisson, annotation_to_ious
from ml4h.tensormap.ukb.mri_vtk import _mri_hd5_to_structured_grids, _mri_tensor_4d
from ml4h.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES


# %%
import logging
logging.getLogger().setLevel('INFO')
hd5s = glob.glob('/mnt/disks/segmented-sax-lax-v20201102/2020-11-02/*.hd5')

# %%
start = int(sys.argv[1])
end = int(sys.argv[2])

# start = 4
# end = start+1
version='v20201102'
# hd5s = ['/mnt/disks/segmented-sax-lax-v20200901/2020-11-02/2032446.hd5']

# %%
views = ['lax_4ch']
views += [f'sax_b{d}' for d in range(1, 13)]
# views = [f'sax_b{d}' for d in range(1, 10)]
view_format_string = 'cine_segmented_{view}'
annot_format_string = 'cine_segmented_{view}_annotated'
annot_time_format_string = 'cine_segmented_{view}_annotated_{t}'

MRI_SAX_SEGMENTED_CHANNEL_MAP = {'RV_cavity': 6, 'LV_cavity': 5, 'LV_free_wall': 3, 'interventricular_septum': 2}
# MRI_SAX_SEGMENTED_CHANNEL_MAP = {'RV_cavity': 5, 'LV_cavity': 4, 'LV_free_wall': 3, 'interventricular_septum': 2}


channels = [
    [
        MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['RV_cavity'],
    ],
]

channels[0] += [MRI_SAX_SEGMENTED_CHANNEL_MAP['RV_cavity'] for d in range(1, 13)]
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
    for t in range(MRI_FRAMES):
        for sax_view in [f'sax_b{d}' for d in range(1, 13)]:
            results[-1][f'{chamber}_iou_{sax_view}_{t}'] = [-1]*(end-start)

start_time = time.time()
for i, hd5 in enumerate(sorted(hd5s)):
    # i = start
    # hd5 = f'/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/{hd5}.hd5'
    sample_id = hd5.split('/')[-1].replace('.hd5', '')
    if i < start:
        continue
    if i == end:
        break

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
                # orig_datasets.append(
                #     _mri_hd5_to_structured_grids(
                #         ff_trad, view_format_string.format(view=view),
                #         view_name=view_format_string.format(view=view),
                #         concatenate=False, annotation=False,
                #         save_path=None, order='F',
                #     )[0],
                # )
                # to_xdmf(annot_datasets[-1], f'{sample_id}_{view}_annotated', squash=True)
                # to_xdmf(orig_datasets[-1], f'{sample_id}_{view}_original', squash=True)

      # except:
      #     pass

        for channel, chamber, result in zip(channels, chambers, results):
            result['sample_id'][i-start] = sample_id
            ious = annotation_to_ious(
                annot_datasets, channel, views,
                annot_time_format_string,
                range(MRI_FRAMES), 0
                )
            for it, iou_t in enumerate(ious):
                for iv, iou_v in enumerate(iou_t):
                    result[f'{chamber}_iou_sax_b{iv+1}_{it}'][i-start] = iou_v

    except FutureWarning as e:
        logging.info(f'Caught exception at {sample_id}: {e}')
        continue
    # break
for chamber, result in zip(chambers, results):
    results_df = pd.DataFrame(result)
    results_df.to_csv(f'{chamber}_iou_{version}_{start}_{end}.csv', index=False)
end_time = time.time()
logging.info(f'Job for {sample_id} completed. Time elapsed: {end_time-start_time:.0f} s')


