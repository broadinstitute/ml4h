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
hd5s = glob.glob('/mnt/disks/segmented-sax-v20201124-lax-v20201122-petersen/2020-11-24/*.hd5')
# %%
start = int(sys.argv[1])
end = int(sys.argv[2])

# start = 4
# end = start+1
version='separation_v20201124_v20201122'
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

# channels[0] += [[MRI_SAX_SEGMENTED_CHANNEL_MAP['RV_cavity'], MRI_SAX_SEGMENTED_CHANNEL_MAP['RA_cavity']] for d in range(1, 13)]
channels[0] += [MRI_SAX_SEGMENTED_CHANNEL_MAP['RV_cavity'] for d in range(1, 13)]
#channels[0].pop(0)
#views.pop(0)
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
    hd5 = f'/mnt/disks/segmented-sax-v20201124-lax-v20201122/2020-11-24/4566955.hd5'
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



        # Shift datasets
        nsax = len(annot_datasets)-1
        # dx = align_datasets(
        #     annot_datasets[nsax//2-1:nsax//2+2], annot_datasets[0],
        #     [f'cine_segmented_sax_b{i}_annotated' for i in range(nsax//2-1, nsax//2+2)],
        #     'cine_segmented_lax_4ch_annotated',
        #     [MRI_SAX_SEGMENTED_CHANNEL_MAP[key] for key in ['RV_cavity', 'LV_cavity']],
        #     [MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP[key] for key in ['RV_cavity', 'LV_cavity']],
        #     t=0,
        # )        
        dx = np.array([0.0, 0.0])
        logging.info(f'SAX-LAX alignment completed. dx=[{dx[0]}, {dx[1]}]')

        dataset_dimensions = list(annot_datasets[1].GetDimensions())
        dataset_dimensions = [x-1 for x in dataset_dimensions if x > 2]
        dataset_dimensions += [MRI_FRAMES]

        from parameterize_segmentation import shift_datasets
        array_names = [f'cine_segmented_sax_b{i}_annotated' for i in range(1, len(annot_datasets[1:])+1)]
        # shift_datasets(annot_datasets[1:], array_names, dataset_dimensions, dx)
        # for ids in range(1, len(annot_datasets[1:])+1):
        #     to_xdmf(annot_datasets[ids], f'aligned_sax_20201026b_b{ids}_annotated')

        poisson_chambers = []
        poisson_volumes = []
        for channel, chamber, result in zip(channels, chambers, results):
            result['sample_id'][i-start] = sample_id
            result['dx_0'][i-start] = dx[0]
            result['dx_1'][i-start] = dx[1]
            atria, volumes = annotation_to_poisson(
                annot_datasets, channel, views,
                annot_time_format_string,
                range(MRI_FRAMES), 0,
            )
            atria, volumes = clip_by_separation_plane(annot_datasets[0], 
                                                      annot_datasets[1:3],
                                                      [MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['RV_cavity'], 
                                                       MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['RA_cavity']],
                                                      atria)
            poisson_chambers.append(atria)
            poisson_volumes.append(volumes)
            for t, poisson_volume in enumerate(poisson_volumes[-1]):
                result[f'{chamber}_poisson_{t}'][i-start] = poisson_volume/1000.0

            for t, atrium in enumerate(poisson_chambers[-1]):
                write_footer = True if t == MRI_FRAMES-1 else False
                append = False if t == 0 else True
                to_xdmf(atrium, f'/home/pdiachil/projects/chambers/poisson_{version}_{chamber}_{sample_id}', append=append, append_time=t, write_footer=write_footer)
            end_time = time.time()
            logging.info(f'Job for {sample_id} completed. Time elapsed: {end_time-start_time:.0f} s')
            start_time = time.time()
    except Exception as e:
        logging.info(f'Caught exception at {sample_id}: {e}')
        continue
    
for chamber, result in zip(chambers, results):
    results_df = pd.DataFrame(result)
    results_df.to_csv(f'{chamber}_processed_{version}_{start}_{end}.csv', index=False)
