import h5py
import glob
from ml4cvd.tensor_from_file import _mri_hd5_to_structured_grids
from ml4cvd.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES
from ml4cvd.tensor_from_file import _mri_project_grids
from vtk.util import numpy_support as ns
import vtk
import sys
import numpy as np
import pandas as pd

proc = int(sys.argv[1])

hd5s = glob.glob('/mnt/disks/sax-lax-40k-lvm/2020-01-29/*.hd5')
hd5s.sort()

hd5s = np.array_split(hd5s, 256)[proc]

results = {f'dice_{d}': [] for d in range(MRI_FRAMES)}
results['patientid'] = []

for i, hd5 in enumerate(hd5s):
    with h5py.File(hd5, 'r') as ff:
        if 'ukb_cardiac_mri' not in ff:
            continue
        try:
            ds_3ch = _mri_hd5_to_structured_grids(ff, 'cine_segmented_lax_3ch_annotated_', view_name='cine_segmented_lax_3ch', concatenate=True, save_path=None, order='F')
            ds_4ch = _mri_hd5_to_structured_grids(ff, 'cine_segmented_lax_4ch_annotated_', view_name='cine_segmented_lax_4ch', concatenate=True, save_path=None, order='F')
            ds_4ch_3ch = _mri_project_grids(ds_3ch, ds_4ch, 'cine_segmented_lax_3ch_annotated_')
            ds_3ch_4ch = _mri_project_grids(ds_4ch, ds_3ch, 'cine_segmented_lax_4ch_annotated_')
        except:
            continue
        
        results['patientid'].append(hd5.split('/')[-1].replace('.hd5', ''))
        for t in range(MRI_FRAMES):
            arr_4onto3 = ns.vtk_to_numpy(ds_3ch_4ch[0].GetCellData().GetArray(f'cine_segmented_lax_4ch_annotated__projected_{t}'))
            arr_3 = ns.vtk_to_numpy(ds_3ch_4ch[0].GetCellData().GetArray(f'cine_segmented_lax_3ch_annotated__{t}'))
            arr_3onto4 = ns.vtk_to_numpy(ds_4ch_3ch[0].GetCellData().GetArray(f'cine_segmented_lax_3ch_annotated__projected_{t}'))
            arr_4 = ns.vtk_to_numpy(ds_4ch_3ch[0].GetCellData().GetArray(f'cine_segmented_lax_4ch_annotated__{t}'))
            cond1 = arr_4onto3 > 0
            inters1 = np.sum(((arr_4onto3==MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']) & cond1) & \
                             ((arr_3==MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium']) & cond1))
            union1 = np.sum(((arr_4onto3==MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']) & cond1) | \
                            ((arr_3==MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium']) & cond1))
            cond2 = arr_3onto4 > 0
            inters2 = np.sum(((arr_3onto4==MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium']) & cond2) & \
                             ((arr_4==MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']) & cond2))
            union2 = np.sum(((arr_3onto4==MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium']) & cond2) | \
                            ((arr_4==MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']) & cond2))
            results[f'dice_{t}'].append(np.mean([inters1/union1, inters2/union2]))

results_df = pd.DataFrame(results)
results_df.to_csv(f'intersected_mri_{proc}.csv', index=False)
