# %%
import h5py
ff_trad = h5py.File('/home/pdiachil/projects/atria/2922335.hd5', 'r')
ff_view = h5py.File('/home/pdiachil/projects/atria/2922335_newviews.hd5', 'r')


# %%
from ml4cvd.tensor_from_file import _mri_hd5_to_structured_grids, _mri_tensor_4d
dss_annot = []
dss_valve = []
for view in ['3ch', '2ch', '4ch']:
    dss_annot.append(_mri_hd5_to_structured_grids(ff_trad, f'cine_segmented_lax_{view}_annotated',
                                                  view_name=f'cine_segmented_lax_{view}', 
                                                  concatenate=True, annotation=True,
                                                  save_path=None, order='F'))

dss_valve.append(_mri_hd5_to_structured_grids(ff_view, f'cine_segmented_lax_inlinevf_zoom_segmented',
                                              view_name=f'cine_segmented_lax_inlinevf', 
                                              concatenate=True, annotation=False,
                                              save_path=None, order='F'))

# %%
from ml4cvd.tensor_from_file import _mri_project_grids
from ml4cvd.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES
from utils import to_xdmf
from vtk.util import numpy_support as ns
import matplotlib.pyplot as plt
import numpy as np

MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP = {}
MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'] = 11
projected_dss = []
for ds_valve, ds_annot, view, la_value in zip(dss_valve[0], dss_annot, 
                                              ['3ch', '2ch', '4ch'],
                                              [MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
                                               MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
                                               MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']]):
    projected_dss.append(_mri_project_grids([ds_valve], ds_annot, 'cine_segmented_lax_inlinevf_zoom_segmented'))
    for t in range(MRI_FRAMES):
        arr_valve = ns.vtk_to_numpy(projected_dss[-1][0].GetCellData().GetArray(f'cine_segmented_lax_inlinevf_zoom_segmented_projected_{t}'))
        arr_annot = ns.vtk_to_numpy(projected_dss[-1][0].GetCellData().GetArray(f'cine_segmented_lax_{view}_annotated_{t}'))
        arr_valve[:] = np.logical_and(arr_valve>0.5, arr_annot==la_value)

    to_xdmf(projected_dss[-1][0], f'/home/pdiachil/projects/atria/projected_{view}', 
            array_name='cine_segmented_lax_inlinevf_zoom_segmented_projected')

# %%
import vtk

MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP = {}
MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'] = 11
la_value = MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium']
arr_annot = ns.vtk_to_numpy(dss_annot[0][0].GetCellData().GetArray(f'cine_segmented_lax_2ch_annotated_0'))


# %%
# %matplotlib inline
arr_valve = ns.vtk_to_numpy(dss_valve[0][0].GetCellData().GetArray(f'cine_segmented_lax_inlinevf_zoom_segmented_0'))
plt.imshow(arr_valve.reshape(-1, 256))

# %%
import cv2
im = (arr_annot==la_value).reshape(-1, 256).astype(np.uint8)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
app = cv2.drawContours(im, contours, 0, 255, 1)
plt.imshow(app==255)
# %%
cv2.__version__

# %%
