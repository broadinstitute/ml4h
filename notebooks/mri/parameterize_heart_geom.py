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
from parameterize_segmentation import annotation_to_poisson
from ml4h.tensormap.ukb.mri_vtk import _mri_hd5_to_structured_grids, _mri_tensor_4d
from ml4h.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES


# %%
hd5s = glob.glob('/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/*.hd5')

# %%
start = int(sys.argv[1])
end = int(sys.argv[2])
# start = 4031939
# end = 4031940

# %%
views = ['3ch', '2ch', '4ch']
view_format_string = 'cine_segmented_lax_{view}'
annot_format_string = 'cine_segmented_lax_{view}_annotated'
annot_time_format_string = 'cine_segmented_lax_{view}_annotated_{t}'

channels = [
    [
        MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
        MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LA_cavity'],
        MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity'],
    ],
    [
        MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_Cavity'],
        MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LV_cavity'],
        MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LV_cavity'],
    ],
    [
        [MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_anteroseptum'], MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_inferior_wall'], MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_Cavity']],
        [MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LV_posterior_wall'], MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LV_anterior_wall'], MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LV_cavity']],
        [MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LV_anterolateral_wall'], MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['interventricular_septum'], MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LV_cavity']],
    ],
]

chambers = ['LA', 'LV', 'LVW']

petersen = pd.read_csv('/home/pdiachil/returned_lv_mass.tsv', sep='\t')
petersen = petersen.dropna()
# hd5s = petersen.sample_id

# %%
start_time = time.time()
results = []
for chamber in chambers:
    results.append({'sample_id': [-1]*(end-start)})
    for t in range(MRI_FRAMES):
        results[-1][f'{chamber}_poisson_{t}'] = [-1]*(end-start)
for i, hd5 in enumerate(sorted(hd5s)):
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
            for view in views:
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
                # to_xdmf(annot_datasets[-1], f'{start}_{view}_annotated')
                # to_xdmf(orig_datasets[-1], f'{start}_{view}_original')
        poisson_chambers = []
        poisson_volumes = []
        for channel, chamber, result in zip(channels, chambers, results):
            result['sample_id'][i] = sample_id
            atria, volumes = annotation_to_poisson(annot_datasets, channel, views, annot_time_format_string, range(MRI_FRAMES))
            poisson_chambers.append(atria)
            poisson_volumes.append(volumes)
            for t, poisson_volume in enumerate(poisson_volumes[-1]):
                result[f'{chamber}_poisson_{t}'][i] = poisson_volume/1000.0

            for t, atrium in enumerate(poisson_chambers[-1]):
                # writer = vtk.vtkXMLPolyDataWriter()
                # writer.SetInputData(atrium)
                # writer.SetFileName(f'/home/pdiachil/projects/chambers/poisson_{chamber}_{sample_id}_{t}.vtp')
                # writer.Update()
                write_footer = True if t == MRI_FRAMES-1 else False
                append = False if t == 0 else True
                to_xdmf(atrium, f'/home/pdiachil/projects/chambers/poisson_{chamber}_{sample_id}', append=append, append_time=t, write_footer=write_footer)
    except:
        continue
    # break
for chamber, result in zip(chambers, results):
    results_df = pd.DataFrame(result)
    results_df.to_csv(f'{chamber}_processed_{start}_{end}.csv')
end_time = time.time()
print(end_time-start_time)





# %%
# import igl
# import matplotlib.pyplot as plt
# from scipy.interpolate import Rbf
# def vtk_to_igl(vtk_mesh):
#     pts = ns.vtk_to_numpy(vtk_mesh.GetPoints().GetData())
#     arr = np.zeros_like(pts, dtype=np.double)
#     arr[:] = pts
#     cells = ns.vtk_to_numpy(vtk_mesh.GetPolys().GetData())
#     cells = cells.reshape(-1, 4)[:, 1:]
#     ret = igl.write_triangle_mesh('tmp.mesh.off', pts.astype(np.double), cells)
#     v, f  = igl.read_triangle_mesh('tmp.mesh.off')
#     return v, f

# for t, (atrium, ventricle) in enumerate(zip(poisson_chambers[0], poisson_chambers[1])):
#     plane, clipped_ventricle, clipped_atrium = separation_plane(atrium, ventricle)

#     for label, surface, cmap in zip(['LV', 'LA'],
#                                     [clipped_ventricle, clipped_atrium],
#                                     ['Blues', 'Reds']):
#         v, triangles  = vtk_to_igl(surface)
#         bnd = igl.boundary_loop(triangles)

#         b = np.array([2, 1])
#         b[0] = bnd[0]
#         b[1] = bnd[int(bnd.size / 2)]
#         bc = np.array([[0.0, 0.0], [1.0, 0.0]])

#         # LSCM parametrization
#         _, uv = igl.lscm(v, triangles, b, bc)
#         f, ax = plt.subplots(1, 3)
#         f.set_size_inches(9, 3)
#         for i in range(3):
#             ax[i].tripcolor(uv[:, 0], uv[:, 1], triangles, v[:, i], edgecolor='k', cmap=cmap)
#         f.savefig(f'/home/pdiachil/projects/chambers/parameterized_{label}_{t}.png', dpi=500)
#         plt.close(f)
#     break
# %%
