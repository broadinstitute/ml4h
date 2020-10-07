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
hd5s = petersen.sample_id

# %%
start_time = time.time()
results = []
for chamber in chambers:
    results.append({'sample_id': []})
    for t in range(MRI_FRAMES):
        results[-1][f'{chamber}_poisson_{t}'] = []
for i, hd5 in enumerate(sorted(hd5s)):
    hd5 = f'/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/{hd5}.hd5'
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
            result['sample_id'].append(sample_id)
            atria, volumes = annotation_to_poisson(annot_datasets, channel, views, annot_time_format_string, range(MRI_FRAMES))
            poisson_chambers.append(atria)
            poisson_volumes.append(volumes)
            for t, poisson_volume in enumerate(poisson_volumes[-1]):
                result[f'{chamber}_poisson_{t}'].append(poisson_volume/1000.0)

            for t, atrium in enumerate(poisson_chambers[-1]):
                # writer = vtk.vtkXMLPolyDataWriter()
                # writer.SetInputData(atrium)
                # writer.SetFileName(f'/home/pdiachil/projects/chambers/poisson_{chamber}_{sample_id}_{t}.vtp')
                # writer.Update()
                write_footer = True if t == MRI_FRAMES-1 else False
                append = False if t == 0 else True
                to_xdmf(atrium, f'/home/pdiachil/projects/chambers/poisson_{chamber}_{sample_id}', append=append, append_time=t, write_footer=write_footer)
    except FloatingPointError:
        continue
    # break
for chamber, result in zip(chambers, results):
    results_df = pd.DataFrame(result)
    results_df.to_csv(f'{chamber}_processed_{start}_{end}.csv')
end_time = time.time()
print(end_time-start_time)

# %%
def clip_polydata(polydata, cog_projected, normal, start_z=10.0, step_z=5.0):
    nedges = 0
    dz = start_z
    while nedges != 1:
        plane = vtk.vtkPlane()
        plane.SetOrigin(cog_projected + dz * normal)
        plane.SetNormal(normal)

        cutter = vtk.vtkClipPolyData()
        cutter.SetInputData(polydata)
        cutter.SetClipFunction(plane)
        cutter.Update()

        connectivity = vtk.vtkConnectivityFilter()
        connectivity.SetInputConnection(cutter.GetOutputPort())
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()

        features = vtk.vtkFeatureEdges()
        features.SetInputConnection(connectivity.GetOutputPort())
        features.ManifoldEdgesOff()
        features.NonManifoldEdgesOff()
        features.FeatureEdgesOff()
        features.BoundaryEdgesOn()
        features.Update()

        feature_connectivity = vtk.vtkConnectivityFilter()
        feature_connectivity.SetInputConnection(features.GetOutputPort())
        feature_connectivity.SetExtractionModeToAllRegions()
        feature_connectivity.Update()
        nedges = feature_connectivity.GetNumberOfExtractedRegions()
        dz += step_z
    return connectivity.GetOutput()


def separation_plane(polydata1, polydata2):
    points1 = ns.vtk_to_numpy(polydata1.GetPoints().GetData())
    points2 = ns.vtk_to_numpy(polydata2.GetPoints().GetData())

    y1 = np.zeros((len(points1), 1))
    y2 = np.ones((len(points2), 1))

    # clf = svm.SVC(kernel='linear', C=1000)
    clf = svm.LinearSVC(max_iter=10000, dual=False, class_weight='balanced')
    clf.fit(np.vstack([points1, points2]), np.vstack([y1, y2]))

    normal = np.array(clf.coef_[0])
    normal /= np.linalg.norm(normal)
    intercept = clf.intercept_[0]

    cog = 0.5*np.mean(points1, axis=0) +\
          0.5*np.mean(points2, axis=0)

    cog_projected = cog - normal*np.dot(normal, cog)

    p0 = np.array([0.0, 0.0, intercept/normal[2]])
    p1 = np.array([0.0, intercept/normal[1], 0.0])
    p1 = p0 + (p1-p0) / np.linalg.norm(p1-p0)
    p2 = p0 + np.cross(normal, p1-p0)

    plane_source = vtk.vtkPlaneSource()
    plane_source.SetOrigin(p0)
    plane_source.SetPoint1(p0 + 100.0*(p1-p0))
    plane_source.SetPoint2(p0 + 100.0*(p2-p0))
    plane_source.SetXResolution(100)
    plane_source.SetYResolution(100)
    plane_source.SetCenter(cog)

    clipped_ventricle = clip_polydata(polydata2, cog, normal, start_z=0.0, step_z=5.0)
    clipped_atrium = clip_polydata(polydata1, cog, -normal, start_z=0.0, step_z=5.0)

    return plane_source, clipped_ventricle, clipped_atrium, normal, cog


# %%
def reorient_chambers(atrium, dataset, normal, origin, channel_septum=5):
    z_axis = -normal

    threshold = vtk.vtkThreshold()
    threshold.SetInputData(dataset)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, 'cine_segmented_lax_4ch_annotated_0')
    threshold.ThresholdByUpper(0.5)
    threshold.Update()

    centers = vtk.vtkCellCenters()
    centers.SetInputData(threshold.GetOutput())
    centers.Update()

    cog = np.mean(ns.numpy_to_vtk(centers.GetOutput().GetPoints().GetData()), axis=0)

    x_axis = cog - origin
    x_axis = x_axis - np.dot(x_axis, z_axis)*z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.hstack([x_axis.reshape(-1, 1), y_axis.reshape(-1, 1), z_axis.reshape(-1, 1)])
    atrium_pts = ns.vtk_to_numpy(atrium.GetPoints().GetData())
    atrium_pts[:] = atrium_pts - origin
    atrium_pts[:] = np.dot(R.T, atrium_pts.T).T

    return atrium

# for t, (atrium, ventricle) in enumerate(zip(poisson_chambers[0], poisson_chambers[1])):
#     plane, clipped_ventricle, clipped_atrium, normal, origin = separation_plane(atrium, ventricle)

#     writer = vtk.vtkXMLPolyDataWriter()
#     writer.SetInputConnection(plane.GetOutputPort())
#     writer.SetFileName(f'/home/pdiachil/projects/chambers/plane_{sample_id}_{t}.vtp')
#     writer.Update()

#     writer = vtk.vtkXMLPolyDataWriter()
#     writer.SetInputData(clipped_ventricle)
#     writer.SetFileName(f'/home/pdiachil/projects/chambers/clipped_lv_{sample_id}_{t}.vtp')
#     writer.Update()

#     writer = vtk.vtkXMLPolyDataWriter()
#     clipped_atrium = reorient_chambers(clipped_atrium, annot_datasets[-1], normal, origin)
#     writer.SetInputData(clipped_atrium)
#     writer.SetFileName(f'/home/pdiachil/projects/chambers/clipped_la_reorient_{sample_id}_{t}.vtp')
#     writer.Update()

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
