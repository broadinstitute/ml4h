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
from sklearn.preprocessing import MinMaxScaler
from notebooks.mri.mri_atria import to_xdmf
from parameterize_segmentation import annotation_to_poisson
from ml4h.tensormap.ukb.mri_vtk import _mri_hd5_to_structured_grids, _mri_tensor_4d
from ml4h.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES

import igl
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from google.cloud import storage



# %%
def clip_polydata(polydata, cog_projected, normal, start_z=10.0, step_z=5.0):
    nedges = 0
    iterations = 0
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

        iterations += 1
        if iterations == 10:
            raise ValueError

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

    cog1 = np.mean(points1, axis=0)
    cog2 = np.mean(points2, axis=0)

    cog = 0.5*(cog1+cog2)

    sign1 = np.sign(np.dot(cog1 - cog, normal))
    sign2 = np.sign(np.dot(cog2 - cog, normal))

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

    # clipped_polydata1 = clip_polydata(polydata1, cog, sign1*normal, start_z=0.0, step_z=5.0)
    # clipped_polydata2 = clip_polydata(polydata2, cog, sign2*normal, start_z=0.0, step_z=5.0)

    return plane_source, polydata1, polydata2, sign1*normal, sign2*normal, cog


# %%
def reorient_chambers(atrium, dataset, normal, origin, channel_septum=5):
    z_axis = normal

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
# %%
def vtk_to_igl(vtk_mesh):
    pts = ns.vtk_to_numpy(vtk_mesh.GetPoints().GetData())
    arr = np.zeros_like(pts, dtype=np.double)
    arr[:] = pts
    cells = ns.vtk_to_numpy(vtk_mesh.GetPolys().GetData())
    cells = cells.reshape(-1, 4)[:, 1:]
    ret = igl.write_triangle_mesh('tmp.mesh.off', pts.astype(np.double), cells)
    v, f  = igl.read_triangle_mesh('tmp.mesh.off')
    return v, f


def hd5_to_mesh(hd5_file, t):
    with h5py.File(hd5_file, 'r') as hd5:
        points = np.copy(hd5[f'points_{t}'][()])
        cells = np.copy(hd5[f'cells_{t}'][()])

        polydata = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(points))
        vtk_cells = vtk.vtkCellArray()
        cells[::4] = 3
        vtk_cells.SetCells(len(cells)//4, ns.numpy_to_vtkIdTypeArray(cells))
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(vtk_cells)
    
    return polydata

# %%
def project_to_image(v, triangles, uv, res=256):
    flat_polydata = vtk.vtkPolyData()
    flat_points_vtk = vtk.vtkPoints()
    flat_points = np.zeros((len(uv), 3))
    flat_points[:, :2] = uv
    flat_points_vtk.SetData(ns.numpy_to_vtk(flat_points))
    flat_cells = np.zeros((len(triangles), 4), dtype=np.int64)
    flat_cells[:, 1:] = triangles
    flat_cells[:, 0] = 3
    flat_cells_vtk = vtk.vtkCellArray()
    flat_cells_vtk.SetCells(len(triangles), ns.numpy_to_vtkIdTypeArray(flat_cells.ravel()))
    flat_polydata.SetPoints(flat_points_vtk)
    flat_polydata.SetPolys(flat_cells_vtk)

    coors_array_vtk = ns.numpy_to_vtk(v)
    coors_array_vtk.SetName('coors')
    flat_polydata.GetPointData().AddArray(coors_array_vtk)

    x = np.linspace(np.min(uv[:, 0]), np.max(uv[:, 0]), 256)
    y = np.linspace(np.min(uv[:, 0]), np.max(uv[:, 1]), 256)
    X, Y = np.meshgrid(x, y)
    probe_polydata = vtk.vtkPolyData()
    probe_points = np.zeros((len(x)*len(y), 3))
    probe_points[:, 0] = X.ravel()
    probe_points[:, 1] = Y.ravel()
    probe_points_vtk = vtk.vtkPoints()
    probe_points_vtk.SetData(ns.numpy_to_vtk(probe_points))
    probe_polydata.SetPoints(probe_points_vtk)

    probe_filter = vtk.vtkProbeFilter()
    probe_filter.SetInputData(probe_polydata)
    probe_filter.SetSourceData(flat_polydata)
    probe_filter.Update()

    image = ns.vtk_to_numpy(probe_filter.GetOutput().GetPointData().GetArray('coors')).reshape(256, 256, 3)
    return image

# %%
import subprocess
from parameterize_segmentation import msh_to_vtk, improve_mesh

import logging
logging.getLogger().setLevel('INFO')
hd5s = open('/home/pdiachil/projects/manifests/patient_list.csv', 'r')
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')

# %%
start = int(sys.argv[1])
end = int(sys.argv[2])

# start = 4
# end = start+1
version='fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122'

views = ['lax_4ch', 'lax_3ch', 'lax_2ch']
MRI_SAX_SEGMENTED_CHANNEL_MAP = {'RV_cavity': 6, 'LV_cavity': 5, 'LV_free_wall': 3, 'interventricular_septum': 2, 'RA_cavity': 14}

channels = [
    [   
        MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity'],
        MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
        MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LA_cavity'],
    ],
    [
        MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LV_cavity'],
        MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_Cavity'],
        MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LV_cavity'],
    ],
]

# channels[0] += [[MRI_SAX_SEGMENTED_CHANNEL_MAP['RV_cavity'], MRI_SAX_SEGMENTED_CHANNEL_MAP['RA_cavity']] for d in range(1, 13)]

channel_septum = MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['interventricular_septum']

chambers = ['LA', 'LV']
# views = [f'sax_b{d}' for d in range(1, 13)]
view_format_string = 'cine_segmented_{view}'
annot_format_string = 'cine_segmented_{view}_annotated'
annot_time_format_string = 'cine_segmented_{view}_annotated_{t}'
annot_time_instance_format_string = 'cine_segmented_{view}_annotated_{t}/{instance}'

results = []
for chamber in chambers:
    results.append({'sample_id': [-1]*2*(end-start)})
    results[-1]['instance'] = [-1]*2*(end-start)
    for t in range(MRI_FRAMES):
        results[-1][f'{chamber}_poisson_{t}'] = [-1]*2*(end-start)
        if 'LA' in chamber:
            results[-1][f'{chamber}_z_axis_{t}'] = [-1]*2*(end-start)
            results[-1][f'{chamber}_x_axis_{t}'] = [-1]*2*(end-start)
            results[-1][f'{chamber}_y_axis_{t}'] = [-1]*2*(end-start)


start_time = time.time()
for i, hd5 in enumerate(hd5s):
    # i = start
    # hd5 = f'/mnt/disks/segmented-sax-v20201116-lax-v20201119-petersen/2020-11-20/5362506.hd5'
    # hd5 = f'/mnt/disks/segmented-sax-lax-v20201102/2020-11-02/5362506.hd5'
    # hd5 = f'/mnt/disks/segmented-sax-v20201124-lax-v20201122/2020-11-24/4566955.hd5'
    # hd5 = f'/home/pdiachil/1000800.hd5'
    if i < start:
        continue
    if i == end:
        break    
    sample_id = hd5.split('/')[-1].replace('.hd5\n', '')
    segmented_path = hd5.replace('gs://ml4cvd/', '')
    segmented_path = segmented_path.replace('\n', '')
    blob = bucket.blob(segmented_path)
    blob.download_to_filename('tmp.hd5')

    try:
        with h5py.File('tmp.hd5', 'r') as ff_trad:
            instances = list(ff_trad['ukb_cardiac_mri/cine_segmented_sax_b1_annotated_1'].keys())

        for ii, instance in enumerate(instances):
            annot_datasets = []
            orig_datasets = []
            with h5py.File('tmp.hd5', 'r') as ff_trad:
                for iv, view in enumerate(views):
                    if annot_time_instance_format_string.format(view=view, t=1, instance=instance) not in ff_trad['ukb_cardiac_mri']:
                        views = views[:iv]
                        channels[0] = channels[0][:iv]
                        break
                    annot_datasets.append(
                        _mri_hd5_to_structured_grids(
                            ff_trad, annot_format_string.format(view=view),
                            view_name=view_format_string.format(view=view),
                            instance=instance,
                            concatenate=True, annotation=True,
                            save_path=None, order='F',
                        )[0],
                    )
                    
            poisson_chambers = []
            poisson_volumes = []
            for channel, chamber, result in zip(channels, chambers, results):
                result['sample_id'][i-start] = sample_id
                atria, volumes = annotation_to_poisson(annot_datasets, channel, views, annot_time_format_string, range(MRI_FRAMES))
                poisson_chambers.append(atria)
                poisson_volumes.append(volumes)
                for t, poisson_volume in enumerate(poisson_volumes[-1]):
                    result[f'{chamber}_poisson_{t}'][i-start] = poisson_volume/1000.0

                for t, atrium in enumerate(poisson_chambers[-1]):
                    # writer = vtk.vtkXMLPolyDataWriter()
                    # writer.SetInputData(atrium)
                    # writer.SetFileName(f'/home/pdiachil/projects/chambers/poisson_{chamber}_{sample_id}_{t}.vtp')
                    # writer.Update()
                    write_footer = True if t == MRI_FRAMES-1 else False
                    append = False if t == 0 else True
                    to_xdmf(atrium, f'/home/pdiachil/projects/la/poisson_{version}_{chamber}_{sample_id}_{instance}', append=append, append_time=t, write_footer=write_footer)

            # Extract major axes
            la_hd5_file = f'/home/pdiachil/projects/la/poisson_{version}_LA_{sample_id}_{instance}.hd5'
            lv_hd5_file = f'/home/pdiachil/projects/la/poisson_{version}_LV_{sample_id}_{instance}.hd5'

            ds_4ch = annot_datasets[0]
            threshold = vtk.vtkThreshold()
            threshold.SetInputData(ds_4ch)
            threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, 'cine_segmented_lax_4ch_annotated_0')
            threshold.ThresholdBetween(channel_septum - 0.5, channel_septum + 0.5)
            threshold.Update()

            centers = vtk.vtkCellCenters()
            if threshold.GetOutput().GetNumberOfPoints() == 0:
                centers.SetInputData(ds_4ch)
            else:
                centers.SetInputData(threshold.GetOutput())
            centers.Update()

            septum_cog = np.mean(ns.numpy_to_vtk(centers.GetOutput().GetPoints().GetData()), axis=0)

            for t in range(MRI_FRAMES):
                try:
                    atrium = hd5_to_mesh(la_hd5_file, t)
                    ventricle = hd5_to_mesh(lv_hd5_file, t)
                    plane, clipped_atrium, clipped_ventricle, normal_atrium, normal_ventricle, cog = separation_plane(atrium, ventricle)
                    # clipped_atrium = reorient_chambers(clipped_atrium, ds_4ch, normal_atrium, cog)
                    # clipped_ventricle = reorient_chambers(clipped_ventricle, ds_4ch, normal_ventricle, cog)
                    # better_atrium = improve_mesh(clipped_atrium)
                    better_atrium = clipped_atrium    

                    atrium_points = ns.vtk_to_numpy(better_atrium.GetPoints().GetData())
                    u, s, v = np.linalg.svd(atrium_points)

                    tree = vtk.vtkOBBTree()
                    tree.SetDataSet(better_atrium)
                    tree.BuildLocator()

                    intersect_points = vtk.vtkPoints()
                    intersect_points.SetNumberOfPoints(2)
                    intersect_list = vtk.vtkIdList()

                    parallel = np.dot(v, normal_atrium)
                    
                    # Z axis
                    longit_axis = v[np.argmax(np.abs(parallel))]
                    # longit_axis = normal_atrium
                    atrium_cog = np.mean(atrium_points, axis=0)
                    tree.IntersectWithLine(atrium_cog - longit_axis*1000, atrium_cog + longit_axis*1000, intersect_points, intersect_list)
                    intersect_points_np = ns.vtk_to_numpy(intersect_points.GetData())

                    results[0][f'LA_z_axis_{t}'] = np.linalg.norm(intersect_points_np[1] - intersect_points_np[0])  

                    # intersect_line = vtk.vtkLineSource()
                    # intersect_line.SetPoint1(intersect_points_np[0])
                    # intersect_line.SetPoint2(intersect_points_np[1])

                    # writer = vtk.vtkXMLPolyDataWriter()
                    # writer.SetInputConnection(intersect_line.GetOutputPort())
                    # writer.SetFileName(f'/home/pdiachil/projects/la/line_longit_{sample_id}_{instance}_{t}.vtp')
                    # writer.Update()

                    # Line 2 -- towards septum
                    towards_septum = septum_cog - atrium_cog
                    towards_septum = towards_septum - np.dot(towards_septum, longit_axis)*longit_axis
                    towards_septum /= np.linalg.norm(towards_septum)

                    tree.IntersectWithLine(atrium_cog - towards_septum*100000, atrium_cog + towards_septum*100000, intersect_points, intersect_list)
                    intersect_points_np = ns.vtk_to_numpy(intersect_points.GetData())

                    results[0][f'LA_x_axis_{t}'] = np.linalg.norm(intersect_points_np[1] - intersect_points_np[0]) 

                    # intersect_line = vtk.vtkLineSource()
                    # intersect_line.SetPoint1(intersect_points_np[0])
                    # intersect_line.SetPoint2(intersect_points_np[1])

                    # writer = vtk.vtkXMLPolyDataWriter()
                    # writer.SetInputConnection(intersect_line.GetOutputPort())
                    # writer.SetFileName(f'/home/pdiachil/projects/la/line_horiz_{sample_id}_{instance}_{t}.vtp')
                    # writer.Update()

                    # Line 3 -- orthogonal to line 1 and line 2
                    orthogonal_line = np.cross(longit_axis, towards_septum)

                    tree.IntersectWithLine(atrium_cog - orthogonal_line*1000, atrium_cog + orthogonal_line*1000, intersect_points, intersect_list)
                    intersect_points_np = ns.vtk_to_numpy(intersect_points.GetData())

                    results[0][f'LA_y_axis_{t}'] = np.linalg.norm(intersect_points_np[1] - intersect_points_np[0]) 

                    # intersect_line = vtk.vtkLineSource()
                    # intersect_line.SetPoint1(intersect_points_np[0])
                    # intersect_line.SetPoint2(intersect_points_np[1])

                    # writer = vtk.vtkXMLPolyDataWriter()
                    # writer.SetInputConnection(intersect_line.GetOutputPort())
                    # writer.SetFileName(f'/home/pdiachil/projects/la/line_ortho_{sample_id}_{instance}_{t}.vtp')
                    # writer.Update() 
                except Exception as e:
                    logging.info('Failed at axis generation {sample_id}, {instance}, {t}')
                    continue             


    except Exception as e:
        logging.info(f'{sample_id}, {instance}, {t}')
        continue

for chamber, result in zip(chambers, results):
    results_df = pd.DataFrame(result)
    results_df.to_csv(f'{chamber}_processed_{version}_{start}_{end}.csv', index=False)
