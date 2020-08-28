# %%
import vtk
import h5py
import time
import sys
import pandas as pd
from parameterize_segmentation import annotation_to_poisson
from ml4cvd.tensor_from_file import _mri_hd5_to_structured_grids, _mri_tensor_4d
from ml4cvd.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES

# %%
start = int(sys.argv[1])
end = int(sys.argv[2])

# %%
views = ['3ch', '2ch', '4ch']
view_format_string = 'cine_segmented_lax_{view}'
annot_format_string = 'cine_segmented_lax_{view}_annotated'
annot_time_format_string = 'cine_segmented_lax_{view}_annotated_{t}'

channels = [MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
            MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LA_cavity'], 
            MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']]

petersen = pd.read_csv('/home/pdiachil/returned_lv_mass.tsv', sep='\t')
petersen = petersen.dropna()

# %%
start_time = time.time()
results = {f'LA_poisson_{t}': [] for t in range(MRI_FRAMES)}
results['sample_id'] = []
for i, (idx, patient) in enumerate(petersen.iterrows()):
    print(i)
    if i < start:
        continue
    if i == end:
        break
    annot_datasets = []
    with h5py.File(f'/mnt/disks/segmented-sax-lax/2020-07-07/{patient.sample_id}.hd5') as ff_trad:
        for view in views:
            annot_datasets.append(_mri_hd5_to_structured_grids(ff_trad, annot_format_string.format(view=view),
                                                            view_name=view_format_string.format(view=view), 
                                                            concatenate=True, annotation=True,
                                                            save_path=None, order='F')[0])    

    poisson_atria, poisson_volumes = annotation_to_poisson(annot_datasets, channels, views, annot_time_format_string, range(MRI_FRAMES))
    results['sample_id'].append(patient.sample_id)
    for t, poisson_volume in enumerate(poisson_volumes):
        results[f'LA_poisson_{t}'].append(poisson_volume/1000.0)

    for t, atrium in enumerate(poisson_atria):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(atrium)                                        
        writer.SetFileName(f'/home/pdiachil/projects/atria/poisson_atria_{patient.sample_id}_{t}.vtp')
        writer.Update()
results_df = pd.DataFrame(results)
results_df.to_csv(f'petersen_processed_{start}_{end}.csv')
end_time = time.time()
print(end_time-start_time)

# %%

dss_annot = []
dss_valve = []
for view in ['3ch', '2ch', '4ch']:
    dss_annot.append(_mri_hd5_to_structured_grids(ff_trad, f'cine_segmented_lax_{view}_annotated',
                                                  view_name=f'cine_segmented_lax_{view}', 
                                                  concatenate=True, annotation=True,
                                                  save_path=None, order='F')[0])

dss_valve.append(_mri_hd5_to_structured_grids(ff_view, f'cine_segmented_lax_inlinevf_zoom_segmented',
                                              view_name=f'cine_segmented_lax_inlinevf', 
                                              concatenate=True, annotation=False,
                                              save_path=None, order='F')[0])

from ml4cvd.tensor_from_file import _mri_project_grids

from utils import to_xdmf
import vtk
from vtk.util import numpy_support as ns
import matplotlib.pyplot as plt
import numpy as np


step = 1
projected_dss = []
directions = np.zeros((MRI_FRAMES//step*3, 3))
radii = np.zeros((MRI_FRAMES//step*3, 3))
center = np.zeros((MRI_FRAMES//step*3, 3))
xs = np.zeros((MRI_FRAMES//step, 3))
ys = np.zeros((MRI_FRAMES//step, 3))
zs = np.zeros((MRI_FRAMES//step, 3))
origins = np.zeros((MRI_FRAMES//step, 3))

for t in range(0, MRI_FRAMES, step):
    print(t)
    for i, (ds_valve, ds_annot, view, la_value) in enumerate(zip(dss_valve[0], dss_annot, 
                                                ['3ch', '2ch', '4ch'],
                                                [MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
                                                MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LA_cavity'],
                                                MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']])):
        projected_dss = _mri_project_grids([ds_valve], ds_annot, 'cine_segmented_lax_inlinevf_zoom_segmented')
            
        arr_valve = ns.vtk_to_numpy(projected_dss[0].GetCellData().GetArray(f'cine_segmented_lax_inlinevf_zoom_segmented_projected_{t}'))
        arr_annot = ns.vtk_to_numpy(projected_dss[0].GetCellData().GetArray(f'cine_segmented_lax_{view}_annotated_{t}'))
        arr_valve[:] = np.logical_and(arr_valve>0.5, arr_annot==la_value)
        
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(projected_dss[0])
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, f"cine_segmented_lax_inlinevf_zoom_segmented_projected_{t}")
        threshold.ThresholdByUpper(0.5)
        threshold.Update()

        # writer = vtk.vtkXMLUnstructuredGridWriter()
        # writer.SetInputConnection(threshold.GetOutputPort())
        # writer.SetFileName(f'/home/pdiachil/projects/atria/view_threshold_{view}_{t}.vtu')
        # writer.Update()
        try:
            centers = vtk.vtkCellCenters()
            centers.SetInputData(threshold.GetOutput())
            centers.Update()
            center_pts = ns.vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
            center_direction = np.mean(center_pts, axis=0)
        
            X = center_pts
            Xm = X - center_direction
            u, s, v = np.linalg.svd((1./X.shape[0])*np.matmul(Xm.T,Xm))
            radii[3*t//step+i] = s[0]/2.0
            directions[3*t//step+i] = u[:, 0]
            center[3*t//step+i] = center_direction
        except AttributeError:
            continue
    zs[t//step] = np.cross(directions[1], directions[0])
    xs[t//step] = directions[0]
    ys[t//step] = np.cross(zs[-1], xs[-1])
    origins[t//step] = np.mean(center, axis=0)

# %%
# sphere = vtk.vtkSphereSource()
# sphere.SetRadius(np.mean(radii))
# sphere.SetCenter([0.0, 0.0, np.mean(radii)*np.sqrt(3)*0.5])
# sphere.SetEndPhi(150)
# sphere.SetThetaResolution(50)
# sphere.SetPhiResolution(50)
# decimate = vtk.vtkDecimatePro()
# decimate.SetInputConnection(sphere.GetOutputPort())
# decimate.Update()
# matrix = np.zeros((4, 4))
# matrix[:3, 0] = x
# matrix[:3, 1] = y
# matrix[:3, 2] = z
# matrix[:3, 3] = origin
# matrix[3, 3] = 1
# transform = vtk.vtkTransform()
# transform.SetMatrix(matrix.ravel().tolist())
# transform_filter = vtk.vtkTransformFilter()
# transform_filter.SetInputConnection(decimate.GetOutputPort())
# transform_filter.SetTransform(transform)
# transform_filter.Update()
# polygon_source = vtk.vtkRegularPolygonSource()
# polygon_source.SetNumberOfSides(50)
# # points = np.vstack(arr for arr in points)
# # polygon_source.SetCenter(*np.mean(points, axis=0).tolist())
# # polygon_source.SetNormal(normal)
# # polygon_source.SetRadius(np.max(radii))
# # polygon_source.Update()
# polygon_writer = vtk.vtkXMLPolyDataWriter()
# polygon_writer.SetInputConnection(transform_filter.GetOutputPort())
# polygon_writer.SetFileName('/home/pdiachil/projects/atria/sphere.vtp')
# polygon_writer.Update()
# # plane_source = vtk.vtkPlaneSource()
# # plane_source.SetCenter(*np.mean(points, axis=0).tolist())
# # plane_source.SetNormal(normal)
# # plane_source.Update()
# # polygon_writer = vtk.vtkXMLPolyDataWriter()
# # polygon_writer.SetInputConnection(plane_source.GetOutputPort())
# # polygon_writer.SetFileName('/home/pdiachil/projects/atria/plane.vtp')
# # polygon_writer.Update()





# %%
%matplotlib inline
import cv2
cnt = 0
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

projected_dss = []
append_vertex_normals = []
append_vertices = []
append_kept_vertices = []
append_kept_normals = []
for t in range(0, MRI_FRAMES, step):
    print(t)
    append_normals = []
    append_filter = vtk.vtkAppendFilter()
    for ds_valve, ds_annot, view, la_value in zip(dss_valve[0], dss_annot, 
                                                ['3ch', '2ch', '4ch'],
                                                [MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
                                                MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LA_cavity'],
                                                MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']]):
        projected_dss = _mri_project_grids([ds_valve], ds_annot, 'cine_segmented_lax_inlinevf_zoom_segmented')
    
        arr_annot = ns.vtk_to_numpy(projected_dss[0].GetCellData().GetArray(f'cine_segmented_lax_{view}_annotated_{t}'))        
        im = (arr_annot==la_value).reshape(-1, 256).astype(np.uint8)
        contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        app = cv2.drawContours(im, contours, 0, 255, 1)
        # f, ax = plt.subplots()
        # plt.imshow(app)
        arr_annot[:] = app.ravel() > 128

        threshold = vtk.vtkThreshold()
        threshold.SetInputData(projected_dss[0])
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, f"cine_segmented_lax_{view}_annotated_{t}")
        threshold.ThresholdByUpper(0.5)
        threshold.Update()
        centers = vtk.vtkCellCenters()
        centers.SetInputData(threshold.GetOutput())
        centers.Update()

        append_filter.AddInputConnection(centers.GetOutputPort())
        append_points = ns.vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
        append_normals.append(append_points - np.mean(append_points, axis=0))
        cnt += 1

    append_filter.Update()
    append_vertex_normals.append(np.vstack(append_normals))
    append_vertex_normals[-1] /= np.linalg.norm(append_vertex_normals[-1], axis=1).reshape(-1, 1)
    append_vertices.append(ns.vtk_to_numpy(append_filter.GetOutput().GetPoints().GetData()))
    keep = np.dot((append_vertices[-1] - origins[t//step]), zs[t//step]) > 0.0
    append_kept_vertices.append(append_vertices[-1][keep])
    append_kept_normals.append(append_vertex_normals[-1][keep])
    append_filter.GetOutput().GetPoints().SetData(ns.numpy_to_vtk(append_kept_vertices[-1]))
    # reconstruction = vtk.vtkSurfaceReconstructionFilter()
    # reconstruction.SetInputConnection(append_filter.GetOutputPort())
    # reconstruction.SetNeighborhoodSize(50)
    # reconstruction.SetSampleSpacing(1.0)
    # reconstruction.Update()
    # image = vtk.vtkXMLImageDataWriter()
    # image.SetInputConnection(reconstruction.GetOutputPort())
    # image.SetFileName('/home/pdiachil/projects/atria/reconstructed.vti')
    # image.Update()
    # append_writer = vtk.vtkXMLUnstructuredGridWriter()
    # append_writer.SetInputConnection(append_filter.GetOutputPort())
    # append_writer.SetFileName(f'/home/pdiachil/projects/atria/append_{t}.vtu')
    # append_writer.Update()

# %%
from pypoisson import poisson_reconstruction
poissons = []
for t in range(0, MRI_FRAMES, step):
    faces, vertices = poisson_reconstruction(append_kept_vertices[t//step], append_kept_normals[t//step])
    faces_tmp = np.zeros((len(faces), 4), dtype=np.int64)
    faces_tmp[:, 0] = 3
    faces_tmp[:, 1:] = faces 
    polydata_points = vtk.vtkPoints()
    polydata_points.SetData(ns.numpy_to_vtk(vertices))
    polydata_cells = vtk.vtkCellArray()
    polydata_cells.SetNumberOfCells(len(faces))
    polydata_cells.SetCells(len(faces), ns.numpy_to_vtkIdTypeArray(faces_tmp.ravel()))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(polydata_points)
    polydata.SetPolys(polydata_cells)

    plane = vtk.vtkPlane()
    plane.SetOrigin(origins[t//step]+3*zs[t//step])
    plane.SetNormal(zs[t//step])

    clip = vtk.vtkClipPolyData()
    clip.SetInputData(polydata)
    clip.SetClipFunction(plane)
    clip.Update()

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(clip.GetOutputPort())
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    poisson = vtk.vtkXMLPolyDataWriter()
    poisson.SetInputData(connectivity.GetOutput())
    poisson.SetFileName(f'/home/pdiachil/projects/atria/poisson_{t}.vtp')
    poisson.Update()

# %%
import igl
from scipy.interpolate import Rbf
def vtk_to_igl(vtk_mesh):
    pts = ns.vtk_to_numpy(vtk_mesh.GetPoints().GetData())
    arr = np.zeros_like(pts, dtype=np.double)
    arr[:] = pts
    cells = ns.vtk_to_numpy(vtk_mesh.GetPolys().GetData())
    cells = cells.reshape(-1, 4)[:, 1:]
    return pts.astype(np.double), cells
    
igl_v, igl_f = vtk_to_igl(connectivity.GetOutput())
ret = igl.write_triangle_mesh('/home/pdiachil/projects/atria/mesh.off', igl_v, igl_f)
v, f  = igl.read_triangle_mesh('/home/pdiachil/projects/atria/mesh.off')
bnd = igl.boundary_loop(f)
bnd_uv = igl.map_vertices_to_circle(v, bnd)
bnd_uv_square = np.zeros_like(bnd_uv)
bnd_uv_square[:, 0] = 0.5 * np.sqrt(2 + bnd_uv[:, 0]**2.0 - bnd_uv[:, 1]**2.0 + 2.0*bnd_uv[:, 0]*np.sqrt(2)) \
                      -0.5  * np.sqrt(2 + bnd_uv[:, 0]**2.0 -bnd_uv[:, 1]**2.0 - + 2.0*bnd_uv[:, 0]*np.sqrt(2))
bnd_uv_square[:, 1] = 0.5 * np.sqrt(2 - bnd_uv[:, 0]**2.0 + bnd_uv[:, 1]**2.0 + 2.0*bnd_uv[:, 1]*np.sqrt(2)) \
                      -0.5  * np.sqrt(2 - bnd_uv[:, 0]**2.0 + bnd_uv[:, 1]**2.0 - 2.0*bnd_uv[:, 1]*np.sqrt(2))
uv = igl.harmonic_weights(v, f, bnd, bnd_uv_square, 1)

f, ax = plt.subplots(1, 3)
f.set_size_inches(9, 3)
connectivity_pts = ns.vtk_to_numpy(connectivity.GetOutput().GetPoints().GetData())
for i in range(3):
    ax[i].scatter(uv[:, 0], uv[:, 1], c=connectivity_pts[:, i])
plt.tight_layout()


rbfs = []
for i in range(3):
    rbfs.append(Rbf(uv[:, 0], uv[:, 1], connectivity_pts[:, i]))
U, V = np.meshgrid(np.linspace(-1.0, 1.0, 100), np.linspace(-1.0, 1.0, 100))
colors = np.zeros((U.shape[0], U.shape[1], 3))
f, ax = plt.subplots(1, 3)
f.set_size_inches(9, 3)
for i, rbf in enumerate(rbfs):
    colors[..., i] = rbf(U.ravel(), V.ravel()).reshape(U.shape)
    ax[i].imshow(colors[..., i])

# %%

locator = vtk.vtkPointLocator()
locator.SetDataSet(transform_filter.GetOutput())
locator.BuildLocator()
map_append_to_sphere = np.zeros((len(kept_points),), dtype=np.int64)
mask = np.ones((len(kept_points),), dtype=np.bool)
for i, point in enumerate(kept_points):
    ptid = locator.FindClosestPoint(point)
    mask[i] = ptid in map_append_to_sphere
    map_append_to_sphere[i] = ptid
rbfs = []
map_append_to_sphere = map_append_to_sphere[np.logical_not(mask)]
kept_points = kept_points[np.logical_not(mask)]
for i in range(3):
    rbfs.append(Rbf(uv[map_append_to_sphere, 0], uv[map_append_to_sphere, 1], kept_points[:, i]))
new_sphere_points = np.zeros((len(uv), 3))
for i, rbf in enumerate(rbfs):
    new_sphere_points[:, i] = rbf(uv[:, 0], uv[:, 1])
transform_filter.GetOutput().GetPoints().SetData(ns.numpy_to_vtk(new_sphere_points))
writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputConnection(transform_filter.GetOutputPort())
writer.SetFileName('/home/pdiachil/projects/atria/deformed_sphere.vtp')
writer.Update()
# %%
plt.scatter(uv[map_append_to_sphere, 0], uv[map_append_to_sphere, 1], c=kept_points[:, 2])

# %%
