# %%
import h5py
ff_trad = h5py.File('/home/pdiachil/projects/atria/2922335.hd5', 'r')
ff_view = h5py.File('/home/pdiachil/projects/atria/2922335_newviews.hd5', 'r')


# %%
from ml4cvd.tensor_from_file import _mri_hd5_to_structured_grids, _mri_tensor_4d
dss_annot = []
dss_valve = []
for view in ['3ch', '4ch']:
    dss_annot.append(_mri_hd5_to_structured_grids(ff_trad, f'cine_segmented_lax_{view}_annotated',
                                                  view_name=f'cine_segmented_lax_{view}', 
                                                  concatenate=True, annotation=True,
                                                  save_path=None, order='F'))

dss_valve.append(_mri_hd5_to_structured_grids(ff_view, f'cine_segmented_lax_inlinevf_zoom_segmented',
                                              view_name=f'cine_segmented_lax_inlinevf', 
                                              concatenate=True, annotation=False,
                                              save_path=None, order='F'))

from ml4cvd.tensor_from_file import _mri_project_grids
from ml4cvd.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES
from utils import to_xdmf
import vtk
from vtk.util import numpy_support as ns
import matplotlib.pyplot as plt
import numpy as np

MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP = {}
MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'] = 11
MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['LV_cavity'] = 10
projected_dss = []
directions = []
radii = []
center = []
for ds_valve, ds_annot, view, la_value in zip(dss_valve[0], dss_annot, 
                                              ['3ch', '4ch'],
                                              [MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_Cavity'],
                                               MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LV_cavity']]):
    projected_dss.append(_mri_project_grids([ds_valve], ds_annot, 'cine_segmented_lax_inlinevf_zoom_segmented'))
    for t in range(MRI_FRAMES):
        
        arr_valve = ns.vtk_to_numpy(projected_dss[-1][0].GetCellData().GetArray(f'cine_segmented_lax_inlinevf_zoom_segmented_projected_{t}'))
        arr_annot = ns.vtk_to_numpy(projected_dss[-1][0].GetCellData().GetArray(f'cine_segmented_lax_{view}_annotated_{t}'))
        # arr_valve[:] = np.logical_and(arr_valve>0.5, arr_annot==la_value)
        
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(projected_dss[-1][0])
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, f"cine_segmented_lax_inlinevf_zoom_segmented_projected_{t}")
        threshold.ThresholdByUpper(0.5)
        threshold.Update()

        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputConnection(threshold.GetOutputPort())
        writer.SetFileName(f'/home/pdiachil/projects/ventricles/view_threshold_{view}.vtu')
        writer.Update()
        
        centers = vtk.vtkCellCenters()
        centers.SetInputData(threshold.GetOutput())
        centers.Update()
        center_pts = ns.vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
        center_direction = np.mean(center_pts, axis=0)
        
        X = center_pts
        Xm = X - center_direction
        u, s, v = np.linalg.svd((1./X.shape[0])*np.matmul(Xm.T,Xm))
        radii.append(s[0]/2.0)
        directions.append(u[:, 0])
        center.append(center_direction)
        break
    to_xdmf(projected_dss[-1][0], f'/home/pdiachil/projects/atria/projected_{view}', 
            array_name='cine_segmented_lax_inlinevf_zoom_segmented_projected')
z = np.cross(directions[1], directions[0])
x = directions[0]
y = np.cross(z, x)
origin = np.mean(center, axis=0)
sphere = vtk.vtkSphereSource()
sphere.SetRadius(np.mean(radii))
sphere.SetCenter([0.0, 0.0, np.mean(radii)*np.sqrt(3)*0.5])
sphere.SetEndPhi(150)
sphere.SetThetaResolution(50)
sphere.SetPhiResolution(50)
decimate = vtk.vtkDecimatePro()
decimate.SetInputConnection(sphere.GetOutputPort())
decimate.Update()
matrix = np.zeros((4, 4))
matrix[:3, 0] = x
matrix[:3, 1] = y
matrix[:3, 2] = z
matrix[:3, 3] = origin
matrix[3, 3] = 1
transform = vtk.vtkTransform()
transform.SetMatrix(matrix.ravel().tolist())
transform_filter = vtk.vtkTransformFilter()
transform_filter.SetInputConnection(decimate.GetOutputPort())
transform_filter.SetTransform(transform)
transform_filter.Update()
polygon_source = vtk.vtkRegularPolygonSource()
polygon_source.SetNumberOfSides(50)
# points = np.vstack(arr for arr in points)
# polygon_source.SetCenter(*np.mean(points, axis=0).tolist())
# polygon_source.SetNormal(normal)
# polygon_source.SetRadius(np.max(radii))
# polygon_source.Update()
polygon_writer = vtk.vtkXMLPolyDataWriter()
polygon_writer.SetInputConnection(transform_filter.GetOutputPort())
polygon_writer.SetFileName('/home/pdiachil/projects/atria/sphere.vtp')
polygon_writer.Update()
# plane_source = vtk.vtkPlaneSource()
# plane_source.SetCenter(*np.mean(points, axis=0).tolist())
# plane_source.SetNormal(normal)
# plane_source.Update()
# polygon_writer = vtk.vtkXMLPolyDataWriter()
# polygon_writer.SetInputConnection(plane_source.GetOutputPort())
# polygon_writer.SetFileName('/home/pdiachil/projects/atria/plane.vtp')
# polygon_writer.Update()


# %%
%matplotlib inline
import cv2
append_filter = vtk.vtkAppendFilter()
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
append_normals = []
for ds_valve, ds_annot, view, la_value in zip(dss_valve[0], dss_annot, 
                                              ['3ch', '2ch', '4ch'],
                                              [MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['LV_Cavity'],
                                               MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
                                               MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']]):
    projected_dss.append(_mri_project_grids([ds_valve], ds_annot, 'cine_segmented_lax_inlinevf_zoom_segmented'))
    for t in range(MRI_FRAMES):
        arr_annot = ns.vtk_to_numpy(projected_dss[-1][0].GetCellData().GetArray(f'cine_segmented_lax_{view}_annotated_{t}'))
        print(np.sum(arr_annot))
        im = (arr_annot==la_value).reshape(-1, 256).astype(np.uint8)
        contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        app = cv2.drawContours(im, contours, 0, 255, 1)
        f, ax = plt.subplots()
        plt.imshow(app)
        arr_annot[:] = app.ravel() > 128
        print(np.sum(arr_annot))
        break
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(projected_dss[-1][0])
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
append_normals = np.vstack(append_normals)
append_normals /= np.linalg.norm(append_normals, axis=1).reshape(-1, 1)
append_points = ns.vtk_to_numpy(append_filter.GetOutput().GetPoints().GetData())
keep = np.dot((append_points - origin), z) > 0.0
kept_points = append_points[keep]
append_filter.GetOutput().GetPoints().SetData(ns.numpy_to_vtk(kept_points))
reconstruction = vtk.vtkSurfaceReconstructionFilter()
reconstruction.SetInputConnection(append_filter.GetOutputPort())
reconstruction.SetNeighborhoodSize(50)
reconstruction.SetSampleSpacing(1.0)
reconstruction.Update()
image = vtk.vtkXMLImageDataWriter()
image.SetInputConnection(reconstruction.GetOutputPort())
image.SetFileName('/home/pdiachil/projects/atria/reconstructed.vti')
image.Update()
append_writer = vtk.vtkXMLUnstructuredGridWriter()
append_writer.SetInputConnection(append_filter.GetOutputPort())
append_writer.SetFileName('/home/pdiachil/projects/atria/append.vtu')
append_writer.Update()

# %%
from pypoisson import poisson_reconstruction

faces, vertices = poisson_reconstruction(append_points, append_normals)
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
plane.SetOrigin(origin)
plane.SetNormal(z)

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
poisson.SetFileName('/home/pdiachil/projects/atria/poisson.vtp')
poisson.Update()


# %%
import igl
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
uv = igl.harmonic_weights(v, f, bnd, bnd_uv, 1)

# %%
f, ax = plt.subplots(1, 3)
connectivity_pts = ns.vtk_to_numpy(connectivity.GetOutput().GetPoints().GetData())
for i in range(3):
    ax[i].scatter(uv[:, 0], uv[:, 1], c=connectivity_pts[:, i])

# %%
from scipy.interpolate import Rbf
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
