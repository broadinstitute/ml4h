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

    clipped_polydata1 = clip_polydata(polydata1, cog, sign1*normal, start_z=0.0, step_z=5.0)
    clipped_polydata2 = clip_polydata(polydata2, cog, sign2*normal, start_z=0.0, step_z=5.0)

    return plane_source, clipped_polydata1, clipped_polydata2, sign1*normal, sign2*normal, cog


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


def vtk_to_msh(vtk_mesh, fname):
    pts = ns.vtk_to_numpy(vtk_mesh.GetPoints().GetData())
    arr = np.zeros((len(pts), 4), dtype=np.double)
    arr[:, :3] = pts

    cells = np.zeros((vtk_mesh.GetNumberOfCells(), 4), dtype=np.int16)
    cells[:, :3] = ns.vtk_to_numpy(vtk_mesh.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    cells += 1

    with open(fname, 'w') as output_file:
        output_file.write('MeshVersionFormatted 2\n\n')
        output_file.write('Dimension 3\n\n')
        output_file.write(f'Vertices\n{len(pts)}\n')
        np.savetxt(output_file, arr, fmt='%f %f %f %d')

        output_file.write(f'\nTriangles\n{len(cells)}\n')
        np.savetxt(output_file, cells, fmt='%d %d %d %d')

        output_file.write('\nEnd')





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

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetInputData(polydata)
        writer.SetFileName(f'/home/pdiachil/projects/chambers/ventricle_1000107_{t}.vtp')
        writer.Update()

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

def msh_to_vtk(fname):
    with open(fname, 'r') as input_file:
        for line in input_file:
            if line=='Vertices\n':
                npts = int(input_file.readline())
                pts = np.zeros((npts, 3))
                for jj in range(npts):
                    try:
                        pts[jj, :] = [float(x) for x in input_file.readline().split()[:-1]]
                    except ValueError:
                        print(input_file.readline())
            if 'Triangles' in line:
                ncells = int(input_file.readline())
                cells = np.zeros((ncells, 4), dtype=np.int64)
                for i in range(ncells):
                    cells[i, 1:] = [int(x) for x in input_file.readline().split()[:-1]]
                cells -= 1
                cells[:, 0] = 3
    polydata = vtk.vtkPolyData()
    pts_vtk = vtk.vtkPoints()
    pts_vtk.SetData(ns.numpy_to_vtk(pts))
    cells_vtk = vtk.vtkCellArray()
    cells_vtk.SetCells(ncells, ns.numpy_to_vtkIdTypeArray(cells.ravel()))
    polydata.SetPoints(pts_vtk)
    polydata.SetPolys(cells_vtk)
    return polydata

def improve_mesh(vtk_mesh):
    vtk_to_msh(vtk_mesh, 'tmp.mesh')
    proc = subprocess.Popen(['mmgs_O3', '-hausd', '0.5', '-nr', 'tmp.mesh'])
    proc.wait()

    improved_mesh = msh_to_vtk('tmp.o.mesh')
    return improved_mesh

# %%
la_hd5_file = '/home/pdiachil/projects/chambers/poisson_LA_1000107.hd5'
lv_hd5_file = '/home/pdiachil/projects/chambers/poisson_LV_1000107.hd5'
hd5_file = '/mnt/disks/segmented-sax-lax-v20200901/2020-09-01/1000107.hd5'
with h5py.File(hd5_file) as ff_trad:
    ds_4ch = _mri_hd5_to_structured_grids(
                ff_trad, 'cine_segmented_lax_4ch_annotated',
                view_name='cine_segmented_lax_4ch',
                concatenate=True, annotation=True,
                save_path=None, order='F',
    )[0]

for t in range(MRI_FRAMES):
    atrium = hd5_to_mesh(la_hd5_file, t)
    ventricle = hd5_to_mesh(lv_hd5_file, t)
    plane, clipped_atrium, clipped_ventricle, normal_atrium, normal_ventricle, cog = separation_plane(atrium, ventricle)
    clipped_atrium = reorient_chambers(clipped_atrium, ds_4ch, normal_atrium, cog)
    clipped_ventricle = reorient_chambers(clipped_ventricle, ds_4ch, normal_ventricle, cog)

    better_ventricle = improve_mesh(clipped_ventricle)
    for label, surface, cmap in zip(
        ['LV'],
        [better_ventricle, clipped_atrium],
        ['Blues', 'Reds'],
    ):
        v, triangles  = vtk_to_igl(surface)
        bnd = igl.boundary_loop(triangles)

        scaler = MinMaxScaler()
        scaler.fit(v[bnd])
        v_scaled = scaler.transform(v)

        b = np.zeros(2, dtype=np.int64)
        b[0] = bnd[np.argmax(v[bnd][:, 0])]
        b[1] = bnd[np.argmax(v[bnd][:, 1])]
        bc = np.array([v_scaled[b[0], :2], v_scaled[b[1], :2]])

        # LSCM parametrization
        _, uv = igl.lscm(v, triangles, b, bc)
        image = project_to_image(v, triangles, uv)
        f, ax = plt.subplots()
        ax.imshow(image[:, :, 2], cmap=cmap, vmin=0, vmax=100.0)
        plt.savefig(f'z_coor_{t}.png', dpi=500)
        plt.close(f)




# %%


vtk_to_msh(clipped_ventricle, 'clipped_ventricle.mesh')
poly = msh_to_vtk('clipped_ventricle.o.mesh')
writer = vtk.vtkXMLPolyDataWriter()
writer.SetInputData(poly)
writer.SetFileName('clipped_ventricle.vtp')
writer.Update()
# %%
