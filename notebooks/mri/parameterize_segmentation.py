import cv2
import numpy as np
from typing import List
from ml4h.tensormap.ukb.mri_vtk import _project_structured_grids
import vtk, vtk.util.numpy_support
from pypoisson import poisson_reconstruction
import logging
import os
import scipy
import h5py
from scipy.optimize import minimize
from sklearn import svm

import subprocess

from ml4h.defines import MRI_FRAMES

def hd5_to_polydatas(fname):
    with h5py.File(fname, 'r') as hd5:
        polydatas = []
        for t in range(MRI_FRAMES):
            polydata = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            points.SetData(vtk.util.numpy_support.numpy_to_vtk(hd5[f'points_{t}'][()]))
            cells_hd5 = hd5[f'cells_{t}'][()]
            cells_numpy = np.zeros((len(cells_hd5)//4, 4), dtype=np.int64)
            cells_numpy[:, 0] = 3
            cells_numpy[:, 1:] = cells_hd5.reshape(-1, 4)[:, 1:]
            cells = vtk.vtkCellArray()
            cells.SetCells(len(cells_hd5)//4, vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells_numpy.ravel()))
            polydata.SetPoints(points)
            polydata.SetPolys(cells)
            polydatas.append(polydata)
    return polydatas


def err_separation(z, points_channel):
    zbig = z*1e4
    weight_0 = len(points_channel)/len(points_channel[points_channel[:, 1]==0])
    weight_1 = len(points_channel)/len(points_channel[points_channel[:, 1]==1])

    wrong_0 = np.sum(points_channel[points_channel[:, 1]==0][:, 0] < zbig)
    wrong_1 = np.sum(points_channel[points_channel[:, 1]==1][:, 0] > zbig)

    return weight_0*wrong_0 + weight_1*wrong_1

def find_separation_coor(points_channel):
    res = minimize(err_separation, 0.0, args=(points_channel), method='Nelder-Mead')
    return res.x*1e4


def clip_by_separation_plane(lax_dataset, sax_datasets, channels, polydatas):
    sax_centers = []
    clipped_polydatas = []
    clipped_volumes = []
    for sax_dataset in sax_datasets:
        sax_points = vtk.util.numpy_support.vtk_to_numpy(sax_dataset.GetPoints().GetData())
        sax_centers.append(np.mean(sax_points, axis=0))

    base_to_apex = (sax_centers[1] - sax_centers[0])/np.linalg.norm(sax_centers[1] - sax_centers[0])
    
    lax_cells = vtk.vtkCellCenters()
    lax_cells.SetInputData(lax_dataset)
    lax_cells.Update()
    lax_points = vtk.util.numpy_support.vtk_to_numpy(lax_cells.GetOutput().GetPoints().GetData())
    for t, polydata in enumerate(polydatas):
        projs_channel = []
        lax_array = vtk.util.numpy_support.vtk_to_numpy(lax_dataset.GetCellData().GetArray(f'cine_segmented_lax_4ch_annotated_{t}'))  
        for i, channel in enumerate(channels):
            points_channel = lax_points[lax_array==channel]
            points_channel = points_channel - sax_centers[0]
            proj_channel = np.zeros((len(points_channel), 2))
            proj_channel[:, 0] = np.dot(points_channel, base_to_apex)
            proj_channel[:, 1] = i
            projs_channel.append(proj_channel)
        points_channel = np.vstack(projs_channel)
        # clf = svm.LinearSVC(max_iter=10000, dual=False, class_weight='balanced')
        # clf.fit(points_channel[:,0].reshape(-1, 1), points_channel[:,1])
        # intercept = clf.intercept_[0]
        intercept = find_separation_coor(points_channel)
        plane = vtk.vtkPlane()
        plane.SetOrigin(sax_centers[0] + base_to_apex * intercept)
        plane.SetNormal(base_to_apex)

        plane_collection = vtk.vtkPlaneCollection()
        plane_collection.AddItem(plane)

        cutter = vtk.vtkClipClosedSurface()
        cutter.SetInputData(polydata)
        cutter.SetClippingPlanes(plane_collection)
        cutter.Update()
        
        # smooth = improve_mesh_ACVD(cutter.GetOutput())

        clipped_polydatas.append(cutter.GetOutput())
        clipped_volumes.append(polydata_to_volume(clipped_polydatas[-1]))

    return clipped_polydatas, clipped_volumes


def vtk_to_msh(vtk_mesh, fname):
    pts = vtk.util.numpy_support.vtk_to_numpy(vtk_mesh.GetPoints().GetData())
    arr = np.zeros((len(pts), 4), dtype=np.double)
    arr[:, :3] = pts

    cells = np.zeros((vtk_mesh.GetNumberOfCells(), 4), dtype=np.int16)
    cells[:, :3] = vtk.util.numpy_support.vtk_to_numpy(vtk_mesh.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    cells += 1

    with open(fname, 'w') as output_file:
        output_file.write('MeshVersionFormatted 2\n\n')
        output_file.write('Dimension 3\n\n')
        output_file.write(f'Vertices\n{len(pts)}\n')
        np.savetxt(output_file, arr, fmt='%f %f %f %d')

        output_file.write(f'\nTriangles\n{len(cells)}\n')
        np.savetxt(output_file, cells, fmt='%d %d %d %d')

        output_file.write('\nEnd')


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
    pts_vtk.SetData(vtk.util.numpy_support.numpy_to_vtk(pts))
    cells_vtk = vtk.vtkCellArray()
    cells_vtk.SetCells(ncells, vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells.ravel()))
    polydata.SetPoints(pts_vtk)
    polydata.SetPolys(cells_vtk)
    return polydata

def improve_mesh(vtk_mesh):
    vtk_to_msh(vtk_mesh, 'tmp.mesh')
    proc = subprocess.Popen(['/home/pdiachil/src/mmg/build/bin/mmgs_O3', '-hausd', '1.0', '-nr', 'tmp.mesh'])
    proc.wait()

    improved_mesh = msh_to_vtk('tmp.o.mesh')
    return improved_mesh


def improve_mesh_ACVD(vtk_mesh, nvertices=3000):
    writer = vtk.vtkPLYWriter()
    writer.SetInputData(vtk_mesh)
    writer.SetFileName('tmp.ply')
    writer.SetFileTypeToASCII()
    writer.Update()
    my_env = os.environ.copy()
    if 'LD_LIBRARY_PATH' not in my_env:
        my_env['LD_LIBRARY_PATH'] = ''
    my_env['LD_LIBRARY_PATH'] = '/home/pdiachil/src/VTK-7.1.1/build/lib:'+my_env['LD_LIBRARY_PATH']
    proc = subprocess.Popen(['/home/pdiachil/src/ACVD/build/bin/ACVD', 'tmp.ply', str(nvertices), '0', '-of', 'tmp_smoothed.ply'], env=my_env)
    proc.wait()

    improved_mesh = vtk.vtkPLYReader()
    improved_mesh.SetFileName('tmp_smoothed.ply')
    improved_mesh.Update()
    return improved_mesh.GetOutput()


def polydata_to_volume(polydata):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(polydata)
    mass.Update()
    return mass.GetVolume()


def intersection_over_union(image, stripe, image_channel, stripe_channel):
    mask = stripe > 0
    union = np.sum(np.logical_and(mask, np.logical_or(image==image_channel, stripe==stripe_channel)))
    if union < 1:
        return -1
    intersection = np.sum(np.logical_and(mask, np.logical_and(image==image_channel, stripe==stripe_channel)))
    iou = intersection/union
    if iou < 1e-3:
        iou = -1
    return iou


def points_normals_to_poisson(
    points: np.ndarray,
    normals: np.ndarray,
)->vtk.vtkPolyData:

    faces, vertices = poisson_reconstruction(points, normals, depth=9)
    # vtk encodes triangle faces as 4 element-arrays starting with 3
    faces_for_vtk = np.zeros((len(faces), 4), dtype=np.int64)
    faces_for_vtk[:, 0] = 3
    faces_for_vtk[:, 1:] = faces

    polydata_points = vtk.vtkPoints()
    polydata_points.SetData(vtk.util.numpy_support.numpy_to_vtk(vertices))
    polydata_cells = vtk.vtkCellArray()
    polydata_cells.SetNumberOfCells(len(faces))
    polydata_cells.SetCells(len(faces), vtk.util.numpy_support.numpy_to_vtkIdTypeArray(faces_for_vtk.ravel()))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(polydata_points)
    polydata.SetPolys(polydata_cells)

    boundary_edges = vtk.vtkFeatureEdges()
    boundary_edges.SetInputData(polydata)
    boundary_edges.BoundaryEdgesOn()
    boundary_edges.FeatureEdgesOff()
    boundary_edges.NonManifoldEdgesOff()
    boundary_edges.ManifoldEdgesOff()
    boundary_strips = vtk.vtkStripper()
    boundary_strips.SetInputConnection(boundary_edges.GetOutputPort())
    boundary_strips.Update()
    boundary_poly = vtk.vtkPolyData()
    boundary_poly.SetPoints(boundary_strips.GetOutput().GetPoints())
    boundary_poly.SetPolys(boundary_strips.GetOutput().GetLines())

    append = vtk.vtkAppendPolyData()
    append.UserManagedInputsOn()
    append.SetNumberOfInputs(2)
    append.SetInputDataByNumber(0, polydata)
    append.SetInputDataByNumber(1, boundary_poly)
    append.Update()

    clean = vtk.vtkCleanPolyData()
    clean.ConvertLinesToPointsOff()
    clean.ConvertPolysToLinesOff()
    clean.ConvertStripsToPolysOff()
    clean.PointMergingOn()
    clean.SetInputConnection(append.GetOutputPort())
    clean.Update()
    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputConnection(clean.GetOutputPort())
    triangle_filter.Update()

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputConnection(triangle_filter.GetOutputPort())
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    return connectivity.GetOutput()


def _error_projection(
    dx, datasets, reference_dataset,
    array_names, reference_array,
    dimensions, reference_dimensions,
    t, channels, reference_channels,
):
    scaled_dx = np.array(dx)*100000.
    iou_datasets = []
    for dataset, array_name in zip(datasets, array_names):
        dataset_array = np.copy(vtk.util.numpy_support.vtk_to_numpy(dataset.GetCellData().GetArray(f'{array_name}_{t}'))).reshape(dimensions[:2])
        shifted_array = scipy.ndimage.shift(dataset_array, scaled_dx, np.int64)
        aligned_array = vtk.util.numpy_support.vtk_to_numpy(dataset.GetCellData().GetArray(f'{array_name}_{t}'))
        aligned_array[:] = shifted_array.ravel()
        projected_array = np.zeros(reference_dimensions)
        _project_structured_grids([dataset], [reference_dataset], array_name, projected_array)
        ious = np.zeros(len(channels))
        for i, (channel, reference_channel) in enumerate(zip(channels, reference_channels)):
            ious[i] = intersection_over_union(
                projected_array[:, :, t], reference_array,
                channel, reference_channel,
            )
        iou_datasets.append(np.mean(ious))
        aligned_array[:] = dataset_array.ravel()
    return 1.0-np.mean(iou_datasets)


def align_datasets(
    datasets, reference_dataset, array_names, reference_array_name,
    channels, reference_channels, t,
):
    reference_dimensions = list(reference_dataset.GetDimensions())
    reference_dimensions = [x-1 for x in reference_dimensions if x > 2]
    reference_dimensions += [MRI_FRAMES]

    dataset_dimensions = list(datasets[0].GetDimensions())
    dataset_dimensions = [x-1 for x in dataset_dimensions if x > 2]
    dataset_dimensions += [MRI_FRAMES]


    # for i in range(MRI_FRAMES):
    #     dataset_array_zeros = np.zeros_like(dataset_array.ravel())
    #     dataset_array_vtk = vtk.util.numpy_support.numpy_to_vtk(dataset_array_zeros)
    #     dataset_array_vtk.SetName(f'aligned_{array_name}_{i}')
    #     dataset.GetCellData().AddArray(dataset_array_vtk)

    reference_array = vtk.util.numpy_support.vtk_to_numpy(reference_dataset.GetCellData().GetArray(f'{reference_array_name}_{t}')).reshape(reference_dimensions[:2])

    dx = [0., 0.]
    initial_error = _error_projection(
        dx, datasets, reference_dataset,
        array_names, reference_array,
        dataset_dimensions, reference_dimensions,
        0, channels, reference_channels,
    )
    if initial_error > 0.7:
        res = minimize(
            _error_projection, dx, method='Nelder-Mead',
            args=(
                datasets, reference_dataset,
                array_names, reference_array,
                dataset_dimensions, reference_dimensions,
                0, channels, reference_channels,
            ),
        )

        dx = res.x * 100000.

    return dx


def shift_datasets(datasets, array_names, dataset_dimensions, dx):
    for dataset, array_name in zip(datasets, array_names):
            for i in range(MRI_FRAMES):
                dataset_array = vtk.util.numpy_support.vtk_to_numpy(dataset.GetCellData().GetArray(f'{array_name}_{i}'))
                shifted_array = scipy.ndimage.shift(dataset_array.reshape(dataset_dimensions[:2]), dx, np.int64)
                # aligned_array = vtk.util.numpy_support.vtk_to_numpy(dataset.GetCellData().GetArray(f'aligned_{array_name}_{i}'))
                dataset_array[:] = shifted_array.ravel()


def annotation_to_discs(
    datasets: List[vtk.vtkStructuredGrid],
    channels: List[int],
    views: List[str],
    format_view: str,
    times: List[int],
    projection_ds_idx: int = None,
    include_projection: bool = True,
    save_path: str = None,
)->List[List[float]]:

    projection_dimensions = list(datasets[projection_ds_idx].GetDimensions())
    projection_dimensions = [x-1 for x in projection_dimensions if x > 2]
    projection_dimensions += [MRI_FRAMES]

    ious = []
    pixels = np.zeros(len(times))
    for t in times:
        ious.append([])
        slices = 0
        for i, (dataset, channel, view) in enumerate(zip(datasets, channels, views)):
            if (projection_ds_idx != i):
                projected_array = np.zeros(projection_dimensions)
                _project_structured_grids([datasets[i]], [datasets[projection_ds_idx]], f'cine_segmented_{view}_annotated', projected_array)
                projection_array = vtk.util.numpy_support.vtk_to_numpy(datasets[projection_ds_idx].GetCellData().GetArray(f'cine_segmented_{views[projection_ds_idx]}_annotated_{t}')).reshape(projection_dimensions[:2])
                iou = intersection_over_union(projection_array, projected_array[:, :, t], channels[projection_ds_idx], channel)
                ious[-1].append(iou)
                if iou >= 0.3:
                    slices += 1
                    dataset_arr = vtk.util.numpy_support.vtk_to_numpy(datasets[i].GetCellData().GetArray(f'cine_segmented_{view}_annotated_{t}'))
                    pixels[t] += np.sum(dataset_arr==channel)
        logging.info(f'Intersection over union: {slices} timestep {t} out of {len(times)}')
    return pixels


def annotation_to_ious(
    datasets: List[vtk.vtkStructuredGrid],
    channels: List[int],
    views: List[str],
    format_view: str,
    times: List[int],
    projection_ds_idx: int = None,
    include_projection: bool = True,
    save_path: str = None,
)->List[List[float]]:

    projection_dimensions = list(datasets[projection_ds_idx].GetDimensions())
    projection_dimensions = [x-1 for x in projection_dimensions if x > 2]
    projection_dimensions += [MRI_FRAMES]

    ious = []

    for t in times:
        ious.append([])
        for i, (dataset, channel, view) in enumerate(zip(datasets, channels, views)):
            if (projection_ds_idx != i):
                projected_array = np.zeros(projection_dimensions)
                _project_structured_grids([datasets[i]], [datasets[projection_ds_idx]], f'cine_segmented_{view}_annotated', projected_array)
                projection_array = vtk.util.numpy_support.vtk_to_numpy(datasets[projection_ds_idx].GetCellData().GetArray(f'cine_segmented_{views[projection_ds_idx]}_annotated_{t}')).reshape(projection_dimensions[:2])
                iou = intersection_over_union(projection_array, projected_array[:, :, t], channels[projection_ds_idx], channel)
                ious[-1].append(iou)
        logging.info(f'Intersection over union: timestep {t} out of {len(times)}')
    return ious


def annotation_to_poisson(
    datasets: List[vtk.vtkStructuredGrid],
    channels: List[int],
    views: List[str],
    format_view: str,
    times: List[int],
    projection_ds_idx: int = None,
    include_projection: bool = True,
    save_path: str = None,
)->List[vtk.vtkPolyData]:
    ncols = 256
    ncolors = 256

    if projection_ds_idx is not None:
        projection_dimensions = list(datasets[projection_ds_idx].GetDimensions())
        projection_dimensions = [x-1 for x in projection_dimensions if x > 2]
        projection_dimensions += [MRI_FRAMES]


    poisson_polydatas = []
    poisson_volumes = []
    for t in times:
        logging.info(f'Poisson surface generation: timestep {t} out of {len(times)}')
        points = []
        normals = []
        for i, (dataset, channel, view) in enumerate(zip(datasets, channels, views)):

            if (projection_ds_idx is not None) and (projection_ds_idx != i):
                projected_array = np.zeros(projection_dimensions)
                _project_structured_grids([datasets[i]], [datasets[projection_ds_idx]], f'cine_segmented_{view}_annotated', projected_array)
                projection_array = vtk.util.numpy_support.vtk_to_numpy(datasets[projection_ds_idx].GetCellData().GetArray(f'cine_segmented_{views[projection_ds_idx]}_annotated_{t}')).reshape(projection_dimensions[:2])
                iou_projection_channel = channels[projection_ds_idx]
                iou_projected_channel = channel
                if isinstance(channels[projection_ds_idx], list):
                    iou_projection_channel = channels[projection_ds_idx][0]
                if isinstance(channel, list):
                    # for channel_elem in channel[1:]:
                    #     projected_array[projected_array==channel_elem] = channel[0]
                    iou_projected_channel = channel[0]
                iou = intersection_over_union(projection_array, projected_array[:, :, t], iou_projection_channel, iou_projected_channel)
                if iou < 0.0 :
                    logging.info(f'Skipping {view}')
                    continue

            arr_annot = vtk.util.numpy_support.vtk_to_numpy(dataset.GetCellData().GetArray(format_view.format(view=view, t=t)))
            arr_annot_copy = np.copy(arr_annot)
            if isinstance(channel, list):
                for channel_elem in channel[1:]:
                    arr_annot[arr_annot==channel_elem] = channel[0]
                channel = channel[0]

            # Extract contours of the segmentation
            im = (arr_annot==channel).reshape(-1, ncols).astype(np.uint8)
            contours, hierarchy  = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
            if len(areas) == 0:
                continue
            max_index = np.argmax(areas)
            app = cv2.drawContours(im, [contours[max_index]], 0, ncolors-1, 1)
            arr_annot[:] = app.ravel() > (ncolors//2)

            threshold = vtk.vtkThreshold()
            threshold.SetInputData(dataset)
            threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, format_view.format(view=view, t=t))
            threshold.ThresholdByUpper(0.5)
            threshold.Update()
            centers = vtk.vtkCellCenters()
            centers.SetInputData(threshold.GetOutput())
            centers.Update()

            dataset_points = vtk.util.numpy_support.vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
            dataset_normals = np.zeros_like(dataset_points)
            dataset_normals[:] = dataset_points - np.mean(dataset_points, axis=0)

            points.append(dataset_points)
            normals.append(dataset_normals)

            arr_annot[:] = arr_annot_copy

        if (projection_ds_idx is not None) and not(include_projection):
            points.pop(projection_ds_idx)
            normals.pop(projection_ds_idx)

        points_arr = np.vstack(points)
        normals_arr = np.vstack(normals)

        tmp_polydata = points_normals_to_poisson(points_arr, normals_arr)
        tmp_points = vtk.util.numpy_support.vtk_to_numpy(tmp_polydata.GetPoints().GetData())
        tmp_cog = np.mean(tmp_points, axis=0)

        for j, (dataset_points, dataset_normals) in enumerate(zip(points, normals)):
            dataset_normals[:] = dataset_points - tmp_cog

            if save_path:
                polydata = vtk.vtkPolyData()
                pts_vtk = vtk.vtkPoints()
                pts_vtk.SetData(vtk.util.numpy_support.numpy_to_vtk(dataset_points))
                polydata.SetPoints(pts_vtk)
                normals_arr = vtk.util.numpy_support.numpy_to_vtk(dataset_normals)
                normals_arr.SetName('Normals')
                polydata.GetPointData().AddArray(normals_arr)
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetInputData(polydata)
                writer.SetFileName(f'{save_path}_{j}_{t}.vtp')
                writer.Update()

        points_arr = np.vstack(points)
        normals_arr = np.vstack(normals)

        poisson_polydatas.append(points_normals_to_poisson(points_arr, normals_arr))
        poisson_volumes.append(polydata_to_volume(poisson_polydatas[-1]))
    return poisson_polydatas, poisson_volumes
