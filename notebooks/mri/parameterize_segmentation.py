import cv2
import numpy as np
from typing import List

import vtk, vtk.util.numpy_support
from pypoisson import poisson_reconstruction

from ml4cvd.defines import MRI_FRAMES

def polydata_to_volume(polydata):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(polydata)
    mass.Update()
    return mass.GetVolume()


def points_normals_to_poisson(points: np.ndarray,
                              normals: np.ndarray)->vtk.vtkPolyData:

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

    return triangle_filter.GetOutput()
                              
                              
def annotation_to_poisson(datasets: List[vtk.vtkStructuredGrid],
                          channels: List[int],
                          views: List[str],
                          format_view: str,
                          times: List[int],
                          save_path: str = None)->List[vtk.vtkPolyData]:
    ncols = 256
    ncolors = 256

    poisson_polydatas = []
    poisson_volumes = []
    for t in times:
        points = []
        normals = []
        for dataset, channel, view in zip(datasets, channels, views):
            arr_annot = vtk.util.numpy_support.vtk_to_numpy(dataset.GetCellData().GetArray(format_view.format(view=view, t=t)))
            arr_annot_copy = np.copy(arr_annot)
            # Extract contours of the segmentation
            im = (arr_annot==channel).reshape(-1, ncols).astype(np.uint8)
            contours, hierarchy  = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours]
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

            if save_path:
                normals_arr = vtk.util.numpy_support.numpy_to_vtk(dataset_normals)
                normals_arr.SetName('Normals')
                centers.GetOutput().GetPointData().AddArray(normals_arr)
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetInputConnection(centers.GetOutputPort())
                writer.SetFileName(f'{save_path}_{view}_{t}.vtp')
                writer.Update()

            points.append(dataset_points)
            normals.append(dataset_normals)

            arr_annot[:] = arr_annot_copy

        points_arr = np.vstack(points)
        normals_arr = np.vstack(normals)

        poisson_polydatas.append(points_normals_to_poisson(points_arr, normals_arr))
        poisson_volumes.append(polydata_to_volume(poisson_polydatas[-1]))
    return poisson_polydatas, poisson_volumes




