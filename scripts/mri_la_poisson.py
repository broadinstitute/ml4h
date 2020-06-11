import h5py
import glob
import seaborn as sns
from ml4cvd.tensor_from_file import _mri_tensor_4d, _mri_hd5_to_structured_grids
from ml4cvd.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES
from ml4cvd.tensor_from_file import _mri_project_grids
from vtk.util import numpy_support as ns
import vtk
import numpy as np
import pandas as pd
from pypoisson import poisson_reconstruction
import os
import imageio
from scipy.spatial import ConvexHull
import sys

MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP = {'left_atrium': 11}

cli_idxs = [int(idx) for idx in sys.argv[1:]]

def project_3dpts_plane(pts):
    N = np.cross(pts[10] - pts[0], pts[200] - pts[0])
    U = (pts[10] - pts[0])/np.linalg.norm(pts[10] - pts[0])
    uN = N / np.linalg.norm(N)
    u = pts[0] + U  
    V = np.cross(U, uN)
    v = pts[0] + V
    n = pts[0] + uN
    S = np.ones((4, 4))
    S[:-1, 0] = pts[0]
    S[:-1, 1] = u
    S[:-1, 2] = v
    S[:-1, 3] = n
    Sinv = np.linalg.inv(S)
    D = np.zeros((4, 4))
    D[-1] = 1
    D[0, 1] = 1
    D[1, 2] = 1
    D[2, 3] = 1
    M = np.dot(D, Sinv)
#     f, ax = plt.subplots()
#     ax = f.add_subplot(111, projection='3d')
#     ax.scatter(*pts[0])
#     ax.scatter(*v)
#     ax.scatter(*n)
#     ax.scatter(*u)
    pts_tmp = np.zeros((4, len(pts)))
    pts_tmp[:-1, :] = pts.T
    pts_tmp[-1] = 1
    return np.dot(M, pts_tmp)[:-1].T

petersen = pd.read_csv('/home/pdiachil/ml/notebooks/mri/returned_lv_mass.tsv', sep='\t')
petersen_idxs = petersen['sample_id'].values
for t in range(MRI_FRAMES):
    petersen[f'LA_poisson_{t}'] = 0.0



for i in cli_idxs:
    idx = petersen_idxs[i]
    try:
        with h5py.File(f'/mnt/disks/sax-and-lax-zip-2019-09-30/unzip-sax-and-lax-44k-2020-06-05/{idx}.hd5', 'r') as ff:
            ds_2ch = _mri_hd5_to_structured_grids(ff, 'cine_segmented_lax_2ch_annotated_', view_name='cine_segmented_lax_2ch', concatenate=True, save_path=None, order='F')
            ds_3ch = _mri_hd5_to_structured_grids(ff, 'cine_segmented_lax_3ch_annotated_', view_name='cine_segmented_lax_3ch', concatenate=True, save_path=None, order='F')
            ds_4ch = _mri_hd5_to_structured_grids(ff, 'cine_segmented_lax_4ch_annotated_', view_name='cine_segmented_lax_4ch', concatenate=True, save_path=None, order='F')
        
            centers_2ch = vtk.vtkCellCenters()
            centers_2ch.SetInputData(ds_2ch[0])
            centers_2ch.Update()
            centers_3ch = vtk.vtkCellCenters()
            centers_3ch.SetInputData(ds_3ch[0])
            centers_3ch.Update()
            centers_4ch = vtk.vtkCellCenters()
            centers_4ch.SetInputData(ds_4ch[0])
            centers_4ch.Update()
            for t in range(MRI_FRAMES):
                arr_annot_2ch = ns.vtk_to_numpy(ds_2ch[0].GetCellData().GetArray(f'cine_segmented_lax_2ch_annotated__{t}'))
                arr_annot_3ch = ns.vtk_to_numpy(ds_3ch[0].GetCellData().GetArray(f'cine_segmented_lax_3ch_annotated__{t}'))
                arr_annot_4ch = ns.vtk_to_numpy(ds_4ch[0].GetCellData().GetArray(f'cine_segmented_lax_4ch_annotated__{t}'))
                idx_2ch = np.where(arr_annot_2ch == MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'])
                idx_3ch = np.where(arr_annot_3ch == MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'])
                idx_4ch = np.where(arr_annot_4ch == MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity'])
                pts_2ch = ns.vtk_to_numpy(centers_2ch.GetOutput().GetPoints().GetData())[idx_2ch]
                pts_2ch_2d = project_3dpts_plane(pts_2ch)
                n_2ch = np.cross(pts_2ch[10] - pts_2ch[0], pts_2ch[200] - pts_2ch[0])
                n_2ch /= np.linalg.norm(n_2ch)
                pts_3ch = ns.vtk_to_numpy(centers_3ch.GetOutput().GetPoints().GetData())[idx_3ch]
                pts_3ch_2d = project_3dpts_plane(pts_3ch)
                n_3ch = np.cross(pts_3ch[10] - pts_3ch[0], pts_3ch[200] - pts_3ch[0])
                n_3ch /= np.linalg.norm(n_3ch)
                pts_4ch = ns.vtk_to_numpy(centers_4ch.GetOutput().GetPoints().GetData())[idx_4ch]
                pts_4ch_2d = project_3dpts_plane(pts_4ch)
                n_4ch = np.cross(pts_4ch[10] - pts_4ch[0], pts_4ch[200] - pts_4ch[0])
                n_4ch /= np.linalg.norm(n_4ch)
                hull_2ch = ConvexHull(pts_2ch_2d[:, :-1])
                hull_3ch = ConvexHull(pts_3ch_2d[:, :-1])
                hull_4ch = ConvexHull(pts_4ch_2d[:, :-1])
                pts_hull_2ch = pts_2ch[hull_2ch.vertices]
                pts_hull_3ch = pts_3ch[hull_3ch.vertices]
                pts_hull_4ch = pts_4ch[hull_4ch.vertices]
                pts1_hull_2ch = np.zeros_like(pts_hull_2ch)
                pts1_hull_3ch = np.zeros_like(pts_hull_3ch)
                pts1_hull_4ch = np.zeros_like(pts_hull_4ch)
                pts1_hull_2ch[:-1] = pts_hull_2ch[1:]
                pts1_hull_2ch[-1] = pts_hull_2ch[0]
                pts1_hull_3ch[:-1] = pts_hull_3ch[1:]
                pts1_hull_3ch[-1] = pts_hull_3ch[0]
                pts1_hull_4ch[:-1] = pts_hull_4ch[1:]
                pts1_hull_4ch[-1] = pts_hull_4ch[0]
                n_hull_2ch = np.cross(pts1_hull_2ch-pts_hull_2ch, n_2ch)
                n_hull_2ch /= np.linalg.norm(n_hull_2ch, axis=1).reshape(-1, 1)
                n_hull_3ch = np.cross(pts1_hull_3ch-pts_hull_3ch, n_3ch)
                n_hull_3ch /= np.linalg.norm(n_hull_3ch, axis=1).reshape(-1, 1)
                n_hull_4ch = np.cross(pts1_hull_4ch-pts_hull_4ch, n_4ch)
                n_hull_4ch /= np.linalg.norm(n_hull_4ch, axis=1).reshape(-1, 1)
                n1_hull_2ch = np.zeros_like(n_hull_2ch)
                n1_hull_2ch[1:] = n_hull_2ch[:-1]
                n1_hull_2ch[0] = n_hull_2ch[-1]
                n_hull_2ch = -0.5*(n_hull_2ch + n1_hull_2ch)
                n1_hull_3ch = np.zeros_like(n_hull_3ch)
                n1_hull_3ch[1:] = n_hull_3ch[:-1]
                n1_hull_3ch[0] = n_hull_3ch[-1]
                n_hull_3ch = -0.5*(n_hull_3ch + n1_hull_3ch)
                n1_hull_4ch = np.zeros_like(n_hull_4ch)
                n1_hull_4ch[1:] = n_hull_4ch[:-1]
                n1_hull_4ch[0] = n_hull_4ch[-1]
                n_hull_4ch = -0.5*(n_hull_4ch + n1_hull_4ch)       
                pts = np.vstack([pts_hull_2ch, pts_hull_3ch, pts_hull_4ch])
                normals = np.vstack([n_hull_2ch, n_hull_3ch, n_hull_4ch])
                faces, vertices = poisson_reconstruction(pts, normals, depth=16)
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
            
                mass = vtk.vtkMassProperties()
                mass.SetInputConnection(clean.GetOutputPort())
                mass.Update()
                petersen.loc[i, f'LA_poisson_{t}'] = mass.GetVolume()
                break
    except:
        pass

petersen.iloc[cli_idxs].to_csv(f'petersen_{cli_idxs[0]}.csv', sep='\t', index=False)