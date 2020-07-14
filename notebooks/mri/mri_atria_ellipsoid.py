# %%
import h5py
import sys
import glob
from ml4cvd.tensor_from_file import _mri_tensor_4d, _mri_hd5_to_structured_grids
from ml4cvd.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES
from ml4cvd.tensor_from_file import _mri_project_grids
from vtk.util import numpy_support as ns
import vtk
import numpy as np
import pandas as pd
# from pypoisson import poisson_reconstruction
import os
import imageio

XDMF_TRIANGLE=4
MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP = {}
MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'] = 11
# petersen = pd.read_csv('/home/pdiachil/ml/atria/returned_lv_mass.tsv', sep='\t')
petersen_idxs = petersen['sample_id'].values

# %%
for t in range(MRI_FRAMES):
    petersen[f'LA_ellipsoid_vol_{t}'] = 0.0
    for i in range(9):
      petersen[f'LA_ellipsoid_evecs_{t}_{i}'] = 0.0
    for i in range(3):
      petersen[f'LA_ellipsoid_radii_{t}_{i}'] = 0.0
    for i in range(3):
      petersen[f'LA_ellipsoid_center_{t}_{i}'] = 0.0


def ellipsoid_fit(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v


def project_3dpts_plane(pts):
    
    N = np.cross(pts[len(pts)//2] - pts[0], pts[-1] - pts[0])
    U = (pts[len(pts)//2] - pts[0])/np.linalg.norm(pts[len(pts)//2] - pts[0])
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
    pts_tmp = np.zeros((4, len(pts)))
    pts_tmp[:-1, :] = pts.T
    pts_tmp[-1] = 1
    return np.dot(M, pts_tmp)[:-1].T

def to_xdmf(vtk_object, filename, append=False, append_time=0, write_footer=True):
    write_mode = 'a' if append else 'w'
    ff_xml = open(f'{filename}.xmf', write_mode)
    ff_hd5 = h5py.File(f'{filename}.hd5', write_mode)
    arr_pts = vtk.util.numpy_support.vtk_to_numpy(vtk_object.GetPoints().GetData())
    if not append:
        ff_xml.write("""<?xml version="1.0" encoding="utf-8"?>
<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="3.0">
  <Domain>
    <Grid CollectionType="Temporal" GridType="Collection" Name="Collection">""")
    if isinstance(vtk_object, vtk.vtkStructuredGrid):
        extent = vtk_object.GetExtent()
        arr_name = vtk_object.GetCellData().GetArray(0).GetName().split('__')[0]
        ff_hd5.create_dataset('points', data=arr_pts, compression="gzip", compression_opts=9)
        for t in range(MRI_FRAMES):
            ff_xml.write(f"""
      <Grid Name="Grid">
        <Time Value="{t}"/>)
        <Geometry Origin="" Type="XYZ">
          <DataItem DataType="Float" Dimensions="{vtk_object.GetNumberOfPoints()} 3" Format="HDF" Precision="8">{filename}.hd5:points</DataItem>
        </Geometry>
        <Topology Dimensions="{extent[5]+1} {extent[1]+1} {extent[3]+1}" Type="3DSMesh"/>
        <Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="{arr_name}" Type="None">
          <DataItem DataType="Float" Dimensions="{extent[5]} {extent[1]} {extent[3]}" Format="HDF" Precision="8">{filename}.hd5:{arr_name}_{t}</DataItem>
        </Attribute>
      </Grid>""")
            arr = vtk.util.numpy_support.vtk_to_numpy(vtk_object.GetCellData().GetArray(f'{arr_name}__{t}'))
            arr = arr.reshape(extent[5], extent[1], extent[3])
            ff_hd5.create_dataset(f'{arr_name}_{t}', data=arr, compression="gzip", compression_opts=9)        
    elif isinstance(vtk_object, vtk.vtkPolyData):
        arr_pts = vtk.util.numpy_support.vtk_to_numpy(vtk_object.GetPoints().GetData())
        arr_cells = np.copy(vtk.util.numpy_support.vtk_to_numpy(vtk_object.GetPolys().GetData()))
        arr_cells[::4] = XDMF_TRIANGLE
        ff_xml.write(f"""
      <Grid Name="Grid">
        <Time Value="{append_time}"/>)
        <Geometry Origin="" Type="XYZ">
          <DataItem DataType="Float" Dimensions="{vtk_object.GetNumberOfPoints()} 3" Format="HDF" Precision="8">{filename}.hd5:points_{append_time}</DataItem>
        </Geometry>
        <Topology Dimensions="{vtk_object.GetNumberOfCells()}" Type="Mixed">
          <DataItem DataType="Int" Dimensions="{len(arr_cells)}" Format="HDF" Precision="8">{filename}.hd5:cells_{append_time}</DataItem>
        </Topology>
      </Grid>""")
        ff_hd5.create_dataset(f'points_{append_time}', data=arr_pts, compression="gzip", compression_opts=9)
        ff_hd5.create_dataset(f'cells_{append_time}', data=arr_cells, compression="gzip", compression_opts=9)  
  
    if write_footer:
        ff_xml.write("""
    </Grid>
  </Domain>
</Xdmf>""")
    ff_xml.close()
    ff_hd5.close()         

if __name__ == '__main__':

  from scipy.spatial import ConvexHull
  volumes = []
  petersen_processed = []
  for i, idx in enumerate(petersen_idxs):
      if i < int(sys.argv[1]): 
          continue
      if i > int(sys.argv[1]): 
          break
      
      with h5py.File(f'/mnt/disks/segmented-sax-lax/2020-07-07/{idx}.hd5', 'r') as ff:
          dss = []
          for view in ['2ch', '3ch', '4ch']:
              dss.append(_mri_hd5_to_structured_grids(ff, f'cine_segmented_lax_{view}_annotated_',
                                                      view_name=f'cine_segmented_lax_{view}', concatenate=True,
                                                      save_path=None, order='F'))
              # to_xdmf(dss[-1][0], f'{idx}_{view}')            

          for t in range(MRI_FRAMES):
              pts = []
              normals = []
              for ds, view, la_value in zip(dss, ['2ch', '3ch', '4ch'],
                                              [MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
                                              MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP['left_atrium'],
                                              MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['LA_cavity']]):
                  centers = vtk.vtkCellCenters()
                  centers.SetInputData(ds[0])
                  centers.Update()
                  arr_annot = ns.vtk_to_numpy(ds[0].GetCellData().GetArray(f'cine_segmented_lax_{view}_annotated__{t}'))
                  idx_view = np.where(arr_annot == la_value)
                  pts_view = ns.vtk_to_numpy(centers.GetOutput().GetPoints().GetData())[idx_view]
                  pts_view_2d = project_3dpts_plane(pts_view)
                  n_view = np.cross(pts_view[10] - pts_view[0], pts_view[-1] - pts_view[0])
                  n_view /= np.linalg.norm(n_view)
                  hull_view = ConvexHull(pts_view_2d[:, :-1])
                  pts_hull_view = pts_view[hull_view.vertices]
                  pts1_hull_view = np.zeros_like(pts_hull_view)
                  pts1_hull_view[:-1] = pts_hull_view[1:]
                  pts1_hull_view[-1] = pts_hull_view[0]
                  n_hull_view = np.cross(pts1_hull_view-pts_hull_view, n_view)
                  n_hull_view /= np.linalg.norm(n_hull_view, axis=1).reshape(-1, 1)
                  n1_hull_view = np.zeros_like(n_hull_view)
                  n1_hull_view[1:] = n_hull_view[:-1]
                  n1_hull_view[0] = n_hull_view[-1]
                  n_hull_view = -0.5*(n_hull_view + n1_hull_view)    
                  pts.append(pts_hull_view)
                  normals.append(n_hull_view)
              pts = np.vstack(pts)
              normals = np.vstack(normals)
              center, evecs, radii, v = ellipsoid_fit(pts)

              ellipsoid = vtk.vtkParametricEllipsoid()
              ellipsoid.SetXRadius(1.0)
              ellipsoid.SetYRadius(1.0)
              ellipsoid.SetZRadius(1.0)
              ellipsoid_surf = vtk.vtkParametricFunctionSource()
              ellipsoid_surf.SetParametricFunction(ellipsoid)
              ellipsoid_surf.Update()

              transformation_matrix = np.zeros((4, 4))
              for col in range(3):
                  transformation_matrix[:3, col] = evecs[col]*radii[col]
              transformation_matrix[:3, 3] = center
              transformation_matrix[3, 3] = 1.0

              transform = vtk.vtkTransform()
              transform.SetMatrix(transformation_matrix.ravel().tolist())
              
              transform_filter = vtk.vtkTransformFilter()
              transform_filter.SetInputConnection(ellipsoid_surf.GetOutputPort())
              transform_filter.SetTransform(transform)
              transform_filter.Update()

              # writer = vtk.vtkXMLPolyDataWriter()
              # writer.SetInputConnection(transform_filter.GetOutputPort())
              # writer.SetFileName('ellipsoid.vtp')
              # writer.Update()

              # faces_tmp = np.zeros((len(faces), 4), dtype=np.int64)
              # faces_tmp[:, 0] = 3
              # faces_tmp[:, 1:] = faces 
              # polydata_points = vtk.vtkPoints()
              # polydata_points.SetData(ns.numpy_to_vtk(vertices))
              # polydata_cells = vtk.vtkCellArray()
              # polydata_cells.SetNumberOfCells(len(faces))
              # polydata_cells.SetCells(len(faces), ns.numpy_to_vtkIdTypeArray(faces_tmp.ravel()))
              # polydata = vtk.vtkPolyData()
              # polydata.SetPoints(polydata_points)
              # polydata.SetPolys(polydata_cells)
              # boundary_edges = vtk.vtkFeatureEdges()
              # boundary_edges.SetInputData(polydata)
              # boundary_edges.BoundaryEdgesOn()
              # boundary_edges.FeatureEdgesOff()
              # boundary_edges.NonManifoldEdgesOff()
              # boundary_edges.ManifoldEdgesOff()    
              # boundary_strips = vtk.vtkStripper()
              # boundary_strips.SetInputConnection(boundary_edges.GetOutputPort())
              # boundary_strips.Update()
              # boundary_poly = vtk.vtkPolyData()
              # boundary_poly.SetPoints(boundary_strips.GetOutput().GetPoints())
              # boundary_poly.SetPolys(boundary_strips.GetOutput().GetLines())
              
              # append = vtk.vtkAppendPolyData()
              # append.UserManagedInputsOn()
              # append.SetNumberOfInputs(2)
              # append.SetInputDataByNumber(0, polydata)
              # append.SetInputDataByNumber(1, boundary_poly)
              # append.Update()
              
              # clean = vtk.vtkCleanPolyData()
              # clean.ConvertLinesToPointsOff()
              # clean.ConvertPolysToLinesOff()
              # clean.ConvertStripsToPolysOff()
              # clean.PointMergingOn()
              # clean.SetInputConnection(append.GetOutputPort())
              # clean.Update()
              # triangle_filter = vtk.vtkTriangleFilter()
              # triangle_filter.SetInputConnection(clean.GetOutputPort())
              # triangle_filter.Update()
              
              append = False if (t == 0) else True
              write_footer = True if (t == MRI_FRAMES - 1) else False
              # to_xdmf(transform_filter.GetOutput(), f'{idx}_atrium_ellipsoid', append=append, 
              #         append_time=t, write_footer=write_footer)

              mass = vtk.vtkMassProperties()
              mass.SetInputConnection(transform_filter.GetOutputPort())
              mass.Update()
              petersen.loc[i, f'LA_ellipsoid_vol_{t}'] = mass.GetVolume()
              petersen.loc[i, [f'LA_ellipsoid_evecs_{t}_{j}' for j in range(9)]] = np.ravel(evecs)
              petersen.loc[i, [f'LA_ellipsoid_radii_{t}_{j}' for j in range(3)]] = np.ravel(radii)
              petersen.loc[i, [f'LA_ellipsoid_center_{t}_{j}' for j in range(3)]] = np.ravel(center)       
      petersen_processed.append(i)
  petersen.loc[petersen_processed].to_csv(f'petersen_processed_{i-1}.csv', sep='\t', index=False)