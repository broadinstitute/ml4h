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

def project_3dpts_plane(pts):
    pt0, pt1, pt2 = [pts[idx] for idx in [0, len(pts)//2, -1]]
    N = np.cross(pt0 - pt1, pt2 - pt0)
    U = (pt1 - pt0)/np.linalg.norm(pt1 - pt0)
    uN = N / np.linalg.norm(N)
    u = pt0 + U  
    V = np.cross(U, uN)
    v = pt0 + V
    n = pt0 + uN
    S = np.ones((4, 4))
    S[:-1, 0] = pt0
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

def to_xdmf(vtk_object, filename, array_name=None, append=False, append_time=0, write_footer=True):
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
        if array_name is None:
            array_name = vtk_object.GetCellData().GetArray(0).GetName().split('__')[0]
        ff_hd5.create_dataset('points', data=arr_pts, compression="gzip", compression_opts=9)
        for t in range(MRI_FRAMES):
            ff_xml.write(f"""
      <Grid Name="Grid">
        <Time Value="{t}"/>)
        <Geometry Origin="" Type="XYZ">
          <DataItem DataType="Float" Dimensions="{vtk_object.GetNumberOfPoints()} 3" Format="HDF" Precision="8">{os.path.basename(filename)}.hd5:points</DataItem>
        </Geometry>
        <Topology Dimensions="{extent[5]+1} {extent[1]+1} {extent[3]+1}" Type="3DSMesh"/>
        <Attribute Center="Cell" ElementCell="" ElementDegree="0" ElementFamily="" ItemType="" Name="{array_name}" Type="None">
          <DataItem DataType="Float" Dimensions="{extent[5]} {extent[1]} {extent[3]}" Format="HDF" Precision="8">{os.path.basename(filename)}.hd5:{array_name}_{t}</DataItem>
        </Attribute>
      </Grid>""")
            arr = vtk.util.numpy_support.vtk_to_numpy(vtk_object.GetCellData().GetArray(f'{array_name}_{t}'))
            arr = arr.reshape(extent[5], extent[1], extent[3])
            ff_hd5.create_dataset(f'{array_name}_{t}', data=arr, compression="gzip", compression_opts=9)        
    elif isinstance(vtk_object, vtk.vtkPolyData):
        arr_pts = vtk.util.numpy_support.vtk_to_numpy(vtk_object.GetPoints().GetData())
        arr_cells = np.copy(vtk.util.numpy_support.vtk_to_numpy(vtk_object.GetPolys().GetData()))
        arr_cells[::4] = XDMF_TRIANGLE
        ff_xml.write(f"""
      <Grid Name="Grid">
        <Time Value="{append_time}"/>)
        <Geometry Origin="" Type="XYZ">
          <DataItem DataType="Float" Dimensions="{vtk_object.GetNumberOfPoints()} 3" Format="HDF" Precision="8">{os.path.basename(filename)}.hd5:points_{append_time}</DataItem>
        </Geometry>
        <Topology Dimensions="{vtk_object.GetNumberOfCells()}" Type="Mixed">
          <DataItem DataType="Int" Dimensions="{len(arr_cells)}" Format="HDF" Precision="8">{os.path.basename(filename)}.hd5:cells_{append_time}</DataItem>
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