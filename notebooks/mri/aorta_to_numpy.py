# %%
import os
import vtk
from vtk.util import numpy_support as ns
import glob

#%%
patient = 1000107

vti = glob.glob(f'/home/pdiachil/projects/aorta/simvascular/{patient}/Images/*.vti')[0]
vtp = glob.glob(f'/home/pdiachil/projects/aorta/simvascular/{patient}/Models/*.vtp')[0]

images = vtk.vtkXMLImageDataReader()
images.SetFileName(vti)
images.Update()
size = [
    images.GetOutput().GetExtent()[1]+1,
    images.GetOutput().GetExtent()[3]+1,
    images.GetOutput().GetExtent()[5]+1
]
images_arr = ns.vtk_to_numpy(images.GetOutput().GetPointData().GetArray('Scalars_')).reshape(size, order='F')

aorta = vtk.vtkImageData()
aorta.DeepCopy(images.GetOutput())
aorta_tmp = ns.vtk_to_numpy(aorta.GetPointData().GetArray('Scalars_'))
aorta_tmp[:] = 0.0

aorta_reader = vtk.vtkXMLPolyDataReader()
aorta_reader.SetFileName(vtp)
aorta_reader.Update()

pol2stenc = vtk.vtkPolyDataToImageStencil()
pol2stenc.SetInputConnection(aorta_reader.GetOutputPort())
pol2stenc.SetOutputOrigin(aorta.GetOrigin())
pol2stenc.SetOutputSpacing(aorta.GetSpacing())
pol2stenc.SetOutputWholeExtent(aorta.GetExtent())

imgstenc = vtk.vtkImageStencil()
imgstenc.SetInputData(aorta)
imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
imgstenc.ReverseStencilOff()
imgstenc.SetBackgroundValue(100.0)
imgstenc.Update()

aorta_arr = ns.vtk_to_numpy(imgstenc.GetOutput().GetPointData().GetArray('Scalars_')).reshape(size, order='F')
#aorta_arr = np.array(aorta_arr > 0.0, dtype=np.uint8)

#%%
import matplotlib.pyplot as plt
import numpy as np

z = 360
f, ax = plt.subplots()
aorta_ma = np.ma.masked_array(aorta_arr, mask=aorta_arr>50.0)
ax.imshow(images_arr[:, :, 377], cmap='gray')
ax.imshow(aorta_ma[:, :, 377])

np.save(f'{patient}_images', images_arr)
np.save(f'{patient}_aorta', aorta_arr)