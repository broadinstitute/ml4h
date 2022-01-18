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
# %%
import seaborn as sns

sns.histplot(aorta_arr[:, :, 370].ravel())
#%%
import matplotlib.pyplot as plt
import numpy as np

z = 360
f, ax = plt.subplots()
aorta_ma = np.ma.masked_array(aorta_arr, mask=aorta_arr>50.0)
ax.imshow(images_arr[:, :, 377], cmap='gray')
ax.imshow(aorta_ma[:, :, 377])
# %%

ingest_mri_dicoms_zipped(
    sample_id=patient,
    instance=2,
    file=f'/home/pdiachil/projects/aorta/{patient}_20201_2_0.zip',
    destination=f'bodymri_allraw_{patient}_2_0',
    in_memory=True,
    save_dicoms=False,
    output_name=f'bodymri_{patient}',
)

# %%
os.makedirs(f'bodymri_allraw_{patient}_2_0/projected', exist_ok=True)
build_projection_hd5(
    f'bodymri_allraw_{patient}_2_0/bodymri_{patient}.h5',
    f'bodymri_allraw_{patient}_2_0',
    f'bodymri_allraw_{patient}_2_0/projected'
)

# %%
# !rm bodymri_allraw_1000107_2_0/projected/bodymri_1000107.h5


# %%
import h5py
hd5 = h5py.File(f'bodymri_allraw_{patient}_2_0/projected/bodymri_{patient}.h5')
# %%
arr = read_compressed(hd5['instance/2/w_sagittal'])
# %%
import matplotlib.pyplot as plt
plt.imshow(arr)
# %%
old_hd5 = h5py.File(f'bodymri_allraw_{patient}_2_0/bodymri_{patient}.h5', 'r')
# %%
instance = 2
data = {
                int(name): read_compressed(old_hd5[f'instance/{instance}/series/{name}'])
                for name in old_hd5[f'instance/{instance}/series']
            }
# %%
import matplotlib.pyplot as plt
plt.imshow(data[4][:, 119, :])
# %%

# %%
import h5py
import pandas as pd
from scipy.ndimage import zoom
import numpy as np

def center_pad_3d(x: np.ndarray, width: int) -> np.ndarray:
    """Pad an image on the left and right with 0s to a specified
    target width.
    Args:
        x (np.ndarray): Input data.
        width (int): Desired width.
    Returns:
        np.ndarray: Padded data.
    """
    new_x = np.zeros((width, x.shape[1], x.shape[2]))
    offset = (width - x.shape[0]) // 2
    new_x[offset : offset+x.shape[0], ...] = x
    return new_x


def center_pad_stack_3d(xs: np.ndarray) -> np.ndarray:
    """Center and pad input data and then stack the images
    with different widths.
    Args:
        xs (np.ndarray): Input data.
    Returns:
        np.ndarray: Center-padded and stacked data.
    """
    max_width = max(x.shape[0] for x in xs)
    return np.concatenate([center_pad_3d(x, max_width) for x in xs], axis=2)

hd5 = h5py.File(f'/home/pdiachil/ml/notebooks/mri/bodymri_allraw_{patient}_2_0/bodymri_{patient}.h5', 'r')
meta_data = pd.read_parquet(f'/home/pdiachil/ml/notebooks/mri/bodymri_allraw_{patient}_2_0/bodymri_{patient}_2.pq')

station_z_scales = 3.0, 4.5, 4.5, 4.5, 3.5, 4.0
# station_z_scales = [scale / 3 for scale in station_z_scales]
station_xscale = meta_data['col_pixel_spacing_mm'].mean()
station_yscale = meta_data['row_pixel_spacing_mm'].mean()

data = {
        int(name): read_compressed(hd5[f'instance/2/series/{name}'])
        for name in hd5[f'instance/2/series']
       }

z_pos = meta_data.groupby("series_number")["image_position_z"].agg(["min", "max"])
slices = build_z_slices(
           [data[i].shape[-1] for i in range(1, 25, 4)],
           [z_pos.loc[i] for i in range(1, 25, 4)],
        )

horizontal_lines = [
        (idx.stop - idx.start) * scale for idx, scale in zip(slices, station_z_scales)
    ]
horizontal_lines = np.cumsum(horizontal_lines).astype(np.uint16)[:-1]
body = {"horizontal_line_idx": horizontal_lines}

for type_idx, series_type_name in zip(range(4), ("in", "opp", "f", "w")):
    print(type_idx)
    full_slice_to_stack = []
    for station_idx in range(1, 25, 4):  # neck, upper ab, lower ab, legs
        print(station_idx)
        series_num = station_idx + type_idx
        station_slice = slices[station_idx // 4]
        scale = station_z_scales[station_idx // 4]
        full_slice = data[series_num][..., station_slice]
        full_slice_scaled = zoom(full_slice, [station_xscale, station_yscale, scale])
        full_slice_to_stack.append(full_slice_scaled)
    body[f"{series_type_name}"] = normalize(center_pad_stack_3d(full_slice_to_stack))
# %%
import vtk
from vtk.util import numpy_support as ns
for b, bb in body.items():
    if 'line' in b:
        continue
    img = vtk.vtkImageData()
    img.SetOrigin(0.0, 0.0, 0.0)
    img.SetExtent(0, bb.shape[1]-1, 0, bb.shape[0]-1, 0, bb.shape[2]-1)
    img.SetSpacing(1.0, 1.0, 1.0)
    

    bbt = bb.swapaxes(0, 1)
    bbt = np.flip(bbt, axis=2)
    # bbt = bbt.reshape(*reversed(bb.shape))
    # nslice = bbt.shape[0]*bb.shape[1]
    # for z_slice in range(bbt.shape[2]):
    #     arr[nslice*z_slice:nslice*(z_slice+1)] = bbt[:, :, z_slice].ravel('F')
    arr_vtk = ns.numpy_to_vtk(bbt.ravel('F'), deep=True, array_type=vtk.VTK_INT)
    arr_vtk.SetName('ImageScalars')
    img.GetPointData().SetScalars(arr_vtk)

    resize = vtk.vtkImageResize()
    resize.SetInputData(img)
    resize.SetOutputDimensions(bbt.shape[0]//2, bbt.shape[1]//2, bbt.shape[2]//2)
    resize.Update()
    img_writer = vtk.vtkXMLImageDataWriter()
    img_writer.SetInputConnection(resize.GetOutputPort())
    img_writer.SetFileName(f'bodymri_allraw_{patient}_2_0/{b}.vti')
    img_writer.Update()
# %%

# %%
