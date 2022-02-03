#%%
patient = 1000107
import numpy as np
import glob
import vtk
from vtk.util import numpy_support

aorta_model = np.load(f'/home/pdiachil/projects/aorta/{patient}_aorta_predicted.npy', encoding='bytes')
aorta_model = np.moveaxis(aorta_model, [0, 1, 2], [2, 0, 1])
aorta_paolo = np.load(f'/home/pdiachil/projects/aorta/{patient}_aorta.npy', encoding='bytes')
vti = glob.glob(f'/home/pdiachil/projects/aorta/simvascular/{patient}/Images/*.vti')[0]
aorta_vti_reader = vtk.vtkXMLImageDataReader()
aorta_vti_reader.SetFileName(vti)
aorta_vti_reader.Update()

aorta_vti = numpy_support.vtk_to_numpy(aorta_vti_reader.GetOutput().GetPointData().GetArray('Scalars_')).reshape(aorta_paolo.shape, order='F')
aorta_vti_model = np.zeros_like(aorta_vti, dtype=np.float)
aorta_vti_model = np.zeros_like(aorta_vti, dtype=np.float)

x_center = (256 - 125) // 2
y_center = (256 - 100) // 2

aorta_vti_model[75:175, 35:160, 250:] = aorta_model[y_center:y_center+100, x_center:x_center+125, :]
aorta_vti_model_arr = numpy_support.numpy_to_vtk(aorta_vti_model.ravel(order='F'))
aorta_vti_model_arr.SetName('Scalars_')
aorta_vti_paolo_arr = numpy_support.numpy_to_vtk(aorta_paolo.ravel(order='F'))
aorta_vti_paolo_arr.SetName('segmentation')
aorta_vti_reader.GetOutput().GetPointData().AddArray(aorta_vti_model_arr)
aorta_vti_reader.GetOutput().GetPointData().AddArray(aorta_vti_paolo_arr)

aorta_vti_writer = vtk.vtkXMLImageDataWriter()
aorta_vti_writer.SetFileName(f'{patient}_model.vti')
aorta_vti_writer.SetInputConnection(aorta_vti_reader.GetOutputPort())
aorta_vti_writer.Update()
# %%
import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.imshow(aorta_paolo[:, :, 400], alpha=0.25)
ax.imshow(aorta_vti_model[:, :, 400], alpha=0.25)

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
