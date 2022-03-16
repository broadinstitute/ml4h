# %%
import os
import pandas as pd
import sys
from google.cloud import storage
from ml4h.applications.ingest.ingest_mri import ingest_mri_dicoms_zipped, read_compressed
from ml4h.applications.ingest.two_d_projection import build_projection_hd5, build_z_slices, normalize
import h5py
import pandas as pd
from scipy.ndimage import zoom
import numpy as np
import vtk
from vtk.util import numpy_support as ns

# # %%
# bodymris = pd.read_csv('/home/pdiachil/projects/aorta/bodymris.csv')
# bodymris_done = pd.read_csv('/home/pdiachil/projects/aorta/bodymris_done_2.csv')
# bodymris['patient'] = bodymris['filepath'].str.split('/').str[-1].str.split('_').str[0].apply(int)
# bodymris['instance'] = bodymris['filepath'].str.split('/').str[-1].str.split('_').str[2].apply(int)
# bodymris['patient_instance'] = bodymris['patient'].apply(str)+'_'+bodymris['instance'].apply(str)
# bodymris_done['patient'] = bodymris_done['filepath'].str.split('/').str[-1].str.split('_').str[0].apply(int)
# bodymris_done['instance'] = bodymris_done['filepath'].str.split('/').str[-1].str.split('_').str[2].apply(int)
# bodymris_done['patient_instance'] = bodymris_done['patient'].apply(str)+'_'+bodymris_done['instance'].apply(str)

# todo_still = list(set(bodymris['patient_instance'])-set(bodymris_done['patient_instance']))
# bodymris = bodymris[bodymris['patient_instance'].isin(todo_still)]
# bodymris.to_csv('/home/pdiachil/ml/notebooks/mri/bodymris.csv')

# %%
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


#%%
#start = int(sys.argv[1])
#end = int(sys.argv[2])

os.chdir('/home/pdiachil/ml/notebooks/mri')

start = 0
end = 1

bodymris = pd.read_csv('/home/pdiachil/ml/notebooks/mri/bodymris.csv')

# bodymris_done = pd.read_csv('/home/pdiachil/projects/aorta/bodymris_done.csv')
rows = bodymris.iloc[start:end]
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('bulkml4cvd')
# bodymris['patient'] = bodymris['filepath'].str.split('/').str[-1].str.split('_').str[0].apply(int)
# bodymris['instance'] = bodymris['filepath'].str.split('/').str[-1].str.split('_').str[2].apply(int)
# bodymris['patient_instance'] = bodymris['patient'].apply(str)+'_'+bodymris['instance'].apply(str)
# bodymris_done['patient'] = bodymris_done['filepath'].str.split('/').str[-1].str.split('_').str[0].apply(int)
# bodymris_done['instance'] = bodymris_done['filepath'].str.split('/').str[-1].str.split('_').str[2].apply(int)
# bodymris_done['patient_instance'] = bodymris_done['patient'].apply(str)+'_'+bodymris['instance'].apply(str)

# todo_still = list(set(bodymris['patient_instance'])-set(bodymris_done['patient_instance']))
# bodymris = bodymris[bodymris['patient_instance'].isin(todo_still)]
# bodymris.to_csv('/home/pdiachil/ml/notebooks/mri/bodymris.csv')
# %%

for i, row in rows.iterrows():
    prefix = f'bodymri/all/raw/{row["patient"]}_20201_{row["instance"]}_0.zip'
    blobs = storage_client.list_blobs(bucket, prefix=prefix)
    for blob in blobs:
        blob.download_to_filename(f'/home/pdiachil/projects/aorta/{row["patient"]}_20201_{row["instance"]}_0.zip')

    patient = row['patient']
    ingest_mri_dicoms_zipped(
        sample_id=patient,
        instance=2,
        file=f'/home/pdiachil/projects/aorta/{patient}_20201_{row["instance"]}_0.zip',
        destination=f'bodymri_allraw_{patient}_{row["instance"]}_0',
        in_memory=True,
        save_dicoms=False,
        output_name=f'bodymri_{patient}_{row["instance"]}_0',
    )

    patient = row['patient']
    os.makedirs(f'/home/pdiachil/ml/notebooks/mri/bodymri_allraw_{patient}_{row["instance"]}_0/projected', exist_ok=True)
    build_projection_hd5(
        f'bodymri_allraw_{patient}_{row["instance"]}_0/bodymri_{patient}_{row["instance"]}_0.h5',
        f'bodymri_allraw_{patient}_{row["instance"]}_0',
        f'bodymri_allraw_{patient}_{row["instance"]}_0/projected'
    )

    hd5 = h5py.File(f'/home/pdiachil/ml/notebooks/mri/bodymri_allraw_{patient}_{row["instance"]}_0/bodymri_{patient}_{row["instance"]}_0.h5', 'r')
    meta_data = pd.read_parquet(f'/home/pdiachil/ml/notebooks/mri/bodymri_allraw_{patient}_{row["instance"]}_0/bodymri_{patient}_{row["instance"]}_0_2.pq')

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
        if type_idx < 3:
            continue
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

    for b, bb in body.items():
        if 'line' in b:
            continue
        if b != 'w':
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
        img_writer.SetFileName(f'bodymri_allraw_{patient}_{row["instance"]}_0/{b}.vti')
        img_writer.Update()

        size = [
            resize.GetOutput().GetExtent()[1]+1,
            resize.GetOutput().GetExtent()[3]+1,
            resize.GetOutput().GetExtent()[5]+1
        ]
        images_arr = ns.vtk_to_numpy(resize.GetOutput().GetPointData().GetArray('ImageScalars')).reshape(size, order='F')

        np.save(f'{patient}_20201_{row["instance"]}_0_images', images_arr)

# %%
