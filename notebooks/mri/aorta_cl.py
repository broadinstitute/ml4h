# %%
import os
import pandas as pd
import sys
from google.cloud import storage
import h5py
import pandas as pd
from scipy.ndimage import zoom
import numpy as np
import vtk
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import itk
import blosc

start = int(sys.argv[1])
end = int(sys.argv[2])

# start = 0
# end = 10


# %%
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.bucket('ml4cvd')

patients = pd.read_csv('/home/pdiachil/projects/aorta/remaining.csv')
patient_rows = patients.iloc[start:end]
for i, (j, row) in enumerate(patient_rows.iterrows()):
    patient_instance = row['predicted']
    blob_path = f'mdrk/whole_body_mri/aorta/temp/{patient_instance}__prediction.bin'
    blob = bucket.blob(blob_path)
    blob_basename = blob_path.split('/')[-1]
    patient_instance_nofield = patient_instance.split('_')[0] +'_'+patient_instance.split('_')[2]
    model_fname = f'/home/pdiachil/projects/aorta/{blob_basename}'
    blob.download_to_filename(model_fname)

    blob = bucket.blob(f'pdiachil/aorta_for_marcus/aorta_numpy_46k/{patient_instance}_0_images.npy')
    images_fname = f'/home/pdiachil/projects/aorta/{patient_instance}_images.npy'
    blob.download_to_filename(images_fname)

    blob = bucket.blob(f'pdiachil/aorta_for_marcus/aorta_vti_46k/bodymri_allraw_{patient_instance_nofield}_0/w.vti')
    vti_fname = f'/home/pdiachil/projects/aorta/{patient_instance}.vti'
    blob.download_to_filename(vti_fname)

    blob = bucket.blob(f'pdiachil/aorta_for_marcus/aorta_vti_46k/bodymri_allraw_{patient_instance_nofield}_0/bodymri_{patient_instance_nofield}_0_2.pq')
    pq_fname = f'/home/pdiachil/projects/aorta/{patient_instance}.pq'
    blob.download_to_filename(pq_fname)

    with open(model_fname, 'rb') as f: 
        model_predictions = blosc.unpack_array(f.read())
    model_predictions = np.moveaxis(model_predictions, [0, 1, 2], [2, 0, 1])

    
    images = np.load(images_fname)
    aorta_vti_reader = vtk.vtkXMLImageDataReader()
    aorta_vti_reader.SetFileName(vti_fname)
    aorta_vti_reader.Update()

    aorta_vti = numpy_support.vtk_to_numpy(aorta_vti_reader.GetOutput().GetPointData().GetArray('ImageScalars')).reshape(images.shape, order='F')
    aorta_vti_model = np.zeros_like(aorta_vti, dtype=np.float)

    x_center = (128 - 125) // 2
    y_center = (128 - 100) // 2

    aorta_vti_model[75:175, 50:175, 250:] = model_predictions[y_center:y_center+100, x_center:x_center+125, :]
    aorta_vti_model_arr = numpy_support.numpy_to_vtk(aorta_vti_model.ravel(order='F'))
    aorta_vti_model_arr.SetName('ImageScalars2')
    aorta_vti_reader.GetOutput().GetPointData().AddArray(aorta_vti_model_arr)
    aorta_vti_reader.GetOutput().GetPointData().SetActiveScalars('ImageScalars2')

    # gauss = vtk.vtkImageGaussianSmooth()
    # gauss.SetInputConnection(aorta_vti_reader.GetOutputPort())
    # gauss.SetStandardDeviation(1.0)
    # gauss.SetRadiusFactor(5.0)
    # gauss.SetDimensionality(3)
    # gauss.Update()

    aorta_surf = vtk.vtkContourFilter()
    aorta_surf.SetInputConnection(aorta_vti_reader.GetOutputPort())
    aorta_surf.SetValue(0, 0.5)
    aorta_surf.Update()

    aorta_connectivity = vtk.vtkPolyDataConnectivityFilter()
    aorta_connectivity.SetInputConnection(aorta_surf.GetOutputPort())
    aorta_connectivity.SetExtractionModeToAllRegions()
    aorta_connectivity.ColorRegionsOn()
    aorta_connectivity.Update()

    aorta_connectivity.GetOutput().GetPointData().SetActiveScalars('RegionId')

    nregions = aorta_connectivity.GetNumberOfExtractedRegions()
    volumes = []
    surfaces = []
    cogs = []
    extremes_top = []
    extremes_bottom = []
    for ir in range(nregions):
        threshold = vtk.vtkThreshold()
        threshold.SetInputConnection(aorta_connectivity.GetOutputPort())
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'RegionId')
        threshold.SetLowerThreshold(ir-0.5)
        threshold.SetUpperThreshold(ir+0.5)
        threshold.Update()

        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputConnection(threshold.GetOutputPort())
        surface.Update()

        surface_points = numpy_support.vtk_to_numpy(surface.GetOutput().GetPoints().GetData())
        cogs.append(surface_points.mean(axis=0))
        extremes_top.append(np.max(surface_points[:, 2]))
        extremes_bottom.append(np.min(surface_points[:, 2]))

        surfaces.append(surface.GetOutput())

        mass = vtk.vtkMassProperties()
        mass.SetInputConnection(surface.GetOutputPort())
        mass.Update()
        
        volumes.append(mass.GetSurfaceArea())

    sorted_indices = np.argsort(volumes)[::-1]
    indices_to_keep = [sorted_indices[0]]
    max_vol = volumes[sorted_indices[0]]
    volumes_to_keep = [max_vol]
    if nregions > 1:
        for idx in sorted_indices[1:]:
            if volumes[idx] < (max_vol / 50.0):
                break
            if extremes_top[idx] < extremes_bottom[sorted_indices[0]]:
                indices_to_keep.append(idx)
                volumes_to_keep.append(volumes[idx])
                break
            

    pq_df = pd.read_parquet(pq_fname)
    water = pq_df['series_description'].str.lower().str.endswith('_w')
    pq_df = pq_df[water]
    pq_df_z = pq_df['image_position_z'].values - pq_df['image_position_z'].values.min()
    print(patient_instance)
    rows_to_keep = []
    z_resolution = aorta_vti_reader.GetOutput().GetSpacing()[2]
    prev_bottom = -1
    for jjj, idx in enumerate(indices_to_keep):
        # writer = vtk.vtkXMLPolyDataWriter()
        # writer.SetInputData(surfaces[idx])
        # writer.SetFileName(f'/home/pdiachil/projects/aorta/{blob_basename}_surf{jjj+1}.vtp')
        # writer.Update()

        z_resolution = aorta_vti_reader.GetOutput().GetSpacing()[2]

        # extractor = vtk.vtkExtractVOI()
        # extractor.SetInputConnection(aorta_vti_reader.GetOutputPort())
        # extractor.SetVOI(0, images.shape[0]-1, 0, images.shape[1]-1, int(extremes_top[idx]//z_resolution), int(extremes_top[idx]//z_resolution)+1)
        # extractor.Update()

        # image_writer = vtk.vtkXMLImageDataWriter()
        # image_writer.SetInputConnection(extractor.GetOutputPort())
        # image_writer.SetFileName(f'/home/pdiachil/projects/aorta/{blob_basename}img{jjj+1}.vti')
        # image_writer.Update()

        # extractor = vtk.vtkExtractVOI()
        # extractor.SetInputConnection(aorta_vti_reader.GetOutputPort())
        # extractor.SetVOI(0, images.shape[0]-1, 0, images.shape[1]-1, int(extremes_bottom[idx]//z_resolution), int(extremes_bottom[idx]//z_resolution)+1)
        # extractor.Update()

        # image_writer = vtk.vtkXMLImageDataWriter()
        # image_writer.SetInputConnection(extractor.GetOutputPort())
        # image_writer.SetFileName(f'/home/pdiachil/projects/aorta/{blob_basename}bot{jjj+1}.vti')
        # image_writer.Update()

        top_loc = np.argmin(np.abs(pq_df_z - extremes_top[idx]))
        bottom_loc = np.argmin(np.abs(pq_df_z - extremes_bottom[idx]))
        print(volumes[idx])
        print(pq_df.iloc[top_loc]['dicom_name'])
        print(pq_df.iloc[bottom_loc]['dicom_name'])
        rows_to_keep.extend([top_loc, bottom_loc])

        if jjj > 0:
            mid_loc = np.argmin(np.abs(pq_df_z - (extremes_top[idx] + prev_bottom)*0.5))
            rows_to_keep.append(mid_loc)
            # extractor = vtk.vtkExtractVOI()
            # extractor.SetInputConnection(aorta_vti_reader.GetOutputPort())
            # extractor.SetVOI(0, images.shape[0]-1, 0, images.shape[1]-1, int((extremes_top[idx] + prev_bottom)*0.5//z_resolution), int((extremes_top[idx] + prev_bottom)*0.5//z_resolution)+1)
            # extractor.Update()

            # image_writer = vtk.vtkXMLImageDataWriter()
            # image_writer.SetInputConnection(extractor.GetOutputPort())
            # image_writer.SetFileName(f'/home/pdiachil/projects/aorta/{blob_basename}mid{jjj+1}.vti')
            # image_writer.Update()
        prev_bottom = extremes_bottom[idx]

    pq_to_keep = pq_df.iloc[rows_to_keep]
    pq_to_keep['patient_instance'] = patient_instance
    pq_to_keep['n_connected_regions'] = nregions
    pq_to_keep['n_regions_for_extremes'] = len(indices_to_keep)
    pq_to_keep.to_csv(f'/home/pdiachil/projects/aorta/{patient_instance}_slices_to_segment.csv', index=False)

    os.remove(vti_fname)
    os.remove(pq_fname)
    os.remove(images_fname)
    os.remove(model_fname)

# # %%
# import glob
# csvs = glob.glob('/home/pdiachil/projects/aorta/*slices_to_segment.csv')
# results = pd.DataFrame()
# for csv in csvs:
#     results = pd.concat([results, pd.read_csv(csv)])

# # %%
# patients = pd.read_csv('/home/pdiachil/projects/aorta/aortas_predicted.csv', names=['predicted'])
# patient_list = [p['predicted'].split('__')[0].split('/')[-1] for i, p in patients.iterrows() if p['predicted'].endswith('.bin')]

# remaining = set(patient_list) - set(results['patient_instance'])
# len(remaining)

# # %%
# remaining_df = pd.DataFrame({'predicted': list(remaining)})

# # %%
# remaining_df.to_csv('/home/pdiachil/projects/aorta/remaining.csv', index=False)
# # %%
# import seaborn as sns

# f, ax = plt.subplots()
# sns.histplot(
#     results.groupby('patient_instance').sample(1)['n_connected_regions'], bins=range(50),
#     ax=ax
# )
# f.savefig('regions_distributions.png', dpi=500)

# # %%
# f, ax = plt.subplots()
# sns.histplot(
#     results.groupby('patient_instance').sample(1)['n_regions_for_extremes'],
#     ax=ax
# )
# f.savefig('regions_for_extremes.png', dpi=500)
# # %%
# results[results['n_regions_for_extremes']<1.5].sample(1)[['patient_instance']]

# # %%
# results[results['n_regions_for_extremes']>1.5].sample(1)[['patient_instance']]


# # %%
# # pq = pd.read_parquet('/home/pdiachil/projects/aorta/bodymri_1002337_2_0_2.pq')
# # # %%
# # aorta_model_itk = itk.image_from_vtk_image(aorta_vti_reader.GetOutput())
# # skeleton = itk.BinaryThinningImageFilter3D.New(aorta_model_itk)

# # #%%
# # start = int(sys.argv[1])
# # end = int(sys.argv[2])

# # os.chdir('/home/pdiachil/ml/notebooks/mri')

# # # start = 0
# # # end = 1

# # bodymris = pd.read_csv('/home/pdiachil/ml/notebooks/mri/bodymris.csv')

# # # bodymris_done = pd.read_csv('/home/pdiachil/projects/aorta/bodymris_done.csv')
# # rows = bodymris.iloc[start:end]
# # storage_client = storage.Client('broad-ml4cvd')
# # bucket = storage_client.get_bucket('bulkml4cvd')
# # # bodymris['patient'] = bodymris['filepath'].str.split('/').str[-1].str.split('_').str[0].apply(int)
# # # bodymris['instance'] = bodymris['filepath'].str.split('/').str[-1].str.split('_').str[2].apply(int)
# # # bodymris['patient_instance'] = bodymris['patient'].apply(str)+'_'+bodymris['instance'].apply(str)
# # # bodymris_done['patient'] = bodymris_done['filepath'].str.split('/').str[-1].str.split('_').str[0].apply(int)
# # # bodymris_done['instance'] = bodymris_done['filepath'].str.split('/').str[-1].str.split('_').str[2].apply(int)
# # # bodymris_done['patient_instance'] = bodymris_done['patient'].apply(str)+'_'+bodymris['instance'].apply(str)

# # # todo_still = list(set(bodymris['patient_instance'])-set(bodymris_done['patient_instance']))
# # # bodymris = bodymris[bodymris['patient_instance'].isin(todo_still)]
# # # bodymris.to_csv('/home/pdiachil/ml/notebooks/mri/bodymris.csv')
# # # %%

# # for i, row in rows.iterrows():
# #     prefix = f'bodymri/all/raw/{row["patient"]}_20201_{row["instance"]}_0.zip'
# #     blobs = storage_client.list_blobs(bucket, prefix=prefix)
# #     for blob in blobs:
# #         blob.download_to_filename(f'/home/pdiachil/projects/aorta/{row["patient"]}_20201_{row["instance"]}_0.zip')

# #     patient = row['patient']
# #     ingest_mri_dicoms_zipped(
# #         sample_id=patient,
# #         instance=2,
# #         file=f'/home/pdiachil/projects/aorta/{patient}_20201_{row["instance"]}_0.zip',
# #         destination=f'bodymri_allraw_{patient}_{row["instance"]}_0',
# #         in_memory=True,
# #         save_dicoms=False,
# #         output_name=f'bodymri_{patient}_{row["instance"]}_0',
# #     )

# #     patient = row['patient']
# #     os.makedirs(f'/home/pdiachil/ml/notebooks/mri/bodymri_allraw_{patient}_{row["instance"]}_0/projected', exist_ok=True)
# #     build_projection_hd5(
# #         f'bodymri_allraw_{patient}_{row["instance"]}_0/bodymri_{patient}_{row["instance"]}_0.h5',
# #         f'bodymri_allraw_{patient}_{row["instance"]}_0',
# #         f'bodymri_allraw_{patient}_{row["instance"]}_0/projected'
# #     )

# #     hd5 = h5py.File(f'/home/pdiachil/ml/notebooks/mri/bodymri_allraw_{patient}_{row["instance"]}_0/bodymri_{patient}_{row["instance"]}_0.h5', 'r')
# #     meta_data = pd.read_parquet(f'/home/pdiachil/ml/notebooks/mri/bodymri_allraw_{patient}_{row["instance"]}_0/bodymri_{patient}_{row["instance"]}_0_2.pq')

# #     station_z_scales = 3.0, 4.5, 4.5, 4.5, 3.5, 4.0
# #     # station_z_scales = [scale / 3 for scale in station_z_scales]
# #     station_xscale = meta_data['col_pixel_spacing_mm'].mean()
# #     station_yscale = meta_data['row_pixel_spacing_mm'].mean()

# #     data = {
# #             int(name): read_compressed(hd5[f'instance/2/series/{name}'])
# #             for name in hd5[f'instance/2/series']
# #         }

# #     z_pos = meta_data.groupby("series_number")["image_position_z"].agg(["min", "max"])
# #     slices = build_z_slices(
# #             [data[i].shape[-1] for i in range(1, 25, 4)],
# #             [z_pos.loc[i] for i in range(1, 25, 4)],
# #             )

# #     horizontal_lines = [
# #             (idx.stop - idx.start) * scale for idx, scale in zip(slices, station_z_scales)
# #         ]
# #     horizontal_lines = np.cumsum(horizontal_lines).astype(np.uint16)[:-1]
# #     body = {"horizontal_line_idx": horizontal_lines}

# #     for type_idx, series_type_name in zip(range(4), ("in", "opp", "f", "w")):
# #         print(type_idx)
# #         if type_idx < 3:
# #             continue
# #         full_slice_to_stack = []
# #         for station_idx in range(1, 25, 4):  # neck, upper ab, lower ab, legs
# #             print(station_idx)
# #             series_num = station_idx + type_idx
# #             station_slice = slices[station_idx // 4]
# #             scale = station_z_scales[station_idx // 4]
# #             full_slice = data[series_num][..., station_slice]
# #             full_slice_scaled = zoom(full_slice, [station_xscale, station_yscale, scale])
# #             full_slice_to_stack.append(full_slice_scaled)
# #         body[f"{series_type_name}"] = normalize(center_pad_stack_3d(full_slice_to_stack))

# #     for b, bb in body.items():
# #         if 'line' in b:
# #             continue
# #         if b != 'w':
# #             continue
# #         img = vtk.vtkImageData()
# #         img.SetOrigin(0.0, 0.0, 0.0)
# #         img.SetExtent(0, bb.shape[1]-1, 0, bb.shape[0]-1, 0, bb.shape[2]-1)
# #         img.SetSpacing(1.0, 1.0, 1.0)
        

# #         bbt = bb.swapaxes(0, 1)
# #         bbt = np.flip(bbt, axis=2)
# #         arr_vtk = ns.numpy_to_vtk(bbt.ravel('F'), deep=True, array_type=vtk.VTK_INT)
# #         arr_vtk.SetName('ImageScalars')
# #         img.GetPointData().SetScalars(arr_vtk)

# #         resize = vtk.vtkImageResize()
# #         resize.SetInputData(img)
# #         resize.SetOutputDimensions(bbt.shape[0]//2, bbt.shape[1]//2, bbt.shape[2]//2)
# #         resize.Update()
# #         img_writer = vtk.vtkXMLImageDataWriter()
# #         img_writer.SetInputConnection(resize.GetOutputPort())
# #         img_writer.SetFileName(f'bodymri_allraw_{patient}_{row["instance"]}_0/{b}.vti')
# #         img_writer.Update()

# #         size = [
# #             resize.GetOutput().GetExtent()[1]+1,
# #             resize.GetOutput().GetExtent()[3]+1,
# #             resize.GetOutput().GetExtent()[5]+1
# #         ]
# #         images_arr = ns.vtk_to_numpy(resize.GetOutput().GetPointData().GetArray('ImageScalars')).reshape(size, order='F')

# #         np.save(f'{patient}_20201_{row["instance"]}_0_images', images_arr)

# # # %%

# # %%
