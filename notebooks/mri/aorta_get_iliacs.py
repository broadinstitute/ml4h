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
import imageio


start = int(sys.argv[1])
end = int(sys.argv[2])

# start = 0
# end = 10

# %%
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.bucket('ml4cvd')

patients = pd.read_csv('/home/pdiachil/projects/aorta/bodymris.csv')
tmp = patients['filepath'].str.replace('gs://bulkml4cvd/bodymri/all/raw/', '')
tmp = tmp.str.replace('_0.zip', '')
patients['predicted'] = tmp
patient_rows = patients.iloc[start:end]
good = {'patient': [], 'ratio': []}
for i, (j, row) in enumerate(patient_rows.iterrows()):
    patient_instance = row['predicted']
    print(patient_instance)
    try:
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
    except:
        continue

    with open(model_fname, 'rb') as f: 
        model_predictions = blosc.unpack_array(f.read())
    model_predictions = np.moveaxis(model_predictions, [0, 1, 2], [2, 0, 1])

    
    images = np.load(images_fname)
    aorta_vti_reader = vtk.vtkXMLImageDataReader()
    aorta_vti_reader.SetFileName(vti_fname)
    aorta_vti_reader.Update()

    aorta_vti = numpy_support.vtk_to_numpy(aorta_vti_reader.GetOutput().GetPointData().GetArray('ImageScalars')).reshape(images.shape, order='F')
    aorta_vti_model = np.zeros_like(aorta_vti, dtype=float)

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

    # surf_writer = vtk.vtkSTLWriter()
    # surf_writer.SetInputConnection(aorta_surf.GetOutputPort())
    # surf_writer.SetFileName(f'/home/pdiachil/projects/aorta/{patient_instance}_predictions.stl')
    # surf_writer.Update()

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
    extremes_zp = []
    extremes_zm = []
    extremes_xp = []
    extremes_xm = []
    extremes_yp = []
    extremes_ym = []

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
        extremes_zp.append(np.max(surface_points[:, 2]))
        extremes_zm.append(np.min(surface_points[:, 2]))
        extremes_xp.append(np.max(surface_points[:, 0]))
        extremes_xm.append(np.min(surface_points[:, 0]))
        extremes_yp.append(np.max(surface_points[:, 1]))
        extremes_ym.append(np.min(surface_points[:, 1]))

        surfaces.append(surface.GetOutput())

        mass = vtk.vtkMassProperties()
        mass.SetInputConnection(surface.GetOutputPort())
        mass.Update()
        
        volumes.append(mass.GetSurfaceArea())

    volumes_arr = np.array(volumes)
    sorted_indices = np.argsort(volumes)[::-1]
    ratio = volumes[sorted_indices[0]] / volumes[sorted_indices[1]]
    good['patient'].append(patient_instance)
    good['ratio'].append(ratio)
    indices_to_keep = [sorted_indices[0]]
    max_vol = volumes[sorted_indices[0]]
    # volumes_to_keep = [max_vol]
    # if nregions > 1:
    #     for idx in sorted_indices[1:]:
    #         if volumes[idx] < (max_vol / 50.0):
    #             break
    #         if extremes_zp[idx] < extremes_zm[sorted_indices[0]]:
    #             indices_to_keep.append(idx)
    #             volumes_to_keep.append(volumes[idx])
    #             break
            

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
        # writer.SetFileName(f'/home/pdiachil/projects/aorta/{patient_instance}_surf{jjj+1}.vtp')
        # writer.Update()

        z_resolution = aorta_vti_reader.GetOutput().GetSpacing()[2]
        y_resolution = aorta_vti_reader.GetOutput().GetSpacing()[1]
        x_resolution = aorta_vti_reader.GetOutput().GetSpacing()[0]

        minx = int(extremes_xm[idx]//x_resolution)-50
        maxx = int(extremes_xp[idx]//x_resolution)+50
        miny = int(extremes_ym[idx]//y_resolution)-20
        maxy = int(extremes_yp[idx]//y_resolution)-20
        minz = int(extremes_zm[idx]//z_resolution)-50
        maxz = int(extremes_zm[idx]//z_resolution)+50

        if ratio < 10.0:
            minz = minz - 50
            maxz = maxz - 50

        extractor = vtk.vtkExtractVOI()
        extractor.SetInputConnection(aorta_vti_reader.GetOutputPort())
        extractor.SetVOI(
            minx, maxx, miny, maxy, minz, maxz
        )
        extractor.Update()

        # image_writer = vtk.vtkXMLImageDataWriter()
        # image_writer.SetInputConnection(extractor.GetOutputPort())
        # image_writer.SetFileName(f'/home/pdiachil/projects/aorta/{patient_instance}img{jjj+1}.vti')
        # image_writer.Update()

        chunk = numpy_support.vtk_to_numpy(extractor.GetOutput().GetPointData().GetArray('ImageScalars')).reshape((maxx-minx+1, maxy-miny+1, maxz-minz+1), order='F')
        chunk_model = numpy_support.vtk_to_numpy(extractor.GetOutput().GetPointData().GetArray('ImageScalars2')).reshape((maxx-minx+1, maxy-miny+1, maxz-minz+1), order='F')

        # Axial slices
        slices = []
        for iz in range(chunk.shape[2]):
            image = chunk[:, :, iz]
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2) / np.max(image)
            image[:, :, 0] = image[:, :, 0] + (chunk_model[:, :, iz]>0.5)*0.25
            # image = image - np.min(image, axis=0)
            # image = image / np.max(image)
            # image = np.asarray(image*255., dtype=np.uint8)    
            slices.append(np.transpose(image, (1, 0, 2)))
        imageio.mimsave(f'/home/pdiachil/projects/aorta/{patient_instance}_axial.gif', slices)

        # Coronal slices
        slices = []
        for iz in range(chunk.shape[1]):
            image = chunk[:, iz, :]
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2) / np.max(image)
            image[:, :, 0] = image[:, :, 0] + (chunk_model[:, iz, :]>0.5)*0.25
            # image = image + np.min(image)
            # image = image / np.max(image)
            # image = np.asarray(image*255., dtype=np.uint8)
            slices.append(np.flip(np.transpose(image, (1, 0, 2)), axis=0))
        imageio.mimsave(f'/home/pdiachil/projects/aorta/{patient_instance}_coronal.gif', slices)
        slices = np.array(slices)
        slices_MIP = slices.max(axis=0)
        imageio.imsave(f'/home/pdiachil/projects/aorta/{patient_instance}_coronal_MIP.gif', slices_MIP)

        # extractor = vtk.vtkExtractVOI()
        # extractor.SetInputConnection(aorta_vti_reader.GetOutputPort())
        # extractor.SetVOI(0, images.shape[0]-1, 0, images.shape[1]-1, int(extremes_bottom[idx]//z_resolution), int(extremes_bottom[idx]//z_resolution)+1)
        # extractor.Update()

        # image_writer = vtk.vtkXMLImageDataWriter()
        # image_writer.SetInputConnection(extractor.GetOutputPort())
        # image_writer.SetFileName(f'/home/pdiachil/projects/aorta/{blob_basename}bot{jjj+1}.vti')
        # image_writer.Update()

    #     # top_loc = np.argmin(np.abs(pq_df_z - extremes_top[idx]))
    #     slice_z = int(extremes_bottom[idx] / z_resolution)
    #     slice_y = int(extremes_backward[idx] / y_resolution)
    #     slice_x = int(extremes_right[idx] / x_resolution)

    # chunk = aorta_vti[max(0, slice_x-20): slice_x+20, max(0, slice_x-20): slice_y+20, slice_z-100:slice_z+100]
    # chunk_model = aorta_vti_model[max(0, slice_x-20): slice_x+20, max(0, slice_x-20): slice_y+20, slice_z-100:slice_z+100]
    

    # pq_to_keep = pq_df.iloc[rows_to_keep[1:]]
    # pq_to_keep['patient_instance'] = patient_instance
    # pq_to_keep['n_connected_regions'] = nregions
    # pq_to_keep['n_regions_for_extremes'] = len(indices_to_keep)
    # pq_to_keep.to_csv(f'/home/pdiachil/projects/aorta/{patient_instance}_slices_to_segment.csv', index=False)

    os.remove(vti_fname)
    os.remove(pq_fname)
    os.remove(images_fname)
    os.remove(model_fname)

good_df = pd.DataFrame(good)
good_df.to_csv(f'/home/pdiachil/projects/aorta/gifs_ratio_{start}_{end}.csv', index=False)

