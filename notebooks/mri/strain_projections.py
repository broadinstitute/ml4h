# %%
# from ml4h.applications.ingest.ingest_mri import ingest_mri_dicoms_zipped, read_compressed
# from ml4h.applications.ingest.two_d_projection import build_projection_hd5, build_z_slices, normalize
from vtk.util import numpy_support
from google.cloud import storage
import os
import h5py
import pandas as pd
import numpy as np
import vtk
import pickle
import logging
import glob
import blosc
import sys
from pydicom.valuerep import DSfloat

#%%

mri_notebooks_dir = '/home/pdiachil/ml/notebooks/mri/'

start = int(sys.argv[1])
end = int(sys.argv[2])
import random

df_dic = {'sample_id': []}
df_dic['rep_time'] = []
for stra, strain in enumerate(['circ', 'rad']):
    for sli, slice in enumerate(['bas', 'mid', 'api']):
        for sec, sector in enumerate(['S', 'A', 'L', 'P']):
            for t in range(20):
                df_dic[f'{strain}_{slice}_{sector}_{t}'] = []

with open('/home/pdiachil/ml/notebooks/mri/list_of_patients.csv', 'r') as patient_list:
    for ccc, line in enumerate(patient_list):
        if ccc < start or ccc >= end:
            continue
        if not line.endswith('.h5\n'):
            continue

        patient = int(line.split('_')[-3])
        logging.warning(f'Processing {patient}; {ccc}')

        df_dic['sample_id'].append(patient)
        for stra, strain in enumerate(['circ', 'rad']):
            for sli, slice in enumerate(['bas', 'mid', 'api']):
                for sec, sector in enumerate(['S', 'A', 'L', 'P']):
                    for t in range(20):
                        df_dic[f'{strain}_{slice}_{sector}_{t}'].append(-1e-3)

        storage_client = storage.Client('broad-ml4cvd')
        bucket = storage_client.get_bucket('ml4cvd')

        prefix = f'mdrk/ukb_bulk/cardiac/ingested/BROAD_ml4h_mdrk__cardiac__{patient}'
        blobs = storage_client.list_blobs(bucket, prefix=prefix)
        
        blob_cnt = 0
        for blob in blobs:
            filename = os.path.basename(blob.name)
            blob.download_to_filename(os.path.join(mri_notebooks_dir, filename))
            blob_cnt += 1

        if blob_cnt < 1:
            logging.warning(f'Patient {patient} does not have MRI images. Skipping')
            continue

        hd5 = h5py.File(os.path.join(mri_notebooks_dir, filename), 'r')
        try:
            slices = hd5['field/20211/instance/2'].keys()
        except KeyError:
            logging.warning(f'Patient {patient} does not have strain images. Skipping')
            continue

        prefix = f'pdiachil/surface_reconstruction/sax_4ch/fastai_v20201202_v20201122/xdmf/poisson_fastai_sax_v20201202_lax_v20201122_RV_{patient}_2'
        blobs = storage_client.list_blobs(bucket, prefix=prefix)
        blob_cnt = 0
        for blob in blobs:
            filename = os.path.basename(blob.name)
            blob.download_to_filename(os.path.join(mri_notebooks_dir, filename))
            blob_cnt += 1

        if blob_cnt < 1:
            logging.warning(f'Patient {patient} does not have RV reconstruction. Skipping')
            continue

        rv_filename = os.path.join(mri_notebooks_dir, filename.split('.')[0])
        rv_hd5 = h5py.File(f'{rv_filename}.hd5', 'r')
        rv_points = rv_hd5['points_0'][()]
        rv_cells = rv_hd5['cells_0'][()]
        rv_cells = rv_cells.reshape(-1, 4)
        rv_cells = np.ascontiguousarray(rv_cells[:, 1:].ravel())

        rv_grid = vtk.vtkPolyData()
        rv_vtk_points = vtk.vtkPoints()
        rv_vtk_points.SetData(numpy_support.numpy_to_vtk(rv_points))
        rv_vtk_cells = vtk.vtkCellArray()
        rv_vtk_cells.SetData(3, numpy_support.numpy_to_vtkIdTypeArray(rv_cells))
        rv_grid.SetPoints(rv_vtk_points)
        rv_grid.SetPolys(rv_vtk_cells)

        prefix = f'mdrk/cardiac_strain/results_updated/{patient}'
        blobs = storage_client.list_blobs(bucket, prefix=prefix)
        blob_cnt = 0
        for blob in blobs:
            filename = os.path.basename(blob.name)
            blob.download_to_filename(os.path.join(mri_notebooks_dir, filename))
            blob_cnt += 1
        if blob_cnt < 1:
            logging.warning(f'Patient {patient} does not have strain landmarks. Skipping')
            continue

        strain_hd5 = h5py.File(os.path.join(mri_notebooks_dir, filename))

        C_sector_rad = np.ones((len(slices), 20, 4))*(-1e3)
        C_sector_circ = np.ones((len(slices), 20, 4))*(-1e3)
        for sss, (slice, slice_name) in enumerate(zip(slices, ['bas', 'mid', 'api'])):
            serieses = hd5[f'field/20211/instance/2/{slice}/series'].keys()
            for series in serieses:
                if 'pandas' in series:
                    df = pickle.loads(blosc.decompress(
                            hd5[f'field/20211/instance/2/{slice}/series/{series}'][()]
                            ))
                else:
                    img = np.frombuffer(blosc.decompress(
                            hd5[f'field/20211/instance/2/{slice}/series/{series}'][()]
                            ), 
                            dtype=np.uint16
                            )
            series = series.split('_')[0]
            sample = df.sample(1)
            nrows = sample['Rows'].values[0]
            ncols = sample['Columns'].values[0]
            height = sample['Pixel Spacing'].values[0][0].real
            width = sample['Pixel Spacing'].values[0][1].real
            rep_time = sample['Repetition Time'].values[0]
            if sss == 0:
                df_dic['rep_time'].append(rep_time)
            positions = np.array([
                sample['Image Position (Patient)'].values[0][0].real,
                sample['Image Position (Patient)'].values[0][1].real,
                sample['Image Position (Patient)'].values[0][2].real,
            ])
            # print(positions)

            orientations = np.array([
                [
                    sample['Image Orientation (Patient)'].values[0][3].real,
                    sample['Image Orientation (Patient)'].values[0][4].real,
                    sample['Image Orientation (Patient)'].values[0][5].real,
                ],
                [
                    sample['Image Orientation (Patient)'].values[0][0].real,
                    sample['Image Orientation (Patient)'].values[0][1].real,
                    sample['Image Orientation (Patient)'].values[0][2].real,
                ]
            ])

            thickness = sample['Slice Thickness'].values[0]

            img = img.reshape(-1, nrows, ncols)

            grid = vtk.vtkStructuredGrid()
            nslices = 1
            transform = vtk.vtkTransform()
            n_orientation = np.cross(orientations[1, :], orientations[0, :])
            # 4x4 transform matrix to align to the patient reference system
            transform.SetMatrix([
                orientations[1, 0]*height, orientations[0, 0]*width, n_orientation[0]*thickness, positions[0],
                orientations[1, 1]*height, orientations[0, 1]*width, n_orientation[1]*thickness, positions[1],
                orientations[1, 2]*height, orientations[0, 2]*width, n_orientation[2]*thickness, positions[2],
                0, 0, 0, 1,
            ])
            x_coors = np.arange(0, ncols+1) - 0.5
            y_coors = np.arange(0, nrows+1) - 0.5
            z_coors = np.arange(0, 1+1) - 0.5
            xyz_meshgrid = np.meshgrid(x_coors, y_coors, z_coors, copy=True, indexing='ij')
            xyz_pts = np.zeros(((ncols+1) * (nrows+1) * (1+1), 3))
            for dim in range(3):
                xyz_pts[:, dim] = xyz_meshgrid[dim].ravel(order='F')
            vtk_pts = vtk.vtkPoints()
            vtk_pts.SetData(numpy_support.numpy_to_vtk(xyz_pts))
            grid.SetPoints(vtk_pts)
            grid.SetDimensions(len(x_coors), len(y_coors), len(z_coors))
            grid.SetExtent(0, len(x_coors)-1, 0, len(y_coors)-1, 0, len(z_coors)-1)

            transform_filter = vtk.vtkTransformFilter()
            transform_filter.SetInputData(grid)
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            grid.DeepCopy(transform_filter.GetOutput())

            slice_centers = vtk.vtkCellCenters()
            slice_centers.SetInputData(grid)
            slice_centers.Update()

            slice_points = numpy_support.vtk_to_numpy(slice_centers.GetOutput().GetPoints().GetData())
            slice_cog = np.mean(slice_points, axis=0)

            slice_plane = vtk.vtkPlane()
            slice_plane.SetOrigin(slice_cog[0], slice_cog[1], slice_cog[2])
            slice_normal = np.cross(slice_points[5*ncols] - slice_points[0], slice_points[1] - slice_points[0])
            slice_normal = slice_normal / np.linalg.norm(slice_normal)
            slice_plane.SetNormal(slice_normal[0], slice_normal[1], slice_normal[2])

            rv_slice_cutter = vtk.vtkCutter()
            rv_slice_cutter.SetCutFunction(slice_plane)
            rv_slice_cutter.SetInputData(rv_grid)
            rv_slice_cutter.Update()
            
            rv_slice_points = numpy_support.vtk_to_numpy(rv_slice_cutter.GetOutput().GetPoints().GetData())
            mul_rv = 0.0
            should_skip = False
            while len(rv_slice_points) == 0:                
                mul_rv = (mul_rv + 5.0) if sss < 2 else (mul_rv - 5.0)
                logging.warning(f'Patient {patient} slice {slice_name} does not intersect with RV. Shifting {mul_rv} mm')
                slice_plane.SetOrigin(slice_cog[0]+slice_normal[0]*mul_rv, slice_cog[1]+slice_normal[1]*mul_rv, slice_cog[2]+slice_normal[2]*mul_rv)
                rv_slice_cutter.Update()
                rv_slice_points = numpy_support.vtk_to_numpy(rv_slice_cutter.GetOutput().GetPoints().GetData())

                if np.abs(mul_rv) > 50.0:
                    should_skip = True
                    break
            if should_skip:
                logging.warning(f'Patient {patient} slice {slice_name} does not intersect with RV. Skipping')
                continue



            # rv_slice_writer = vtk.vtkXMLPolyDataWriter()
            # rv_slice_writer.SetInputConnection(rv_slice_cutter.GetOutputPort())
            # rv_slice_writer.SetFileName('rv_slice.vtp')
            # rv_slice_writer.Update()

            slice_cutter = vtk.vtkCutter()
            slice_cutter.SetCutFunction(slice_plane)
            slice_cutter.SetInputData(grid)
            slice_cutter.Update()

            strain_landmarks = strain_hd5['landmark_predictions'][()][sss]
            ratios = strain_hd5['resize_ratios']
            bbox = strain_hd5['bbox_predictions'][()][sss]

            # f, ax = plt.subplots()
            # ax.imshow(img[0, :, :])
            # ax.set_title(f'{patient} {slice_name}')

            ny, nx = np.shape(img[0, :, :])
            pad_x = 0
            pad_y = 0
            if (nx != ny):
                max_size = max(nx, ny)
                pad_x = (max_size-nx)//2
                pad_y = (max_size-ny)//2

            for t in range(min(img.shape[0], 20)):
                img_vtk = numpy_support.numpy_to_vtk(img[t, :, :].ravel(order='C'), deep=True)
                img_vtk.SetName(f'tagged_mri')
                grid.GetCellData().AddArray(img_vtk)

                # strain_x_coors = strain_landmarks[t, 0, :].reshape(-1, 1) / ratios[0] + bbox[0]
                strain_x_coors = strain_landmarks[t, 0, :].reshape(-1, 1) - pad_x
                # strain_y_coors = strain_landmarks[t, 1, :].reshape(-1, 1) / ratios[0] + bbox[1]
                strain_y_coors = strain_landmarks[t, 1, :].reshape(-1, 1) - pad_y
                strain_z_coors = np.zeros_like(strain_x_coors)

    #             if t == 5:
    #                 break
    #         break
    #     break
    # f, ax = plt.subplots()

    
    # if (nx != ny):
    #     max_size = max(nx,ny)
    #     pad_x = (max_size-nx)//2
    #     pad_y = (max_size-ny)//2
    #     new_img = np.pad(img[5, :, :], ((pad_x,pad_x),(pad_y,pad_y)), 'constant')
    # else:
    #     new_img = img[5, :, :]

    # # new_img = np.full((256, 256), 0)
    # # x_center = (256 - ncols) // 2
    # # y_center = (256 - nrows) // 2

    # # new_img[y_center:y_center+nrows, x_center:x_center+ncols] = img[0, :, :]
    # ax.imshow(img[5, :, :])
    # rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    # ax.set_title(f'{patient} {slice_name}')
    # ax.add_patch(rect)
    # ax.plot(strain_x_coors, strain_y_coors, 'o', markersize=2)

    # # f, ax = plt.subplots()
    # # ax.imshow(new_img[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])])

    # # %%
                strain_xyz = np.hstack([strain_x_coors, strain_y_coors, strain_z_coors])
                strain_grid = vtk.vtkPolyData()
                strain_vtk_points = vtk.vtkPoints()
                strain_vtk_points.SetData(numpy_support.numpy_to_vtk(strain_xyz))
                strain_grid.SetPoints(strain_vtk_points)

                strip = np.array([
                    [0, 1, 8, 7],
                    [1, 2, 9, 8],
                    [2, 3, 10, 9],
                    [3, 4, 11, 10],
                    [4, 5, 12, 11],
                    [5, 6, 13, 12]
                ], dtype=np.int64)

                strain_cell_list = []
                for i in range(24):
                    this_strip = (strip+i*7) % 168
                    strain_cell_list.append(this_strip)

                strain_cells = np.vstack(strain_cell_list)

                strain_vtk_cells = vtk.vtkCellArray()
                strain_vtk_cells.SetData(4, numpy_support.numpy_to_vtk(strain_cells.ravel()))
                strain_grid.SetPolys(strain_vtk_cells)

                transform_filter = vtk.vtkTransformFilter()
                transform_filter.SetInputData(strain_grid)
                transform_filter.SetTransform(transform)
                transform_filter.Update()
                strain_grid = vtk.vtkPolyData()
                strain_grid.DeepCopy(transform_filter.GetOutput())
                strain_grid_points = numpy_support.vtk_to_numpy(strain_grid.GetPoints().GetData())

                # writer = vtk.vtkXMLPolyDataWriter()
                # writer.SetFileName(f'{patient}_{slice_name}_strain_grid_debug.vtp')
                # writer.SetInputData(strain_grid)
                # writer.Update()

                if t == 0:
                    strain_grid_reference = vtk.vtkPolyData()
                    strain_grid_reference.DeepCopy(strain_grid)
                    strain_grid_reference_points = numpy_support.vtk_to_numpy(strain_grid_reference.GetPoints().GetData())
                    
                    strain_grid_locator = vtk.vtkPointLocator()
                    strain_grid_locator.SetDataSet(strain_grid)
                    strain_grid_locator.BuildLocator()

                    rv_close_points = []
                    for rv_slice_pt in rv_slice_points:
                        rv_close_points.append(strain_grid_locator.FindClosestPoint(rv_slice_pt))
                    
                    rv_close_points = list(set(rv_close_points))#.intersection(set(range(6, 169, 7))))
                    rv_close_elements = []
                    rv_close_strips = []
                    for rv_close_point in rv_close_points:
                        new_strip = rv_close_point//7
                        rv_close_strips.append(new_strip)
                        rv_close_elements += list(range(new_strip*6, new_strip*6+6))
                    rv_close_elements = sorted(list(set(rv_close_elements)))
                    rv_close_strips = sorted(list(set(rv_close_strips)))
                    smaller_than_12 = [r for r in rv_close_strips if r < 12]
                    if smaller_than_12:
                        last_rv_close_strip = max(smaller_than_12) 
                    else:
                        last_rv_close_strip = max(rv_close_strips)
                    
                    sector_width = (24 - len(rv_close_strips)) // 3
                    anterior_strips = []
                    anterior_elements = []
                    for jj in range(sector_width):
                        new_strip = (last_rv_close_strip+1+jj) % 24
                        anterior_strips.append(new_strip)
                        anterior_elements += list(range(new_strip*6, new_strip*6+6))
                    left_strips = []
                    left_elements = []
                    for jj in range(sector_width):
                        new_strip = (anterior_strips[-1]+1+jj) % 24
                        left_strips.append(new_strip)
                        left_elements += list(range(new_strip*6, new_strip*6+6))
                    posterior_strips = []
                    posterior_elements = []
                    for jj in range(24 - len(rv_close_strips) - len(anterior_strips) - len(left_strips)):
                        new_strip = (left_strips[-1]+1+jj) % 24
                        posterior_strips.append(new_strip)
                        posterior_elements += list(range(new_strip*6, new_strip*6+6))
                    rv_array = np.zeros((144,), dtype=np.int64)
                    rv_array[rv_close_elements] = 1
                    rv_array[anterior_elements] = 2
                    rv_array[left_elements] = 3
                    rv_array[posterior_elements] = 0
                    rv_vtk_array = numpy_support.numpy_to_vtkIdTypeArray(rv_array)
                    rv_vtk_array.SetName('Sectors')
                    

                    strain_grid_reference.GetCellData().AddArray(rv_vtk_array)

                    indices = np.array(list(range(len(strain_grid_reference_points))))
                    indices_next = (indices + 7) % 168
                    circ_array = strain_grid_reference_points[indices_next] - strain_grid_reference_points
                    circ_array = circ_array / np.linalg.norm(circ_array, axis=1).reshape(-1, 1)

                    circ_vtk_array = numpy_support.numpy_to_vtk(circ_array)
                    circ_vtk_array.SetName('Circ')

                    rad_array = np.mean(strain_grid_reference_points, axis=0).reshape(1, -1) - strain_grid_reference_points
                    rad_array = rad_array / np.linalg.norm(rad_array, axis=1).reshape(-1, 1)
                    rad_vtk_array = numpy_support.numpy_to_vtk(rad_array)
                    rad_vtk_array.SetName('Rad')

                    strain_grid_reference.GetPointData().AddArray(circ_vtk_array)
                    strain_grid_reference.GetPointData().AddArray(rad_vtk_array)

                strain_grid.GetCellData().AddArray(rv_vtk_array)
                strain_grid_points = numpy_support.vtk_to_numpy(strain_grid.GetPoints().GetData())
                displacement_array = strain_grid_points - strain_grid_reference_points
                displacement_vtk_array = numpy_support.numpy_to_vtk(displacement_array)
                displacement_vtk_array.SetName('u')
                strain_grid_reference.GetPointData().AddArray(displacement_vtk_array)

                gradient = vtk.vtkGradientFilter()
                gradient.SetInputData(strain_grid_reference)
                gradient.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'u')
                gradient.SetResultArrayName('H')
                gradient.ComputeGradientOn()
                gradient.Update()

                h_array = numpy_support.vtk_to_numpy(gradient.GetOutput().GetPointData().GetArray('H')).reshape(-1, 3, 3)
                F_array = h_array + np.eye(3)
                C_array = np.einsum('aij,ail->ajl', F_array, F_array)
                E_array = 0.5*(C_array - np.eye(3))
                R_array = np.zeros((len(strain_grid_reference_points), 3, 3))
                R_array[:, :, 0] = rad_array
                R_array[:, :, 1] = circ_array
                R_array[:, :, 2] = slice_normal
                C_rot_array = np.einsum('aij,aik,akl -> ajl', R_array, C_array, R_array)
                C_rot_vtk_array = numpy_support.numpy_to_vtk(C_rot_array.reshape(-1, 9))
                C_rot_vtk_array.SetName('C')
                gradient.GetOutput().GetPointData().AddArray(C_rot_vtk_array)

                gradient_cells = vtk.vtkPointDataToCellData()
                gradient_cells.SetInputConnection(gradient.GetOutputPort())
                gradient_cells.Update()

                C_cell_array = numpy_support.vtk_to_numpy(gradient_cells.GetOutput().GetCellData().GetArray('C'))
                C_sector_circ[sss, t, 0] = np.mean(C_cell_array[rv_close_elements, 4])
                C_sector_circ[sss, t, 1] = np.mean(C_cell_array[anterior_elements, 4])
                C_sector_circ[sss, t, 2] = np.mean(C_cell_array[left_elements, 4])
                C_sector_circ[sss, t, 3] = np.mean(C_cell_array[posterior_elements, 4])

                C_sector_rad[sss, t, 0] = np.mean(C_cell_array[rv_close_elements, 0])
                C_sector_rad[sss, t, 1] = np.mean(C_cell_array[anterior_elements, 0])
                C_sector_rad[sss, t, 2] = np.mean(C_cell_array[left_elements, 0])
                C_sector_rad[sss, t, 3] = np.mean(C_cell_array[posterior_elements, 0])

                df_dic[f'circ_{slice_name}_S_{t}'][-1] = C_sector_circ[sss, t, 0]
                df_dic[f'circ_{slice_name}_A_{t}'][-1] = C_sector_circ[sss, t, 1]
                df_dic[f'circ_{slice_name}_L_{t}'][-1] = C_sector_circ[sss, t, 2]
                df_dic[f'circ_{slice_name}_P_{t}'][-1] = C_sector_circ[sss, t, 3]

                df_dic[f'rad_{slice_name}_S_{t}'][-1] = C_sector_rad[sss, t, 0]
                df_dic[f'rad_{slice_name}_A_{t}'][-1] = C_sector_rad[sss, t, 1]
                df_dic[f'rad_{slice_name}_L_{t}'][-1] = C_sector_rad[sss, t, 2]
                df_dic[f'rad_{slice_name}_P_{t}'][-1] = C_sector_rad[sss, t, 3]

                # writer = vtk.vtkXMLPolyDataWriter()
                # writer.SetFileName(f'{patient}_{slice_name}_strain_grid_{t}.vtp')
                # writer.SetInputConnection(gradient_cells.GetOutputPort())
                # writer.Update()

                # slice_plane.SetOrigin(slice_cog[0], slice_cog[1], slice_cog[2])
                # slice_cutter.SetInputData(grid)
                # slice_cutter.Update()

                # slice_writer = vtk.vtkXMLPolyDataWriter()
                # slice_writer.SetInputConnection(slice_cutter.GetOutputPort())
                # slice_writer.SetFileName(f'{patient}_{slice_name}_slice_{t}.vtp')
                # slice_writer.Update()

        remove_list = glob.glob(f'{mri_notebooks_dir}*{patient}*')
        remove_list += glob.glob(f'{mri_notebooks_dir}*.h5')
        remove_list += glob.glob(f'{mri_notebooks_dir}*.hd5')
        for r in remove_list:
            if os.path.isfile(r):
                os.remove(r)

        df = pd.DataFrame(df_dic)
        df.to_csv(os.path.join(mri_notebooks_dir, f'df_strain_{start}_{end}.csv'))




# # %%
# import matplotlib.pyplot as plt

# sector_colors = {
#     'S': [1.0, 0.0, 0.0],
#     'A': [0.0, 1.0, 0.0],
#     'L': [0.0, 0.0, 1.0],
#     'P': [0.0, 0.0, 0.0]
# }
# f, ax = plt.subplots()
# time = np.arange(20)
# for i, (label, color) in enumerate(sector_colors.items()):
#     ax.plot(time, np.sqrt(C_sector_circ[2, :, i]), color = color, label = label)
# ax.set_xlabel('Frame')
# ax.set_ylabel('Circumferential stretch')
# ax.set_xlim([0, 19])
# ax.set_xticks([5, 10, 15])
# ax.legend()
# f, ax = plt.subplots()
# for i, (label, color) in enumerate(sector_colors.items()):
#     ax.plot(time, np.sqrt(C_sector_rad[2, :, i]), color = color, label = label)
# ax.set_xlabel('Frame')
# ax.set_ylabel('Radial stretch')
# ax.set_xlim([0, 19])
# ax.set_xticks([5, 10, 15])
# # %%
# # !rm bodymri_allraw_1000107_2_0/projected/bodymri_1000107.h5


# # %%
# import h5py
# hd5 = h5py.File('bodymri_allraw_1000107_2_0/projected/bodymri_1000107.h5')
# # %%
# arr = read_compressed(hd5['instance/2/w_sagittal'])
# # %%
# import matplotlib.pyplot as plt
# plt.imshow(arr)
# # %%
# old_hd5 = h5py.File('bodymri_allraw_1000107_2_0/bodymri_1000107.h5', 'r')
# # %%
# instance = 2
# data = {
#                 int(name): read_compressed(old_hd5[f'instance/{instance}/series/{name}'])
#                 for name in old_hd5[f'instance/{instance}/series']
#             }
# # %%
# import matplotlib.pyplot as plt
# plt.imshow(data[4][:, 119, :])
# # %%

# # %%
# import h5py
# import pandas as pd
# from scipy.ndimage import zoom
# import numpy as np

# def center_pad_3d(x: np.ndarray, width: int) -> np.ndarray:
#     """Pad an image on the left and right with 0s to a specified
#     target width.
#     Args:
#         x (np.ndarray): Input data.
#         width (int): Desired width.
#     Returns:
#         np.ndarray: Padded data.
#     """
#     new_x = np.zeros((width, x.shape[1], x.shape[2]))
#     offset = (width - x.shape[0]) // 2
#     new_x[offset : offset+x.shape[0], ...] = x
#     return new_x


# def center_pad_stack_3d(xs: np.ndarray) -> np.ndarray:
#     """Center and pad input data and then stack the images
#     with different widths.
#     Args:
#         xs (np.ndarray): Input data.
#     Returns:
#         np.ndarray: Center-padded and stacked data.
#     """
#     max_width = max(x.shape[0] for x in xs)
#     return np.concatenate([center_pad_3d(x, max_width) for x in xs], axis=2)

# hd5 = h5py.File('/home/pdiachil/ml/notebooks/mri/bodymri_allraw_1000107_2_0/bodymri_1000107.h5', 'r')
# meta_data = pd.read_parquet('/home/pdiachil/ml/notebooks/mri/bodymri_allraw_1000107_2_0/bodymri_1000107_2.pq')

# station_z_scales = 3.0, 4.5, 4.5, 4.5, 3.5, 4.0
# # station_z_scales = [scale / 3 for scale in station_z_scales]
# station_xscale = meta_data['col_pixel_spacing_mm'].mean()
# station_yscale = meta_data['row_pixel_spacing_mm'].mean()

# data = {
#         int(name): read_compressed(hd5[f'instance/2/series/{name}'])
#         for name in hd5[f'instance/2/series']
#        }

# z_pos = meta_data.groupby("series_number")["image_position_z"].agg(["min", "max"])
# slices = build_z_slices(
#            [data[i].shape[-1] for i in range(1, 25, 4)],
#            [z_pos.loc[i] for i in range(1, 25, 4)],
#         )

# horizontal_lines = [
#         (idx.stop - idx.start) * scale for idx, scale in zip(slices, station_z_scales)
#     ]
# horizontal_lines = np.cumsum(horizontal_lines).astype(np.uint16)[:-1]
# body = {"horizontal_line_idx": horizontal_lines}

# for type_idx, series_type_name in zip(range(4), ("in", "opp", "f", "w")):
#     full_slice_to_stack = []
#     for station_idx in range(1, 25, 4):  # neck, upper ab, lower ab, legs
#         series_num = station_idx + type_idx
#         station_slice = slices[station_idx // 4]
#         scale = station_z_scales[station_idx // 4]
#         full_slice = data[series_num][..., station_slice]
#         full_slice_scaled = zoom(full_slice, [station_xscale, station_yscale, scale])
#         full_slice_to_stack.append(full_slice_scaled)
#     body[f"{series_type_name}"] = normalize(center_pad_stack_3d(full_slice_to_stack))
# # %%
# import vtk
# from vtk.util import numpy_support as ns
# for b, bb in body.items():
#     if 'line' in b:
#         continue
#     img = vtk.vtkImageData()
#     img.SetOrigin(0.0, 0.0, 0.0)
#     img.SetExtent(0, bb.shape[1]-1, 0, bb.shape[0]-1, 0, bb.shape[2]-1)
#     img.SetSpacing(1.0, 1.0, 1.0)
    

#     bbt = bb.swapaxes(0, 1)
#     bbt = np.flip(bbt, axis=2)
#     # bbt = bbt.reshape(*reversed(bb.shape))
#     # nslice = bbt.shape[0]*bb.shape[1]
#     # for z_slice in range(bbt.shape[2]):
#     #     arr[nslice*z_slice:nslice*(z_slice+1)] = bbt[:, :, z_slice].ravel('F')
#     arr_vtk = ns.numpy_to_vtk(bbt.ravel('F'), deep=True, array_type=vtk.VTK_INT)
#     arr_vtk.SetName('ImageScalars')
#     img.GetPointData().SetScalars(arr_vtk)

#     resize = vtk.vtkImageResize()
#     resize.SetInputData(img)
#     resize.SetOutputDimensions(bbt.shape[0]//2, bbt.shape[1]//2, bbt.shape[2]//2)
#     resize.Update()
#     img_writer = vtk.vtkXMLImageDataWriter()
#     img_writer.SetInputConnection(resize.GetOutputPort())
#     img_writer.SetFileName(f'{b}.vti')
#     img_writer.Update()
# # %%

# %%
