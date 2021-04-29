# %%
import vtk
import h5py
import time
import glob
import sys
import pandas as pd
import numpy as np
from vtk.util import numpy_support as ns
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from notebooks.mri.mri_atria import to_xdmf
from parameterize_segmentation import annotation_to_poisson
from ml4h.tensormap.ukb.mri_vtk import _mri_hd5_to_structured_grids, _mri_tensor_4d
from ml4h.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES


import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from google.cloud import storage

# %%
import subprocess
from parameterize_segmentation import msh_to_vtk, improve_mesh

import logging
logging.getLogger().setLevel('INFO')
hd5s = open('/home/pdiachil/projects/manifests/patient_list_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122.csv', 'r')
storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')

# %%
start = int(sys.argv[1])
end = int(sys.argv[2])

# start = 10
# end = start+1
version='fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122'

views = ['lax_4ch']

channels = [
    [   
        MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['RA_cavity']
    ],
    [
        MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['RV_cavity'] 
    ]
]

chambers = ['RA']

view_format_string = 'cine_segmented_{view}'
annot_format_string = 'cine_segmented_{view}_annotated'
annot_time_format_string = 'cine_segmented_{view}_annotated_{t}'
annot_time_instance_format_string = 'cine_segmented_{view}_annotated_{t}/{instance}'

results = []
for chamber in chambers:
    results.append({'sample_id': [-1]*2*(end-start)})
    results[-1]['instance'] = [-1]*2*(end-start)
    for t in range(MRI_FRAMES):
        results[-1][f'{chamber}_simpson_{t}'] = [-1]*2*(end-start)

start_time = time.time()
for i, hd5 in enumerate(hd5s):
    # i = start
    # hd5 = f'/mnt/disks/segmented-sax-v20201116-lax-v20201119-petersen/2020-11-20/5362506.hd5'
    # hd5 = f'/mnt/disks/segmented-sax-lax-v20201102/2020-11-02/5362506.hd5'
    # hd5 = f'/mnt/disks/segmented-sax-v20201124-lax-v20201122/2020-11-24/4566955.hd5'
    # hd5 = f'/home/pdiachil/1000800.hd5'
    if i < start:
        continue
    if i == end:
        break    
    sample_id = hd5.split('/')[-1].replace('.hd5\n', '')
    print(sample_id)
    segmented_path = hd5.replace('gs://ml4cvd/', '')
    segmented_path = segmented_path.replace('\n', '')
    blob = bucket.blob(segmented_path)
    blob.download_to_filename('tmp.hd5')

    try:
        with h5py.File('tmp.hd5', 'r') as ff_trad:
            instances = list(ff_trad['ukb_cardiac_mri/cine_segmented_sax_b1_annotated_1'].keys())

        for ii, instance in enumerate(instances):
            results[0]['sample_id'][(i-start)*2+ii] = sample_id
            results[0]['instance'][(i-start)*2+ii] = instance
            annot_datasets = []
            orig_datasets = []
            with h5py.File('tmp.hd5', 'r') as ff_trad:
                for iv, view in enumerate(views):
                    if annot_time_instance_format_string.format(view=view, t=1, instance=instance) not in ff_trad['ukb_cardiac_mri']:
                        views = views[:iv]
                        channels[0] = channels[0][:iv]
                        break
                    annot_datasets.append(
                        _mri_hd5_to_structured_grids(
                            ff_trad, annot_format_string.format(view=view),
                            view_name=view_format_string.format(view=view),
                            instance=instance,
                            concatenate=True, annotation=True,
                            save_path=None, order='F',
                        )[0],
                    )
                    
            volumes = np.zeros(50)
            for t in range(50):
                arr = f'cine_segmented_lax_4ch_annotated_{t}'
                # Merge RA with crista terminalis
                annotation_arr = ns.vtk_to_numpy(annot_datasets[0].GetCellData().GetArray(arr))
                crista_terminalis = np.where(np.logical_and(annotation_arr>6.5, annotation_arr<7.5))[0]
                annotation_arr[crista_terminalis] = channels[0][0]

                rarv_thresh = vtk.vtkThreshold()
                rarv_thresh.SetInputData(annot_datasets[0])
                rarv_thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, arr)
                rarv_thresh.ThresholdBetween(channels[0][0]-0.5, channels[1][0]+0.5)
                rarv_thresh.Update()

                rarv_pointdata = vtk.vtkCellDataToPointData()
                rarv_pointdata.SetInputConnection(rarv_thresh.GetOutputPort())
                rarv_pointdata.PassCellDataOn()
                rarv_pointdata.Update()

                rarv_derivative = vtk.vtkCellDerivatives()
                rarv_derivative.SetInputConnection(rarv_pointdata.GetOutputPort())
                rarv_derivative.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, arr)
                rarv_derivative.Update()

                rarv_gradient = ns.vtk_to_numpy(rarv_derivative.GetOutput().GetCellData().GetArray('ScalarGradient'))
                rarv_gradient_mag = np.linalg.norm(rarv_gradient, axis=1)
                rarv_gradient_mag_vtk = ns.numpy_to_vtk(rarv_gradient_mag)
                rarv_gradient_mag_vtk.SetName(f'RARV_Gradient')
                rarv_thresh.GetOutput().GetCellData().AddArray(rarv_gradient_mag_vtk)

                # writer_rarv = vtk.vtkXMLUnstructuredGridWriter()
                # writer_rarv.SetInputConnection(rarv_thresh.GetOutputPort())
                # writer_rarv.SetFileName(f'rarv_{t}.vtu')
                # writer_rarv.Update()

                ra_thresh = vtk.vtkThreshold()
                ra_thresh.SetInputConnection(rarv_thresh.GetOutputPort())
                ra_thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, arr)
                ra_thresh.ThresholdBetween(channels[0][0]-0.5, channels[0][0]+0.5)
                ra_thresh.Update()

                ra_region = vtk.vtkConnectivityFilter()
                ra_region.SetInputConnection(ra_thresh.GetOutputPort())
                ra_region.SetExtractionModeToLargestRegion()
                ra_region.Update()

                ra_centers = vtk.vtkCellCenters()
                ra_centers.SetInputConnection(ra_region.GetOutputPort())
                ra_centers.Update()

                ra_points = ns.vtk_to_numpy(ra_centers.GetOutput().GetPoints().GetData())
                ra_cog = np.mean(ra_points, axis=0)

                ra_gradient_thresh = vtk.vtkThreshold()
                ra_gradient_thresh.SetInputConnection(ra_region.GetOutputPort())
                ra_gradient_thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, 'RARV_Gradient')
                ra_gradient_thresh.ThresholdByUpper(0.1)
                ra_gradient_thresh.Update()

                ra_gradient_connectivity = vtk.vtkConnectivityFilter()
                ra_gradient_connectivity.SetInputConnection(ra_gradient_thresh.GetOutputPort())
                ra_gradient_connectivity.SetExtractionModeToLargestRegion()
                ra_gradient_connectivity.Update()

                ra_gradient_centers = vtk.vtkCellCenters()
                ra_gradient_centers.SetInputConnection(ra_gradient_connectivity.GetOutputPort())
                ra_gradient_centers.Update()

                boundary_points = ns.vtk_to_numpy(ra_gradient_centers.GetOutput().GetPoints().GetData())
                boundary_cog = np.mean(boundary_points, axis=0)

                ra_surface = vtk.vtkDataSetSurfaceFilter()
                ra_surface.SetInputConnection(ra_region.GetOutputPort())
                ra_surface.Update()

                tree = vtk.vtkOBBTree()
                tree.SetDataSet(ra_surface.GetOutput())
                tree.BuildLocator()    

                # Fit line to boundary
                uu, dd, vv = np.linalg.svd(boundary_points - boundary_cog)

                longit_axis = (ra_cog - boundary_cog) / np.linalg.norm(ra_cog - boundary_cog)
                # Remove component parallel to boundary line
                longit_axis = longit_axis - np.dot(vv[0], longit_axis)*vv[0]
                longit_axis = longit_axis / np.linalg.norm(longit_axis)
                longit_proj = np.dot(ra_points - boundary_cog, longit_axis)
                point2 = ra_points[np.argmax(longit_proj)]

                longit_line = vtk.vtkLineSource()
                longit_line.SetPoint1(boundary_cog)
                longit_line.SetPoint2(boundary_cog + longit_axis*np.max(longit_proj))
                longit_line.Update()

                append = vtk.vtkAppendPolyData()
                append.AddInputConnection(longit_line.GetOutputPort())

                # writer = vtk.vtkXMLUnstructuredGridWriter()
                # writer.SetInputConnection(ra_region.GetOutputPort())
                # writer.SetFileName('ra_region.vtu')
                # writer.Update()

                # line_writer = vtk.vtkXMLPolyDataWriter()
                # line_writer.SetInputConnection(longit_line.GetOutputPort())
                # line_writer.SetFileName('longit_line.vtp')
                # line_writer.Update()    
                cnt = 0
                longit_dist = 2.0
                diameters = []
                while longit_dist < np.max(longit_proj):
                    longit_center = boundary_cog + longit_axis * longit_dist
                    perpendicular_axis = (longit_center - vv[0]*1000)
                    perpendicular_axis = perpendicular_axis - np.dot(perpendicular_axis, longit_axis)*longit_axis
                    perpendicular_axis = perpendicular_axis / np.linalg.norm(perpendicular_axis)
                    perpendicular_proj = np.dot(ra_points - longit_center, perpendicular_axis)
                    perpendicular_longit_proj = np.dot(ra_points - longit_center, longit_axis)
                    # cross_ra_idx = np.where(np.abs(perpendicular_longit_proj)<0.5)[0]
                    # point1 = ra_points[cross_ra_idx][np.argmin(perpendicular_proj[cross_ra_idx])]
                    # point2 = ra_points[cross_ra_idx][np.argmax(perpendicular_proj[cross_ra_idx])]

                    intersect_points = vtk.vtkPoints()
                    intersect_points.SetNumberOfPoints(5)
                    intersect_list = vtk.vtkIdList()
                    tree.IntersectWithLine(longit_center - perpendicular_axis*1000, longit_center + perpendicular_axis*1000, intersect_points, intersect_list)        
                    # print(t, intersect_points.GetNumberOfPoints())
                    intersect_points_coors = ns.vtk_to_numpy(intersect_points.GetData())
                    intersect_points_proj = np.dot(intersect_points_coors-longit_center, perpendicular_axis)
                    point1 = intersect_points_coors[np.argmin(intersect_points_proj)]
                    point2 = intersect_points_coors[np.argmax(intersect_points_proj)]
                    longit_line = vtk.vtkLineSource()
                    longit_line.SetPoint1(point1)
                    longit_line.SetPoint2(point2)
                    longit_line.Update()
                    diameters.append(np.linalg.norm(point2-point1))
                    append.AddInputConnection(longit_line.GetOutputPort())
                    cnt += 1
                    longit_dist += 2.0

                volumes[t] = np.sum(np.array(diameters)**2.0 * np.pi / 4.0 * 2.0)
                results[0][f'RA_simpson_{t}'][(i-start)*2+ii] = volumes[t]/1000.0

    except Exception as e:
        logging.warning(f'Exception caught at {e}')
        pass
    # print(t, volumes[t])
    # writer = vtk.vtkXMLPolyDataWriter()
    # writer.SetInputConnection(append.GetOutputPort())
    # writer.SetFileName(f'longit_lines_{t}.vtp')
    # writer.Update()

for chamber, result in zip(chambers, results):
    results_df = pd.DataFrame(result)
    results_df.to_csv(f'{chamber}_simpson_{version}_{start}_{end}.csv', index=False)
# # %%
# f, ax = plt.subplots()
# ax.plot(volumes/1000.0, label='COG')
# ax.plot(volumes_ortho/1000.0, label='ORTHO')
# ax.set_ylabel('RA volume (Simpson''s) (ml)')
# ax.set_xlim([0, 50])
# ax.set_xlabel('Frames')
# ax.legend()
# f.savefig('simpson_ra.png')

# %%
