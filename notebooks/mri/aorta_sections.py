# %%
import vtk
from vtk.util import numpy_support as ns

model_sections = vtk.vtkXMLPolyDataReader()
model_sections.SetFileName('/home/pdiachil/projects/aorta/1000107_model_sections.vtp')
model_sections.Update()

model_cl = vtk.vtkXMLPolyDataReader()
model_cl.SetFileName('/home/pdiachil/projects/aorta/1000107_model_smooth_cl_resample.vtp')
model_cl.Update()

segmentation_sections = vtk.vtkXMLPolyDataReader()
segmentation_sections.SetFileName('/home/pdiachil/projects/aorta/1000107_segmentation_sections.vtp')
segmentation_sections.Update()

segmentation_cl = vtk.vtkXMLPolyDataReader()
segmentation_cl.SetFileName('/home/pdiachil/projects/aorta/1000107_segmentation_cl_resample.vtp')
segmentation_cl.Update()

# %%
print(model_sections.GetOutput().GetCellData())

model_areas = ns.vtk_to_numpy(model_sections.GetOutput().GetCellData().GetArray('CenterlineSectionArea'))
segmentation_areas = ns.vtk_to_numpy(segmentation_sections.GetOutput().GetCellData().GetArray('CenterlineSectionArea'))


model_shapes = ns.vtk_to_numpy(model_sections.GetOutput().GetCellData().GetArray('CenterlineSectionShape'))
segmentation_shapes = ns.vtk_to_numpy(segmentation_sections.GetOutput().GetCellData().GetArray('CenterlineSectionShape'))

model_max_size = ns.vtk_to_numpy(model_sections.GetOutput().GetCellData().GetArray('CenterlineSectionMaxSize'))
segmentation_max_size = ns.vtk_to_numpy(segmentation_sections.GetOutput().GetCellData().GetArray('CenterlineSectionMaxSize'))
model_min_size = ns.vtk_to_numpy(model_sections.GetOutput().GetCellData().GetArray('CenterlineSectionMinSize'))
segmentation_min_size = ns.vtk_to_numpy(segmentation_sections.GetOutput().GetCellData().GetArray('CenterlineSectionMinSize'))


model_cl_radius = ns.vtk_to_numpy(model_cl.GetOutput().GetPointData().GetArray('MaximumInscribedSphereRadius'))
segmentation_cl_radius = ns.vtk_to_numpy(segmentation_cl.GetOutput().GetPointData().GetArray('MaximumInscribedSphereRadius'))
# %%
import matplotlib.pyplot as plt
import numpy as np
xticks = list(range(0, 50, 10))

xticklabels = [str(xtick) for xtick in xticks]
xticklabels[0] += '\nroot'
xticklabels[-1] += '\niliacs'

mask_model = np.logical_or(model_shapes > 0.90, model_shapes < 0.65)
mask_segmentation = np.logical_or(segmentation_shapes > 0.95, segmentation_shapes < 0.5)

f, ax = plt.subplots()
f.set_size_inches(4, 3)
#ax.plot(np.ma.masked_array(model_areas, mask=mask_model) / 100.0, linewidth=2, color='gray', label='model')
#ax.plot(np.ma.masked_array(segmentation_areas, mask=mask_segmentation) / 100.0, linewidth=2, color='black', label='segmentation')
ax.plot(model_areas / 100.0, linewidth=2, color='gray', label='model')
ax.plot(segmentation_areas / 100.0, linewidth=2, color='black', label='segmentation')

ax.legend()
ax.set_ylabel('cross-section area (cm$^2$)')
ax.set_xlabel('aorta arclength (cm)')
ax.set_xlim([0, max(len(model_areas)-1, len(segmentation_areas)-1)])
ax.set_xticklabels(xticklabels)
plt.tight_layout()
f.savefig('1000107_areas.png', dpi=500)
# %%
f, ax = plt.subplots()
f.set_size_inches(4, 3)
ax.plot(model_max_size / 20.0, linewidth=2, color='gray', label='model (section)')
ax.plot(segmentation_max_size / 20.0, linewidth=2, color='black', label='segmentation (section)')
ax.plot(model_cl_radius / 10.0, linewidth=2, color='gray', linestyle='--', label='model (sphere)')
ax.plot(segmentation_cl_radius / 10.0, linewidth=2, color='black', linestyle='--', label='segmentation (sphere)')
ax.legend()
ax.set_ylabel('max radius (cm)')
ax.set_xlabel('aorta arclength (cm)')
ax.set_xlim([0, max(len(model_areas)-1, len(segmentation_areas)-1)])
ax.set_xticklabels(xticklabels)
f.savefig('1000107_max_radius.png', dpi=500)
# %%
f, ax = plt.subplots()
f.set_size_inches(4, 3)
ax.plot(0.5*(model_max_size+model_min_size) / 20.0, linewidth=2, color='gray', label='model (section)')
ax.plot(0.5*(segmentation_max_size+segmentation_min_size) / 20.0, linewidth=2, color='black', label='segmentation (section)')
ax.plot(model_cl_radius / 10.0, linewidth=2, color='gray', linestyle='--', label='model (sphere)')
ax.plot(segmentation_cl_radius / 10.0, linewidth=2, color='black', linestyle='--', label='segmentation (sphere)')
ax.legend()
ax.set_ylabel('mean radius (cm)')
ax.set_xlabel('aorta arclength (cm)')
ax.set_xlim([0, max(len(model_areas)-1, len(segmentation_areas)-1)])
ax.set_xticklabels(xticklabels)
f.savefig('1000107_mean_radius.png', dpi=500)
# %%
f, ax = plt.subplots()
f.set_size_inches(4, 3)
ax.plot(model_shapes, linewidth=2, color='gray', label='model')
ax.plot(segmentation_shapes, linewidth=2, color='black', label='segmentation')
ax.legend()
ax.set_ylabel('max diameter / min diameter')
ax.set_xlabel('aorta arclength (cm)')
ax.set_xlim([0, max(len(model_areas)-1, len(segmentation_areas)-1)])
ax.set_xticklabels(xticklabels)
f.savefig('1000107_eccentricity.png', dpi=500)
# %%
