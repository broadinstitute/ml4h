"""Methods for integration of interactive dicom plots within notebooks.

TODO:
* incorporate gs://ml4cvd/projects/fake_mris/. Per Sam "These HD5s have
  information from the DICOMs and have applied processing to them already.  So
  keys like `systole_frame_b9` or `diastole_frame_b2` have the same array data
  that is in dcm.pixel_array for short axis slices. For long axis the 2D slices
  have been merged into 3D tensors so the keys `cine_segmented_lax_2ch`
  `cine_segmented_lax_3ch` and `cine_segmented_lax_4ch` map to arrays of
  dimension (256, 256, 50) where each of the 50 z slices correspond to a
  dcm.pixel_array."
"""

import collections
import os
import tempfile
import zipfile

from IPython.display import display
from IPython.display import HTML
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import tensorflow as tf

DEFAULT_MRI_FOLDERS = {
    'fake': [
        'gs://ml4cvd/projects/fake_mris/',
        'gs://ml4cvd/projects/fake_brain_mris/',
        ],
    'ukb': [
        'gs://bulkml4cvd/brainmri/t1_structural_07_26_2019/zipped_t1_dicoms/',
        'gs://ml4cvd/data/mris/cardiac/'
        ]
}


def choose_mri(sample_id, gcs_folder=None):
  """Render widget to choose the MRI to plot.

  Args:
    sample_id: The id of the sample to retrieve.
    gcs_folder: The local or Cloud Storage folder under which the files reside.

  Returns:
    ipywidget or HTML upon error.
  """
  if gcs_folder is None:
    if 'fake' in str(sample_id):
      gcs_folders = DEFAULT_MRI_FOLDERS['fake']
    else:
      gcs_folders = DEFAULT_MRI_FOLDERS['ukb']
  else:
    gcs_folders = [gcs_folder]

  sample_mris = []
  sample_mri_glob = str(sample_id) + '_*.zip'
  try:
    for folder in gcs_folders:
      sample_mris.extend(
          tf.io.gfile.glob(pattern=os.path.join(folder, sample_mri_glob)))
  except (tf.errors.NotFoundError, tf.errors.PermissionDeniedError) as e:
    return HTML('''
    <div class="alert alert-block alert-danger">
    <b>Warning:</b> MRI not available for sample {}:
    <hr><p><pre>{}</pre></p>
    </div>
    '''.format(sample_id, e.message))

  if not sample_mris:
    return HTML('''
    <div class="alert alert-block alert-danger">
    <b>Warning:</b> MRI not available for sample {}
    </div>
    '''.format(sample_id))

  mri_chooser = widgets.Dropdown(
      options=sample_mris,
      value=sample_mris[0],
      description='Choose an MRI to visualize for sample {}:'.format(sample_id),
      style={'description_width': 'initial'},
      layout=widgets.Layout(width='800px')
  )
  file_controls_ui = widgets.VBox(
      [widgets.HTML('<h3>File controls</h3>'), mri_chooser],
      layout=widgets.Layout(width='auto', border='solid 1px grey'))
  file_controls_output = widgets.interactive_output(
      choose_mri_series, {'sample_mri': mri_chooser})
  display(file_controls_ui, file_controls_output)


def choose_mri_series(sample_mri):
  """Render widgets and interactive plots for MRIs.

  Args:
    sample_mri: The local or Cloud Storage path to the MRI file.

  Returns:
    ipywidget or HTML upon error.
  """
  with tempfile.TemporaryDirectory() as tmpdirname:
    local_path = os.path.join(tmpdirname, os.path.basename(sample_mri))
    try:
      tf.io.gfile.copy(src=sample_mri, dst=local_path)
      with zipfile.ZipFile(local_path, 'r') as zip_ref:
        zip_ref.extractall(tmpdirname)
    except (tf.errors.NotFoundError, tf.errors.PermissionDeniedError) as e:
      return HTML('''
      <div class="alert alert-block alert-danger">
      <b>Warning:</b> Cardiac MRI not available for sample {}:
      <hr><p><pre>{}</pre></p>
      </div>
      '''.format(os.path.basename(sample_mri), e.message))

    filtered_dicoms = collections.defaultdict(dict)
    for dcm_file in os.listdir(tmpdirname):
      if not dcm_file.endswith('.dcm'):
        continue
      dcm = pydicom.read_file(os.path.join(tmpdirname, dcm_file))
      if 'brain' in sample_mri:
        if dcm.SeriesNumber in [5, 11]:
          continue
        filtered_dicoms[
            dcm.SeriesDescription.lower()][int(dcm.InstanceNumber) - 1] = dcm
      else:
        filtered_dicoms[
            dcm.SeriesDescription.lower()][int(dcm.InstanceNumber) - 1] = dcm

  if not filtered_dicoms:
    print('\n\nNo series available in MRI for sample ',
          os.path.basename(sample_mri),
          '\n\nTry a different MRI.')
    return None

  # Convert dict of dicts to dict of ordered lists.
  dicoms = {}
  max_num_instances = 0
  print(os.path.basename(sample_mri) + ' contains: ')
  for series in filtered_dicoms.keys():
    dicoms[series] = [None] * (max(filtered_dicoms[series]) + 1)
    for idx, val in filtered_dicoms[series].items():
      if dicoms[series][idx] is not None:
        # Notice invalid input, but don't throw an error.
        print('WARNING: Duplicate instances: ' + str(idx))
      dicoms[series][idx] = val
    print('\t{} with {} instances.'.format(series, len(dicoms[series])))
    if max_num_instances < len(dicoms[series]):
      max_num_instances = len(dicoms[series])

  default_series_value = sorted(list(dicoms.keys()))[0]
  default_instance_value = int(len(dicoms[default_series_value]) / 2)

  series_name_chooser = widgets.Dropdown(
      options=sorted(list(dicoms.keys())),
      value=default_series_value,
      description='Choose the MRI series to visualize:',
      style={'description_width': 'initial'},
      layout=widgets.Layout(width='800px')
  )
  # Slide through dicom images using a slide bar.
  instance_chooser = widgets.IntSlider(
      continuous_update=True,
      value=default_instance_value,
      min=1,
      max=max_num_instances,
      description='Image instance to display '
      + '(click on slider, then use left/right arrows):',
      style={'description_width': 'initial'},
      layout=series_name_chooser.layout)
  transpose_chooser = widgets.Checkbox(
      description='Whether to transpose the image.',
      style={'description_width': 'initial'},
      layout=series_name_chooser.layout)
  fig_width_chooser = widgets.IntSlider(
      continuous_update=False,
      value=18,
      min=8,
      max=24,
      description='Width of figure (height will be computed using input data):',
      style={'description_width': 'initial'},
      layout=series_name_chooser.layout)
  viz_controls_ui = widgets.VBox(
      [widgets.HTML('<h3>Visualization controls</h3>'), series_name_chooser,
       instance_chooser, transpose_chooser, fig_width_chooser],
      layout=widgets.Layout(width='auto', border='solid 1px grey'))
  viz_controls_output = widgets.interactive_output(
      dicom_animation,
      {'dicoms': widgets.fixed(dicoms),
       'series_name': series_name_chooser,
       'instance': instance_chooser,
       'fig_width': fig_width_chooser,
       'transpose': transpose_chooser,
       'title_prefix': widgets.fixed(os.path.basename(sample_mri))})
  display(viz_controls_ui, viz_controls_output)


def dicom_animation(dicoms, series_name, instance, transpose, fig_width,
                    title_prefix=''):
  """Render one frame of a dicom animation.

  Args:
    dicoms: the dictionary DICOM series and instances lists
    series_name: the name of the series to be displayed
    instance: the particular instance to display
    transpose: whether or not to transpose the image
    fig_width: the desired width of the figure, note that height computed as
      the proportion of the width based on the data to be plotted
    title_prefix: text to display as the initial portion of the plot title
  """
  if len(dicoms[series_name]) < instance:
    dcm = dicoms[series_name][-1]
    print('Instance {} not available for {}, using final instance'
          ' instead.'.format(str(instance), series_name))
  else:
    dcm = dicoms[series_name][instance - 1]
    if instance != dcm.InstanceNumber:
      # Notice invalid input, but don't throw an error.
      print('WARNING: Instance parameter {} and dicom instance number {} do not'
            ' match'.format(str(instance), str(dcm.InstanceNumber)))

  if transpose:
    height = dcm.pixel_array.T.shape[0]
    width = dcm.pixel_array.T.shape[1]
  else:
    height = dcm.pixel_array.shape[0]
    width = dcm.pixel_array.shape[1]

  fig_height = int(np.ceil(fig_width * (height/width)))

  _, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='beige')
  ax.imshow(dcm.pixel_array.T if transpose else dcm.pixel_array,
            cmap='gray',
            vmin=np.min(dcm.pixel_array),
            vmax=np.max(dcm.pixel_array))
  ax.set_title(title_prefix
               + ', Series: ' + dcm.SeriesDescription
               + ', Instance: ' + str(dcm.InstanceNumber)
               + ', Transpose: ' + str(transpose)
               + ', Figure size:' + str(fig_width) + ' ' + str(fig_height),
               fontsize=fig_width)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
