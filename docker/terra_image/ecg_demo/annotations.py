"""Methods for collecting and submitting annotations within notebooks."""
import os
import tempfile

from IPython.display import HTML
from IPython.display import SVG
import numpy as np

import ipywidgets as widgets


def get_df_sample(sample_info, sample_id):
    df_sample = sample_info[sample_info['sample_id']==str(sample_id)]
    if 0 == df_sample.shape[0]: df_sample = sample_info.query('sample_id == ' + str(sample_id))
    
    return df_sample

def display_annotation_collector(sample_info, sample_id):
  """Method to create a gui (set of widgets) through which the user can create an annotation

  Args:
    sample_id: The selected sample for which the values will be displayed.
    sample_info: dataframe containing all the samples and data

  Returns:
    key: Jupyter widget containing (in .value) the key (column in sample_info) that was selected by the user
    keyvalue: Jupyter widget containing the value specified by key.value for the sample (defined by sample_id)
    comment: The annotator's comment text
  """

  df_sample = get_df_sample(sample_info, sample_id)
  
  # show the sample ID for this annotation
  sample = widgets.HTML(value = f"For sample <b>{sample_id}</b>")

  # allow the user to pick a key about which to comment
  key = widgets.Dropdown(
      options=sample_info.keys(),
      description='Key:',
      disabled=False)

  # return the sample's value for that key
  valuelabel = widgets.Label(value='Value: ')
  keyvalue = widgets.Label(value=str(df_sample[key.value].iloc[0]))
  
  box1 = widgets.HBox([key, valuelabel, keyvalue], 
                      layout=widgets.Layout(width='50%'))

  # allow the user to leave a text comment
  comment = widgets.Textarea(
      value='',
      placeholder='Type your comment here',
      description=f'Comment:',
      disabled=False,
      layout=widgets.Layout(width='80%', height='50px'),
      style={'description_width': 'initial'}
  )

  
  
  def handle_key_change(change):
      keyvalue.value = str(df_sample[key.value].iloc[0]) 
  
  key.observe(handle_key_change, names='value')

  
  # display everything
  display(sample, box1, comment) 
  
  return key, keyvalue, comment

def format_annotation(sample_id, annotation_data):
  # pull out values from output
  key = annotation_data[0].value
  keyvalue = annotation_data[1].value
  comment = annotation_data[2].value

  # Programmatically get the identity of the person running this Terra notebook.
  USER = os.getenv('OWNER_EMAIL')

  # # Also support AI Platform Notebooks.
  # if USER is None:
  #   ai_platform_hostname = !hostname
  #   USER = ai_platform_hostname[0] # By convention, we prefix the hostname with our username.

  # check whether the value is string or numeric
  try:
      if keyvalue == nan: # @Nicole: this may not be how you want to deal with 'nan' values (e.g. in past_tobacco_smoking)
          raise Exception() # this will make nan values return as strings rather than numerics
      value_numeric = float(keyvalue) # this will fail if the value is text
      value_string = 'None'
  except:
      value_numeric = 'None'
      value_string = keyvalue

  params = {
      'sample_id': str(sample_id),
      'annotator': USER,
      'key': key,
      'value_numeric': value_numeric,
      'value_string': value_string,
      'comment': comment
  }

  return params