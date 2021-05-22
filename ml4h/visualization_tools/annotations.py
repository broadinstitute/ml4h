"""Methods for capturing and displaying annotations within notebooks."""

import os
import socket
import pathlib
from typing import Any, Dict, Union, List, Callable, Optional, Type, Sequence

from ipyannotations.images.canvases.point import PointAnnotationCanvas
from ipyannotations.images.canvases._abstract import AbstractAnnotationCanvas

from IPython.display import display
from IPython.display import HTML
import pandas as pd
import ipywidgets as widgets
import traitlets
from ml4h.visualization_tools.annotation_storage import AnnotationStorage
from ml4h.visualization_tools.annotation_storage import TransientAnnotationStorage

DEFAULT_ANNOTATION_STORAGE = TransientAnnotationStorage()


def _get_df_sample(sample_info: pd.DataFrame, sample_id: Union[int, str]) ->  pd.DataFrame:
  """Return a dataframe containing only the row for the indicated sample_id."""
  df_sample = sample_info[sample_info['sample_id'] == str(sample_id)]
  if df_sample.shape[0] == 0: df_sample = sample_info.query('sample_id == ' + str(sample_id))
  return df_sample


def display_annotation_collector(
    sample_info: pd.DataFrame, sample_id: Union[int, str],
    annotation_storage: AnnotationStorage = DEFAULT_ANNOTATION_STORAGE,
    custom_annotation_key: str = None,
) -> None:
  """Method to create a gui (set of widgets) through which the user can create an annotation and submit it to storage.

  Args:
    sample_info: Dataframe containing tabular data for all the samples.
    sample_id: The selected sample for which the values will be displayed.
    annotation_storage: An instance of AnnotationStorage.
    custom_annotation_key: The key for an annotation of data other than the tabular fields.
  """

  df_sample = _get_df_sample(sample_info, sample_id)
  if df_sample.shape[0] == 0:
    display(
        HTML(f'''<div class="alert alert-block alert-danger">
    <b>Warning:</b> Sample {sample_id} not present in sample_info DataFrame.
    </div>'''),
    )
    return

  # Show the sample ID for this annotation.
  sample = widgets.HTML(value=f'For sample <b>{sample_id}</b>')

  # Allow the user to pick a key about which to comment.
  annotation_keys = []
  if custom_annotation_key:
    annotation_keys.append(custom_annotation_key)
  annotation_keys.extend(sorted(sample_info.keys()))
  key = widgets.Dropdown(
      options=annotation_keys,
      description='Key:',
      disabled=False,
  )

  # Return the sample's value for that key, when applicable.
  valuelabel = widgets.Label(value='Value: ')
  keyvalue = widgets.Label(value=None if custom_annotation_key else str(df_sample[key.value].iloc[0]))

  box1 = widgets.HBox(
      [key, valuelabel, keyvalue],
      layout=widgets.Layout(width='50%'),
  )

  # Have keyvalue auto update depending on the selected key.
  def handle_key_change(change):
    if change['new'] == custom_annotation_key:
      keyvalue.value = None
    else:
      keyvalue.value = str(df_sample[key.value].iloc[0])

  key.observe(handle_key_change, names='value')

  # Allow the user to leave a text comment as the main value of the annotation.
  comment = widgets.Textarea(
      value='',
      placeholder='Type your comment here',
      description='Comment:',
      disabled=False,
      layout=widgets.Layout(width='80%', height='50px'),
      style={'description_width': 'initial'},
  )

  # Configure the submission button.
  submit_button = widgets.Button(description='Submit annotation', button_style='success')
  output = widgets.Output()

  def cb_on_button_clicked(b):
    params = _format_annotation(sample_id=sample_id, key=key.value, keyvalue=keyvalue.value, comment=comment.value)
    try:
      success = annotation_storage.submit_annotation(
          sample_id=params['sample_id'],
          annotator=params['annotator'],
          key=params['key'],
          value_numeric=params['value_numeric'],
          value_string=params['value_string'],
          comment=params['comment'],
      )
    except Exception as e:  # pylint: disable=broad-except
      display(
          HTML(f'''<div class="alert alert-block alert-danger">
      <b>Warning:</b> Unable to store annotation.
      <hr><p><pre>{e}</pre></p>
      </div>'''),
      )
      return
    with output:
      if success:  # Show the information that was submitted.
        display(
            HTML(f'''<div class="alert alert-block alert-info">
        Submission successful\n[{annotation_storage.describe()}]
        </div>'''),
        )
        display(annotation_storage.view_recent_submissions(1))
      else:
        display(
            HTML('''<div class="alert alert-block alert-warning">
        Annotation not submitted. Please try again.
        </div>'''),
        )

  submit_button.on_click(cb_on_button_clicked)

  # Display all the widgets.
  display(sample, box1, comment, submit_button, output)


def _format_annotation(
    sample_id: Union[int, str], key: str, keyvalue: Union[int, float, str], comment: str,
) -> Dict[str, Any]:
  """Helper method to clean and reshape info from the widgets and the environment into a dictionary representing the annotation."""
  # Programmatically get the identity of the person running this Terra notebook.
  current_user = os.getenv('OWNER_EMAIL')
  # Also support other environments such as AI Platform Notebooks.
  if current_user is None:
    current_user = socket.gethostname()  # By convention, we prefix the hostname with our username.

  value_numeric = None
  value_string = None
  # Check whether the value is string or numeric.
  if keyvalue is not None:
    try:
      value_numeric = float(keyvalue)  # this will fail if the value is text
      value_string = None
    except ValueError:
      value_numeric = None
      value_string = keyvalue

  # Format into a dictionary.
  params = {
      'sample_id': str(sample_id),
      'annotator': current_user,
      'key': key,
      'value_numeric': value_numeric,
      'value_string': value_string,
      'comment': comment,
  }

  return params


class T1MAPAnnotator(widgets.VBox):
    """A generic image annotation widget.

    Parameters
    ----------
    canvas : AbstractAnnotationCanvas
        An annotation canvas that implements displaying & annotating
        images.
    options : List[str], optional
        The list of classes you'd like to annotate.
    data_postprocessor : Optional[Callable[[List[dict]], Any]], optional
        A function that transforms the annotation data. By default None.
    """

    options = traitlets.List(
        list(), allow_none=False, help="The possible classes"
    )
    options.__doc__ = """The possible classes"""

    CanvasClass: Type[AbstractAnnotationCanvas]

    def __init__(
        self,
        canvas_size=(700, 500),
        options: Sequence[str] = (),
        data_postprocessor: Optional[Callable[[List[dict]], Any]] = None,
        # **kwargs,
    ):
        """Create an annotation widget for images."""
        self.canvas = self.CanvasClass(canvas_size, classes=options)
        self.data_postprocessor = data_postprocessor

        # controls for the data entry:
        data_controls = []
        self.options = options

        self.class_selector = widgets.Dropdown(
            description="Class:",
            options=options,
            layout=widgets.Layout(flex="1 1 auto"),
        )
        widgets.link((self, "options"), (self.class_selector, "options"))
        widgets.link(
            (self.class_selector, "value"), (self.canvas, "current_class")
        )
        data_controls.append(self.class_selector)

        extra_checkboxes = []
        checkboxes_layout = widgets.Layout(
            # width="auto",
            min_width="80px",
            flex="1 1 auto",
            max_width="120px",
        )

        self.lv_fw_checkbox = widgets.Checkbox(
          value=False,
          description='bad LV free wall',
          disabled=False,
          indent=False,
          layout=checkboxes_layout
        )
        
        self.iv_checkbox = widgets.Checkbox(
          value=False,
          description='bad IV septum',
          disabled=False,
          indent=False,
          layout=checkboxes_layout
        )

        self.lv_checkbox = widgets.Checkbox(
          value=False,
          description='bad LV blood pool',
          disabled=False,
          indent=False,
          layout=checkboxes_layout
        )

        self.rv_checkbox = widgets.Checkbox(
          value=False,
          description='bad RV blood pool',
          disabled=False,
          indent=False,
          layout=checkboxes_layout
        )

        extra_checkboxes = [self.lv_fw_checkbox,
                            self.iv_checkbox,
                            self.lv_checkbox,
                            self.rv_checkbox,
                            ]      

        extra_checkboxes = widgets.HBox(
            extra_checkboxes,
            layout={
                "align_items": "stretch",
                "justify_content": "flex-end",
                "flex_flow": "row wrap",
            },
        )

        self.good_button = widgets.Button(
            description="Submit good review",
            icon="tick",
            button_style="success",
            layout=checkboxes_layout,
        )
        self.good_button.on_click(self.good_submit)

        self.bad_button = widgets.Button(
            description="Submit bad review",
            icon="tick",
            button_style="danger",
            layout=checkboxes_layout,
        )
        self.bad_button.on_click(self.bad_submit)

        self.data_controls = widgets.VBox(
            children=(
                widgets.HTML("Review settings"),                
                extra_checkboxes,
                widgets.HBox(
                    (self.good_button, self.bad_button),
                    layout={"justify_content": "flex-end"},
                ),
            ),
            layout={"flex": "1 1 auto", "max_width": "600px"},
        )

        # controls for the visualisation of the data:
        viz_controls = []
        if hasattr(self.canvas, "opacity"):
            self.opacity_slider = widgets.FloatSlider(
                description="Opacity", value=1, min=0, max=1, step=0.025
            )
            widgets.link(
                (self.opacity_slider, "value"), (self.canvas, "opacity")
            )
            viz_controls.append(self.opacity_slider)
        if hasattr(self.canvas, "point_size"):
            self.point_size_slider = widgets.IntSlider(
                description="Point size", value=5, min=1, max=20, step=1
            )
            widgets.link(
                (self.point_size_slider, "value"), (self.canvas, "point_size")
            )
            viz_controls.append(self.point_size_slider)
        self.brightness_slider = widgets.FloatLogSlider(
            description="Brightness", value=1, min=-1, max=1, step=0.0001
        )
        widgets.link(
            (self.brightness_slider, "value"),
            (self.canvas, "image_brightness"),
        )
        viz_controls.append(self.brightness_slider)
        self.contrast_slider = widgets.FloatLogSlider(
            description="Contrast", value=1, min=-1, max=1, step=0.0001
        )
        widgets.link(
            (self.contrast_slider, "value"), (self.canvas, "image_contrast")
        )
        viz_controls.append(self.contrast_slider)

        self.visualisation_controls = widgets.VBox(
            children=(widgets.HTML("Visualization settings"), *viz_controls),
            layout={"flex": "1 1 auto"},
        )

        self.all_controls = widgets.HBox(
            children=(self.visualisation_controls, self.data_controls),
            layout={
                "width": f"{self.canvas.size[0]}px",
                "justify_content": "space-between",
            },
        )

        self.good_submit_callbacks: List[Callable[[Any], None]] = []
        self.bad_submit_callbacks: List[Callable[[Any], None]] = []
        self.undo_callbacks: List[Callable[[], None]] = []
        self.skip_callbacks: List[Callable[[], None]] = []

        super().__init__(layout={"width": f"{self.canvas.size[0]}px"})
        self.children = [self.canvas, self.all_controls]

    def display(self, image: Union[widgets.Image, pathlib.Path]):
        """Clear the annotations and display an image


        Parameters
        ----------
        image : widgets.Image, pathlib.Path, np.ndarray
            The image, or the path to the image.
        """
        self.canvas.clear()
        self.canvas.load_image(image)

    def on_good_submit(self, callback: Callable[[Any], None]):
        """Register a callback to handle data when the user clicks "Submit".

        .. note::
            Callbacks are called in order of registration - first registered,
            first called.

        Parameters
        ----------
        callback : Callable[[Any], None]
            A function that takes in data. Usually, this data is a list of
            dictionaries, but you are able to define data post-processors when
            you create an annotator that get called before this callback is
            called. Any return values are ignored.
        """
        self.good_submit_callbacks.append(callback)

    def on_bad_submit(self, callback: Callable[[Any], None]):
        """Register a callback to handle data when the user clicks "Submit".

        .. note::
            Callbacks are called in order of registration - first registered,
            first called.

        Parameters
        ----------
        callback : Callable[[Any], None]
            A function that takes in data. Usually, this data is a list of
            dictionaries, but you are able to define data post-processors when
            you create an annotator that get called before this callback is
            called. Any return values are ignored.
        """
        self.bad_submit_callbacks.append(callback)

    def good_submit(self, button: Optional[Any] = None):
        """Trigger the "Submit" callbacks.

        This function is called when users click "Submit".

        Parameters
        ----------
        button : optional
            Ignored argument. Supplied when invoked due to a button click.
        """
        for callback in self.good_submit_callbacks:
            callback(self.data)


    def bad_submit(self, button: Optional[Any] = None):
        """Trigger the "Submit" callbacks.

        This function is called when users click "Submit".

        Parameters
        ----------
        button : optional
            Ignored argument. Supplied when invoked due to a button click.
        """
        for callback in self.bad_submit_callbacks:
            callback(self.data)

    def on_undo(self, callback: Callable[[], None]):
        """Register a callback to handle when the user clicks "Undo".

        Note that any callback registered here is only called when the canvas
        is empty - while there are annotations on the canvas, "Undo" actually
        undoes the annotations, until the canvas is empty.

        Parameters
        ----------
        callback : Callable[[], None]
            A function to be called when users press "Undo". This should be
            a function that takes in no arguments; any return values are
            ignored.
        """
        self.undo_callbacks.append(callback)

    def undo(self, button: Optional[Any] = None):
        """Trigger the "Undo" callbacks.

        This function is called when users click "Undo".

        Parameters
        ----------
        button : optional
            Ignored argument. Supplied when invoked due to a button click.
        """
        if self.canvas._undo_queue:
            undo = self.canvas._undo_queue.pop()
            undo()
        else:
            for callback in self.undo_callbacks:
                callback()

    def on_skip(self, callback: Callable[[], None]):
        """Register a callback to handle when the user clicks "Skip".

        Parameters
        ----------
        callback : Callable[[], None]
            The function to be called when the user clicks "Skip". It should
            take no arguments, and any return values are ignored.
        """
        self.skip_callbacks.append(callback)

    def skip(self, button: Optional[Any] = None):
        """Trigger the "Skip" callbacks.

        This function is called when users click "Skip".

        Parameters
        ----------
        button : optional
            Ignored argument. Supplied when invoked due to a button click.
        """
        for callback in self.skip_callbacks:
            callback()

    @property
    def data(self):
        """The annotation data."""
        if self.data_postprocessor is not None:
            return self.data_postprocessor(self.canvas.data)
        else:
            return self.canvas.data


class T1MAPPointAnnotator(T1MAPAnnotator):
    """An annotator for drawing points on an image.

    To add a point, select the class using the dropdown menu, and click
    anywhere on the image. You can undo adding points, and you can adjust the
    point's position using the "Edit" button. To make this easier, you may
    want to adjust the point size using the appropriate slider.

    You can increase or decrease the contrast and brightness  of the image
    using the sliders to make it easier to annotate. Sometimes you need to see
    what's behind already-created annotations, and for this purpose you can
    make them more see-through using the "Opacity" slider.

    Parameters
    ----------
    canvas_size : (int, int), optional
        Size of the annotation canvas in pixels.
    classes : List[str], optional
        The list of classes you want to create annotations for, by default
        None.
    """

    CanvasClass = PointAnnotationCanvas

    @property
    def data(self):
        """
        The annotation data, as List[ Dict ].

        The format is a list of dictionaries, with the following key / value
        combinations:

        +------------------+-------------------------+
        |``'type'``        | ``'point'``             |
        +------------------+-------------------------+
        |``'label'``       | ``<class label>``       |
        +------------------+-------------------------+
        |``'coordinates'`` | ``<xy-tuple>``          |
        +------------------+-------------------------+
        """
        return super().data

    @data.setter
    def data(self, value):
        self.canvas.data = value