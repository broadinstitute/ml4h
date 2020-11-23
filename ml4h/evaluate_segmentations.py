import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.keras import backend as K
from typing import List, Set, Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from ml4h.tensor_generators import TensorGenerator
from tensorflow.python.training import evaluation


class BatchMetricsLogger(Callback):
    def __init__(self, metrics):
        super(BatchMetricsLogger, self).__init__()
        self.metrics = metrics
        self.storage = []
    #
    def on_test_batch_end(self, batch, logs=None):
        self.storage.append(logs)


class DataGenerator(tf.keras.utils.Sequence):
    """Workaround for using infinite ML4H generators with Keras/Tensorflow 
    generators. The ML4H TensorGenerator is very brittle: use at your own
    risk!

    Args:
        files: Files operated on by TensorGenerator.
        tensor_generator: ML4h TensorGenerator instance.
    """
    def __init__(self, files, tensor_generator: TensorGenerator):
        self.files = files
        self.generator = tensor_generator
        self.offset = 0
        self.limit = len(files)
    #
    def __len__(self):
        return self.limit
    #
    def __getitem__(self, index):
        self.offset += 1
        x, y, _, b = next(self.generator)
        # hack because the data has changed on disk for this model
        y = y['output_cine_segmented_sax_slice_jamesp_categorical'][:,:,:,0:12]
        return x,y,[None]
    #
    def on_epoch_end(self):
        self.offset = 0


# Given a TensorFlow model, a dictionary of metrics, and a data generator
# or (x,y) pair, we evaluate the model and return a pair of dictionary for
# per-batch (n=1) metrics and overall mean metrics.
# 
# As a preprocessing step the input `model` must already be re/compiled
# to use the exact metrics in the `metrics` list.
def evaluate_collect_metrics(model: tf.keras.models.Model, 
        data_generator: list, 
        metrics: List[Callable] = None):
    # Reformat metrics list to a metrics dictionary of the form
    # {name: function pointer}.
    if metrics is not None:
        if isinstance(metrics, list):
            metrics_dict = {m.name: m for m in metrics}
        else:
            metrics_dict = metrics
    else:
        metrics_dict = {m.name: m for m in model.metrics}
    # Reload model
    model.compile(optimizer=model.optimizer,
                        loss=model.loss,
                        metrics=list(metrics_dict.values()))
    #
    # Logger for outputs
    logger = BatchMetricsLogger(metrics = metrics_dict)
    try:
        eval = model.evaluate(data_generator, callbacks=[logger])
    except Exception as e:
        raise Exception(f'Failed to evaluate model:\n{e}')
    eval_batch = pd.DataFrame(logger.storage, index = np.arange(len(logger.storage)))
    # Workaround for old Keras
    eval = {out: eval[i] for i, out in enumerate(model.metrics_names)}
    eval = pd.DataFrame(eval,index=[0])
    # Make predictions
    prediction_prob = model.predict(data_generator)
    # Assumes channel_order
    predictions = tf.argmax(prediction_prob, -1)
    # Return eval, eval_batch pair
    return eval, eval_batch, prediction_prob, predictions


def evaluate_segmentation_models(models: List[tf.keras.models.Model], 
        data_generators: list, 
        metrics: List[Callable] = None,
        channel_order: str = 'channels_last'):
    # `metrics` must be either a list of callable metrics functions
    if len(models) != len(data_generators):
        if len(data_generators) != 1:
            raise ValueError("Unequal number of models and data generators. Must be equal in numbers or use only a single data generator for all models.")
        if len(data_generators) == 1:
            data_generators = data_generators * len(models)
    #
    # Reformat metrics list to a metrics dictionary of the form
    # {name: function pointer}.
    if metrics is not None:
        metrics_dict = {m.name: m for m in metrics}
    #
    model_count = 0
    evaluations = list()
    for m,g in zip(models, data_generators):
        if metrics is None:
            metrics_dict = {n.name: n for n in m.metrics}
        #
        # tf.data.Dataset family of generators in TensorFlow 2+
        # Todo: could eventually be extended to include checks for 
        # `dataset._dataset._batch_size` that is used in TensorFlow 1.*
        if isinstance(g, dataset_ops.DatasetV2):
            if hasattr(g, '_batch_size'):
                if g._batch_size.numpy() != 1:
                    raise ValueError("Batch size must be set to 1")
        elif isinstance(g, TensorGenerator):
            files = np.array([p.paths for p in g.path_iters]).flatten()
            print(files)
            g = DataGenerator(files, g)
        else: # yolo
            pass
        # 
        eval, eval_batch, prediction_prob, predictions = evaluate_collect_metrics(m, g, metrics_dict)
        ret = {
            'model': model_count,
            'eval': eval,
            'eval_batch': eval_batch,
            'prediction_prob': prediction_prob,
            'predictions': predictions
        }
        model_count += 1
        evaluations.append(ret)
    return evaluations
        