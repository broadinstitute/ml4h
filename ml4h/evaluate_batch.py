import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.python.data.ops import dataset_ops
from typing import List, Set, Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from ml4h.tensor_generators import TensorGenerator


class BatchMetricsLogger(Callback):
    def __init__(self, metrics):
        super(BatchMetricsLogger, self).__init__()
        self.metrics = metrics
        self.storage = []
    
    def on_test_batch_end(self, batch, logs=None):
        self.storage.append(logs)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, files, tensor_generator: TensorGenerator):
        self.files = files
        self.generator = tensor_generator
        self.offset = 0
        self.limit = len(files)
    
    def __len__(self):
        return self.limit
    
    def __getitem__(self, index):
        x, y, _, b = next(self.generator)
        self.offset += 1
        return x, y, [None]
    
    def on_epoch_end(self):
        self.offset = 0


def evaluate_collect_metrics(model: tf.keras.models.Model, 
        data_generator: list, 
        metrics = None):
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
    data_generator.on_epoch_end()
    try:
        eval = model.evaluate(data_generator, callbacks=[logger], verbose=2)
    except Exception as e:
        raise Exception(f'Failed to evaluate model:\n{e}')

    eval_batch = pd.DataFrame(logger.storage, index = np.arange(len(logger.storage)))
    # Workaround for old Keras
    eval = {out: eval[i] for i, out in enumerate(model.metrics_names)}
    eval = pd.DataFrame(eval,index=[0])
    return eval, eval_batch


def evaluate_models(models: List[tf.keras.models.Model], 
        data_generators: list, 
        metrics: List[Callable] = None,
        channel_order: str = 'channels_last'):
    # `metrics` must be either a list of callable metrics functions
    # or a prepared dictionary of metrics
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
        is_ml4h = False
        files = []
        if isinstance(g, dataset_ops.DatasetV2):
            if hasattr(g, '_batch_size'):
                if g._batch_size.numpy() != 1:
                    raise ValueError("Batch size must be set to 1")
        elif isinstance(g, TensorGenerator):
            files = np.array([p.paths for p in g.path_iters]).flatten()
            g = DataGenerator(files, g)
            is_ml4h = True
        
        eval, eval_batch = evaluate_collect_metrics(m, g, metrics_dict)

        if is_ml4h:
            eval_batch.index = files

        ret = {
            'model': model_count,
            'eval': eval,
            'eval_batch': eval_batch,
        }
        model_count += 1
        evaluations.append(ret)
    return evaluations

