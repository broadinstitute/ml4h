"""DROID model construction shared by the training and inference recipes.

A training run directory (the parent of ``model/chkp``) records everything needed
to rebuild its architecture: ``model_params.json`` holds the label types, dense-layer
flags and survival tasks, and ``classification_class_label_mapping_per_output.json``
holds the class mapping and order of any classification heads.
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from model_descriptions.echo import create_movinet_classifier, create_regressor_classifier

MODEL_PARAMS_FILE = 'model_params.json'
CLASS_MAPPING_FILE = 'classification_class_label_mapping_per_output.json'
OUTPUT_SCALING_FILE = 'output_scaling.json'


def read_json(path):
    with tf.io.gfile.GFile(path, 'r') as json_file:
        return json.load(json_file)


def run_dir_from_checkpoint(checkpoint_path):
    """``<run>/model/chkp`` -> ``<run>``."""
    return os.path.dirname(os.path.dirname(checkpoint_path.rstrip('/')))


def validate_survival_tasks(survival_tasks):
    """Validate recipe survival-task specifications and supply optional defaults."""
    tasks = survival_tasks or []
    task_names = set()
    required_fields = {'name', 'event_column', 'follow_up_days_column', 'intervals', 'days_window'}
    validated_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            raise TypeError('Each survival task must be a JSON object.')
        missing_fields = required_fields.difference(task)
        if missing_fields:
            raise ValueError(f"Survival task is missing required fields: {sorted(missing_fields)}")
        name = task['name']
        if not isinstance(name, str) or not name:
            raise ValueError('Each survival task needs a non-empty string name.')
        if name in task_names:
            raise ValueError(f"Survival task names must be unique; duplicate name: {name}")
        if int(task['intervals']) <= 0 or float(task['days_window']) <= 0:
            raise ValueError(f"Survival task {name} needs positive intervals and days_window values.")
        prevalent_policy = task.get('prevalent_policy', 'first_interval')
        if prevalent_policy not in {'first_interval', 'exclude'}:
            raise ValueError(
                f"Survival task {name} has invalid prevalent_policy {prevalent_policy!r}; "
                "use 'first_interval' or 'exclude'.",
            )
        validated_task = dict(task)
        validated_task['intervals'] = int(task['intervals'])
        validated_task['days_window'] = float(task['days_window'])
        validated_task['blanking_days'] = float(task.get('blanking_days', 0))
        if (
                not np.isfinite(validated_task['days_window'])
                or validated_task['blanking_days'] < 0
                or not np.isfinite(validated_task['blanking_days'])
        ):
            raise ValueError(f"Survival task {name} needs finite days_window and non-negative blanking_days values.")
        validated_task['prevalent_policy'] = prevalent_policy
        validated_tasks.append(validated_task)
        task_names.add(name)
    return validated_tasks


def split_output_labels(o_lbls, o_lbls_types, var_type='output_labels'):
    """Order labels as regression then classification and return ``(labels, n_regression, cls_names)``."""
    if not o_lbls:
        return [], 0, []
    if len(o_lbls_types) == len(o_lbls):
        # Number of task types labels (regression/classification) is equal to the number of output variables
        unq_lbl_types = set([ch for ch in o_lbls_types.lower()])
    elif len(o_lbls_types) == 1:
        # Only one task type label (regression/classification) is given for all output variables
        unq_lbl_types = o_lbls_types.lower()
    else:
        # A wrong number of task type labels was given (empty or different from 1 or 'len(output_labels)')
        raise TypeError(
            f"The lengths of '{var_type}' and '{var_type}_types' do not match (should be equal or 'len({var_type}_types)=1').")
    if not set(unq_lbl_types) <= {'r', 'c'}:
        raise TypeError(f"'{var_type}_types' contains unrecognized letters (should include 'r' and/or 'c' only).")

    if len(unq_lbl_types) > 1:
        output_label_types_int = [0 if (ch == 'r') else 1 for ch in o_lbls_types.lower()]
        o_reg_len = len(output_label_types_int) - sum(output_label_types_int)
        cls_o_names = [o_lbls[i_c] for i_c, c in enumerate(output_label_types_int) if c == 1]
        output_order = np.argsort(output_label_types_int)
        o_lbls = [o_lbls[i] for i in output_order]
        if var_type == 'output_labels':
            logging.info('Training with regression and classification heads')
        else:
            logging.info('Loaded model has regression and classification heads')
        logging.info(f'Updated {var_type} order: {o_lbls}')
    elif 'r' in unq_lbl_types:
        o_reg_len = len(o_lbls)
        cls_o_names = []
        if var_type == 'output_labels':
            logging.info('Training only with a regression head')
        else:
            logging.info('Loaded model has only a regression head')
    else:
        o_reg_len = 0
        cls_o_names = o_lbls
        if var_type == 'output_labels':
            logging.info('Training only with a classification head')
        else:
            logging.info('Loaded model has only a classification head')

    return o_lbls, o_reg_len, cls_o_names


def head_spec(n_regression=0, cls_category_map_dicts=None, survival_tasks=(),
              add_dense_reg=False, add_dense_cls=False):
    """Keyword arguments for ``create_regressor_classifier`` describing the output heads."""
    category_order = list((cls_category_map_dicts or {}).get('cls_output_order', []))
    return {
        'n_output_features': n_regression,
        'categories': {c: len(cls_category_map_dicts[c]) for c in category_order},
        'category_order': category_order or None,
        'survival_heads': {task['name']: task['intervals'] for task in survival_tasks},
        'add_dense': {'regressor': add_dense_reg, 'classifier': add_dense_cls},
    }


@dataclass
class TrainedRun:
    """Head configuration recorded by a training run."""
    params: dict
    output_labels: list
    n_regression: int
    cls_category_map_dicts: dict
    survival_tasks: list
    head_spec: dict


def read_trained_run(checkpoint_path):
    run_dir = run_dir_from_checkpoint(checkpoint_path)
    params = read_json(os.path.join(run_dir, MODEL_PARAMS_FILE))
    output_labels, n_regression, cls_names = split_output_labels(
        params.get('output_labels') or [], params.get('output_labels_types', 'r'),
        var_type='output_signature_labels')
    cls_category_map_dicts = read_json(os.path.join(run_dir, CLASS_MAPPING_FILE)) if cls_names else {}
    survival_tasks = validate_survival_tasks(params.get('survival_task'))
    spec = head_spec(
        n_regression, cls_category_map_dicts, survival_tasks,
        params.get('add_separate_dense_reg', False), params.get('add_separate_dense_cls', False))
    if not (n_regression or cls_names or survival_tasks):
        raise ValueError(f'{run_dir} records no regression, classification, or survival outputs.')
    return TrainedRun(params, output_labels, n_regression, cls_category_map_dicts, survival_tasks, spec)


def build_encoder(n_input_frames, batch_size, movinet_chkp_dir, freeze_backbone=False):
    """MoViNet-A2 backbone from its pretrained checkpoint, flattened to one embedding per clip."""
    _, backbone = create_movinet_classifier(
        n_input_frames,
        batch_size,
        num_classes=600,
        checkpoint_dir=movinet_chkp_dir,
        freeze_backbone=freeze_backbone,
    )
    flatten = tf.keras.layers.Flatten()(backbone.layers[-1].output[0])
    return tf.keras.Model(inputs=[backbone.input], outputs=[flatten])


def build_model(encoder, spec, n_input_frames, trainable=True):
    return create_regressor_classifier(
        encoder, trainable=trainable, input_shape=(n_input_frames, 224, 224, 3), **spec)


def load_trained_model(checkpoint_path, encoder, n_input_frames, trainable=True):
    """Rebuild a trained run's heads on ``encoder`` and load its weights (encoder included)."""
    run = read_trained_run(checkpoint_path)
    model = build_model(encoder, run.head_spec, n_input_frames, trainable=trainable)
    model.load_weights(checkpoint_path)
    return model, run
