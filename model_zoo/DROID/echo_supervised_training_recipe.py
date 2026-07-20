import argparse
import datetime
import io
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from ml4h.metrics import survival_likelihood_loss
from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from data_descriptions.wide_file import EcholabDataDescription
from echo_defines import category_dictionaries
from model_descriptions.echo import create_movinet_classifier, create_regressor_classifier, train_model

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

USER = os.getenv('USER')


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


def survival_task_from_arguments(
        survival_task,
        survival_event_column,
        survival_follow_up_days_column,
        survival_intervals,
        survival_days_window,
        survival_blanking_days,
        survival_prevalent_policy,
):
    """Build the internal task config from the recipe's survival CLI arguments."""
    required_arguments = {
        'survival_event_column': survival_event_column,
        'survival_follow_up_days_column': survival_follow_up_days_column,
        'survival_intervals': survival_intervals,
        'survival_days_window': survival_days_window,
    }
    if survival_task is None:
        supplied_arguments = [name for name, value in required_arguments.items() if value is not None]
        if survival_blanking_days != 0:
            supplied_arguments.append('survival_blanking_days')
        if survival_prevalent_policy != 'first_interval':
            supplied_arguments.append('survival_prevalent_policy')
        if supplied_arguments:
            raise ValueError(
                f"{', '.join(supplied_arguments)} requires --survival_task.",
            )
        return []

    missing_arguments = [name for name, value in required_arguments.items() if value is None]
    if missing_arguments:
        raise ValueError(
            f"--survival_task requires: {', '.join('--' + name for name in missing_arguments)}.",
        )
    return validate_survival_tasks([
        {
            'name': survival_task,
            'event_column': survival_event_column,
            'follow_up_days_column': survival_follow_up_days_column,
            'intervals': survival_intervals,
            'days_window': survival_days_window,
            'blanking_days': survival_blanking_days,
            'prevalent_policy': survival_prevalent_policy,
        },
    ])


def main(
        n_input_frames,
        output_labels,
        wide_file,
        splits_file,
        selected_views,
        selected_doppler,
        selected_quality,
        selected_canonical,
        n_train_patients,
        batch_size,
        epochs,
        skip_modulo,
        lmdb_folder,
        fine_tune,
        model_params,
        pretrained_chkp_dir,
        movinet_chkp_dir,
        output_dir,
        adam,
        scale_outputs,
        es_patience,
        es_loss2monitor,
        output_labels_types,
        add_separate_dense_reg,
        add_separate_dense_cls,
        loss_weights,
        randomize_start_frame,
        survival_task=None,
        survival_event_column=None,
        survival_follow_up_days_column=None,
        survival_intervals=None,
        survival_days_window=None,
        survival_blanking_days=0,
        survival_prevalent_policy='first_interval',
):

    output_labels = output_labels or []
    survival_tasks = survival_task_from_arguments(
        survival_task,
        survival_event_column,
        survival_follow_up_days_column,
        survival_intervals,
        survival_days_window,
        survival_blanking_days,
        survival_prevalent_policy,
    )
    model_params = dict(model_params or {})
    model_params['survival_task'] = survival_tasks
    lmdb_vois = '_'.join(selected_views)
    olabels = '_'.join(output_labels) or 'survival'

    # ---------- Adaptation for regression + classification ---------- #
    def process_labels_types(o_lbls, o_lbls_types, var_type='output_labels'):
        # ---- Processing input values and handling incorrect inputs ----- #
        # Specify parameters for regression and classification heads:
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

    def process_class_categories(df, cls_o_names, var_type='output_labels'):
        # Creating dictionaries specifying number of classes for each output_label name
        # and mapping between wide_file values to class labels:
        clsc_map_dicts = {}
        clsc_len_dict = {}
        for c_lbl in cls_o_names:
            all_cls_vals = np.sort(df[c_lbl].drop_duplicates().tolist())
            val2clsind_map_dict = {val: c_ind for val, c_ind in zip(all_cls_vals, range(len(all_cls_vals)))}
            clsc_map_dicts[c_lbl] = val2clsind_map_dict
            clsc_len_dict[c_lbl] = len(df[c_lbl].drop_duplicates())
            if clsc_len_dict[c_lbl] < 2:
                logging.error(
                    f'Error: Output variable {c_lbl} has a constant value in the train and validation sets - might cause errors in the classifier. Error raised when processing {var_type} related classification variables.')
        clsc_map_dicts['cls_output_order'] = cls_o_names
        return clsc_map_dicts, clsc_len_dict

    def make_dataset(input_dd, output_dd, sample_ids, batch_size, output_signature, shuffle):
        sample_ids = list(sample_ids)

        def generator():
            while True:
                epoch_ids = sample_ids.copy()
                if shuffle:
                    np.random.shuffle(epoch_ids)
                for start_idx in range(0, len(epoch_ids) - batch_size + 1, batch_size):
                    batch_ids = epoch_ids[start_idx:start_idx + batch_size]
                    batch_inputs = []
                    batch_outputs = []
                    for sample_id in batch_ids:
                        batch_inputs.append(input_dd.get_raw_data(sample_id))
                        batch_outputs.append(output_dd.get_raw_data(sample_id))

                    batch_inputs = np.stack(batch_inputs).astype(np.float32, copy=False)
                    if batch_outputs and isinstance(batch_outputs[0], list):
                        batch_outputs = tuple(
                            np.stack([sample_output[output_idx] for sample_output in batch_outputs]).astype(np.float32, copy=False)
                            for output_idx in range(len(batch_outputs[0]))
                        )
                    elif batch_outputs and isinstance(batch_outputs[0], tuple):
                        batch_outputs = tuple(
                            np.stack([sample_output[output_idx] for sample_output in batch_outputs]).astype(np.float32, copy=False)
                            for output_idx in range(len(batch_outputs[0]))
                        )
                    else:
                        batch_outputs = np.stack(batch_outputs).astype(np.float32, copy=False)

                    yield batch_inputs, batch_outputs

        return tf.data.Dataset.from_generator(generator, output_signature=output_signature).prefetch(tf.data.AUTOTUNE)

    # ---------------------------------------------------------------- #
    output_labels, output_reg_len, cls_output_names = process_labels_types(output_labels, output_labels_types,
                                                                           var_type='output_labels')
    # ---------------------------------------------------------------- #
    wide_df = pd.read_parquet(wide_file)

    # Select only view(s) of interest
    selected_views_idx: list[int | dict[str, int | float]] = [category_dictionaries['view'][v] for v in selected_views]
    selected_doppler_idx = [category_dictionaries['doppler'][v] for v in selected_doppler]
    selected_quality_idx = [category_dictionaries['quality'][v] for v in selected_quality]
    selected_canonical_idx = [category_dictionaries['canonical'][v] for v in selected_canonical]
    wide_df_selected = wide_df[
        (wide_df['view_prediction'].isin(selected_views_idx)) &
        (wide_df['doppler_prediction'].isin(selected_doppler_idx)) &
        (wide_df['quality_prediction'].isin(selected_quality_idx)) &
        (wide_df['canonical_prediction'].isin(selected_canonical_idx))
        ]

    required_survival_columns = [
        column
        for task in survival_tasks
        for column in (task['event_column'], task['follow_up_days_column'])
    ]
    required_columns = list(dict.fromkeys(output_labels + required_survival_columns))
    # Drop entries without echolab measurements and get all sample_ids
    wide_df_selected = wide_df_selected.dropna(subset=required_columns)
    for task in survival_tasks:
        events = pd.to_numeric(wide_df_selected[task['event_column']], errors='raise')
        follow_up_days = pd.to_numeric(wide_df_selected[task['follow_up_days_column']], errors='raise')
        if not events.isin([0, 1]).all():
            raise ValueError(f"Survival task {task['name']} event_column must contain only 0 or 1.")
        if not np.isfinite(follow_up_days).all():
            raise ValueError(f"Survival task {task['name']} follow_up_days_column must contain finite values.")
        invalid_censoring = (events == 0) & (follow_up_days < 0)
        if invalid_censoring.any():
            logging.info(
                'Excluding %d samples for survival task %s with censored negative follow-up days.',
                int(invalid_censoring.sum()), task['name'],
            )
            wide_df_selected = wide_df_selected.loc[~invalid_censoring]
            events = events.loc[~invalid_censoring]
            follow_up_days = follow_up_days.loc[~invalid_censoring]
        if task['prevalent_policy'] == 'exclude':
            prevalent_mask = (events == 1) & (follow_up_days <= task['blanking_days'])
            if prevalent_mask.any():
                logging.info(
                    'Dropping %d prevalent samples for survival task %s.',
                    int(prevalent_mask.sum()), task['name'],
                )
                wide_df_selected = wide_df_selected.loc[~prevalent_mask]
    working_ids = wide_df_selected['sample_id'].values.tolist()

    # Read splits and partition dataset
    with open(splits_file, 'r') as json_file:
        splits = json.load(json_file)

    patient_train = splits['patient_train']
    patient_valid = splits['patient_valid']

    if n_train_patients != 'all':
        patient_train = patient_train[:int(int(n_train_patients) * 0.9)]
        patient_valid = patient_valid[:int(int(n_train_patients) * 0.1)]

    train_ids = [t for t in working_ids if int(t.split('_')[0]) in patient_train]
    valid_ids = [t for t in working_ids if int(t.split('_')[0]) in patient_valid]
    print(f"train_ids: {len(train_ids)}") 
    print(f"valid_ids: {len(valid_ids)}") 

    # If scale_outputs, normalize by summary stats of training set
    if scale_outputs:
        wide_df_train = wide_df_selected[wide_df_selected['sample_id'].isin(train_ids)]
        output_labels_to_scale = np.array([l for l in output_labels if l not in cls_output_names])
        if len(output_labels_to_scale) > 0:
            output_labels_to_scale = list(output_labels_to_scale[
                                              np.logical_and(wide_df_train[output_labels_to_scale].dtypes != 'object',
                                                             wide_df_train[output_labels_to_scale].dtypes != 'string')])
        else:
            output_labels_to_scale = []
        logging.info(
            f'Not scaling classification columns and columns containing strings/objects, unscaled columns: {[l for l in output_labels if l not in output_labels_to_scale]}')
        if output_labels_to_scale:
            mean_outputs = np.mean(wide_df_train[output_labels_to_scale].values, axis=0)
            std_outputs = np.std(wide_df_train[output_labels_to_scale].values, axis=0)
            wide_df_selected.loc[:, output_labels_to_scale] = (wide_df_selected[output_labels_to_scale].values - mean_outputs) / std_outputs
            logging.info(mean_outputs)
            logging.info(std_outputs)

    valid_ids = list(set(valid_ids).intersection(set(working_ids)))
    print(f"valid_ids: {len(valid_ids)}") 

    # ---------- Adaptation for regression + classification ---------- #
    cls_category_map_dicts, cls_category_len_dict = process_class_categories(wide_df_selected, cls_output_names,
                                                                             var_type='output_labels')

    if pretrained_chkp_dir:
        cls_lbl_map_path = os.path.join(os.path.split(os.path.dirname(pretrained_chkp_dir))[0],
                                        'classification_class_label_mapping_per_output.json')
        define_new_heads = False
        if os.path.isfile(cls_lbl_map_path):
            with open(cls_lbl_map_path, 'r') as json_file:
                cls_category_signature_map_dicts = json.load(json_file)
            similar_cls = [c for c in cls_output_names if c in cls_category_signature_map_dicts.keys()]
            for c in similar_cls:
                if (len(cls_category_signature_map_dicts[c]) > len(cls_category_map_dicts[c])) and set(
                        cls_category_map_dicts[c].keys()).issubset(set(cls_category_signature_map_dicts[c].keys())):
                    cls_category_map_dicts[c] = cls_category_signature_map_dicts[c]
                    cls_category_len_dict[c] = len(cls_category_map_dicts[c])
                    logging.info(f'Using mapping from pretrained_chkp_dir for classification task on {c}')
                elif not set(
                        cls_category_map_dicts[c].keys()).issubset(set(cls_category_signature_map_dicts[c].keys())):
                    define_new_heads = True
    # ---------------------------------------------------------------- #

    INPUT_DD_TRAIN = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo,
        randomize_start_frame=randomize_start_frame
    )
    
    INPUT_DD_VALID = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo,
        randomize_start_frame = False
    )

    survival_source_columns: list[Unknown] = [
        column for task in survival_tasks
        for column in (task['event_column'], task['follow_up_days_column'])
    ]
    output_dd_columns = list(dict.fromkeys(['sample_id'] + output_labels + survival_source_columns))
    OUTPUT_DD = EcholabDataDescription(
        wide_df=wide_df_selected[output_dd_columns].drop_duplicates(),
        sample_id_column='sample_id',
        column_names=output_labels,
        name='echolab',
        # ---------- Adaptation for regression + classification ---------- #
        cls_categories_map=cls_category_map_dicts if cls_output_names else None,
        survival_task_configs=survival_tasks,
        # ---------------------------------------------------------------- #
    )

    n_train_steps = len(train_ids) // batch_size
    n_valid_steps = len(valid_ids) // batch_size
    print(f"n_train_steps: {n_train_steps}")
    print(f"n_valid_steps: {n_valid_steps}")

    # ---------- Adaptation for regression + classification ---------- #
    # Adapting tensor output sizes for classification heads
    output_shapes = []
    if output_reg_len > 0:
        output_shapes.append((batch_size, output_reg_len))
    output_shapes.extend(
        [(batch_size, cls_category_len_dict[c]) for c in cls_category_map_dicts['cls_output_order']]
    )
    output_shapes.extend(
        [(batch_size, task['intervals'] * 2) for task in survival_tasks]
    )
    if not output_shapes:
        raise ValueError('Specify at least one regression, classification, or survival output.')
    if len(output_shapes) > 1:
        output_signatures = (
            tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32),
            tuple([tf.TensorSpec(shape=shape, dtype=tf.float32) for shape in output_shapes])
        )
    else:
        output_signatures = (
            tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=output_shapes[0], dtype=tf.float32)
        )
    # ---------------------------------------------------------------- #
    io_train_ds = make_dataset(
        INPUT_DD_TRAIN,
        OUTPUT_DD,
        train_ids,
        batch_size,
        output_signatures,
        shuffle=True,
    )

    io_valid_ds = make_dataset(
        INPUT_DD_VALID,
        OUTPUT_DD,
        valid_ids,
        batch_size,
        output_signatures,
        shuffle=False,
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        _, backbone = create_movinet_classifier(
            n_input_frames,
            batch_size,
            num_classes=600,
            checkpoint_dir=movinet_chkp_dir,
            freeze_backbone=fine_tune
        )
        backbone_output = backbone.layers[-1].output[0]
        flatten = tf.keras.layers.Flatten()(backbone_output)
        encoder = tf.keras.Model(inputs=[backbone.input], outputs=[flatten])

        # ---------- Adaptation for regression + classification ---------- #
        # Organize regressor/classifier inputs:
        func_args = {'input_shape': (n_input_frames, 224, 224, 3), 'trainable': not fine_tune,
                     'n_output_features': output_reg_len,
                     'categories': cls_category_len_dict,
                     'survival_heads': {task['name']: task['intervals'] for task in survival_tasks},
                     'category_order': cls_category_map_dicts['cls_output_order'] if cls_category_len_dict else None,
                     'add_dense': {
                         'regressor': add_separate_dense_reg,
                         'classifier': add_separate_dense_cls,
                     }}

        model = create_regressor_classifier(encoder, **func_args)
        # ---------------------------------------------------------------- #

        if pretrained_chkp_dir:
            signature_model_param_path = os.path.join(os.path.split(os.path.dirname(pretrained_chkp_dir))[0],
                                                      'model_params.json')
            f = open(signature_model_param_path)
            signature_model_params = json.load(f)
            sig_add_separate_dense_reg = signature_model_params[
                'add_separate_dense_reg'] if 'add_separate_dense_reg' in signature_model_params.keys() else False
            sig_add_separate_dense_cls = signature_model_params[
                'add_separate_dense_cls'] if 'add_separate_dense_cls' in signature_model_params.keys() else False
            output_signature_labels_types = signature_model_params[
                'output_labels_types'] if 'output_labels_types' in signature_model_params.keys() else 'r'
            output_signature_labels = signature_model_params['output_labels']
            logging.info(f'output_labels of loaded model: {output_signature_labels}')

            output_signature_labels, output_signature_reg_len, cls_output_signature_names = process_labels_types(
                output_signature_labels, output_signature_labels_types, var_type='output_signature_labels')
            signature_survival_tasks = validate_survival_tasks(signature_model_params.get('survival_task', []))

            if 'c' in output_signature_labels_types.lower():
                cls_category_signature_len_dict = {}
                for c_lbl in cls_category_signature_map_dicts['cls_output_order']:
                    cls_category_signature_len_dict[c_lbl] = len(cls_category_signature_map_dicts[c_lbl])
            else:
                cls_category_signature_len_dict = {}

            model = create_regressor_classifier(
                encoder,
                input_shape=(n_input_frames, 224, 224, 3),
                trainable=not fine_tune,
                n_output_features=output_signature_reg_len,
                categories=cls_category_signature_len_dict,
                survival_heads={task['name']: task['intervals'] for task in signature_survival_tasks},
                category_order=cls_category_signature_map_dicts[
                    'cls_output_order'] if cls_category_signature_len_dict else None,
                add_dense={
                    'regressor': sig_add_separate_dense_reg,
                    'classifier': sig_add_separate_dense_cls,
                }
            )
            model.load_weights(pretrained_chkp_dir)

            if (output_labels != output_signature_labels) or (output_signature_reg_len != output_reg_len) or (
                    cls_output_signature_names != cls_output_names) or (
                    survival_tasks != signature_survival_tasks) or define_new_heads:
                logging.info('Redefining regression and/or classification heads due to differences in outputs used')
                # ---------- Adaptation for regression + classification ---------- #
                model = create_regressor_classifier(encoder, **func_args)
                # ---------------------------------------------------------------- #

        if adam:
            optimizer = tf.keras.optimizers.Adam(learning_rate=adam)
        else:
            initial_learning_rate = 0.00005 * batch_size
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate,
                decay_steps=n_train_steps * epochs,
            )

            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate,
                rho=0.9,
                momentum=0.9,
                epsilon=1.0,
                clipnorm=1.0
            )

        classification_metrics = [
            tf.keras.metrics.CategoricalAccuracy(), 
            tf.keras.metrics.AUC(name='AUROC'),
            tf.keras.metrics.AUC(curve="PR", name='AUPRC')
        ]
        
        loss = {'cls_' + k: tf.keras.losses.CategoricalCrossentropy() for k in cls_category_len_dict.keys()}
        metrics = {'cls_' + k: classification_metrics for k in cls_category_len_dict.keys()}
        if output_reg_len > 0:
            loss['echolab'] = tf.keras.losses.MeanSquaredError()
            metrics['echolab'] = tf.keras.metrics.MeanAbsoluteError()
        for task in survival_tasks:
            loss[f"survival_{task['name']}"] = survival_likelihood_loss(task['intervals'])

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights if loss_weights else None
        )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    io_train_ds = io_train_ds.with_options(options)
    io_valid_ds = io_valid_ds.with_options(options)

    # Unique per-run folder derived from the model parameters and date/time. output_dir may be a
    # local path or a gs:// bucket path; all writes below go through tf.io.gfile, which handles
    # both transparently, so checkpoints, TensorBoard logs, and params all land in one place.
    fine_tune_string = f'_fine_tune' if fine_tune else ''
    run_name = f'{datetime.datetime.now().strftime("%Y%m%d%H%M")}_{lmdb_vois}_{olabels}_{n_input_frames}frames{fine_tune_string}_{n_train_patients}'
    output_folder = f'{output_dir.rstrip("/")}/{run_name}'

    tf.io.gfile.makedirs(output_folder)
    with tf.io.gfile.GFile(f'{output_folder}/model_params.json', 'w') as json_file:
        json.dump(model_params, json_file)

    parquet_buffer = io.BytesIO()
    wide_df_selected.to_parquet(parquet_buffer)
    with tf.io.gfile.GFile(f'{output_folder}/wide_df_selected.pq', 'wb') as pq_file:
        pq_file.write(parquet_buffer.getvalue())

    # ---------- Adaptation for regression + classification ---------- #
    # Record output labels new order (after possible reordering of regression and classification):
    with tf.io.gfile.GFile(f'{output_folder}/output_labels_final_ordering.json', 'w') as json_file:
        json.dump(output_labels, json_file)
    # Record output mapping for classification tasks (dictionary that contains column names as well):
    if cls_output_names:
        cls_category_map_dicts['add_separate_dense_cls'] = add_separate_dense_cls
        cls_category_map_dicts['add_separate_dense_reg'] = add_separate_dense_reg
        with tf.io.gfile.GFile(f'{output_folder}/classification_class_label_mapping_per_output.json', 'w') as json_file:
            json.dump(cls_category_map_dicts, json_file)
    # ---------------------------------------------------------------- #

    es_flags = {'es_patience': es_patience, 'es_loss2monitor': es_loss2monitor}

    logging.info(f'Writing run artifacts (checkpoints, logs, params) to {output_folder}')
    logging.info(model.summary())
    run_summary = {
        'run_name': run_name,
        'output_labels': output_labels,
        'selected_views': selected_views,
        'epochs': epochs,
        'batch_size': batch_size,
        'n_train_patients': n_train_patients,
    }
    run_summary = {
        'run_name': run_name,
        'output_labels': output_labels,
        'selected_views': selected_views,
        'epochs': epochs,
        'batch_size': batch_size,
        'n_train_patients': n_train_patients,
    }
    trained_model = train_model(
        model,
        io_train_ds,
        io_valid_ds,
        epochs,
        n_train_steps,
        n_valid_steps,
        output_folder,
        es_flags,
<<<<<<< HEAD
        batch_size=batch_size,
        valid_ids=valid_ids,
        output_labels=output_labels,
        output_reg_len=output_reg_len,
        cls_category_map_dicts=cls_category_map_dicts,
        survival_tasks=survival_tasks,
        run_summary=run_summary,
=======
>>>>>>> 8b196cb (add tensorboard profiling)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_input_frames', type=int, default=32)
    parser.add_argument('-o', '--output_labels', action='append', type=str)
    parser.add_argument('--wide_file', type=str)
    parser.add_argument('-v', '--selected_views', action='append', choices=category_dictionaries['view'].keys(),
                        required=True)
    parser.add_argument('-d', '--selected_doppler', action='append', choices=category_dictionaries['doppler'].keys(),
                        required=True)
    parser.add_argument('-q', '--selected_quality', action='append', choices=category_dictionaries['quality'].keys(),
                        required=True)
    parser.add_argument('-c', '--selected_canonical', action='append',
                        choices=category_dictionaries['canonical'].keys(), required=True)
    parser.add_argument('-n', '--n_train_patients', type=str, required=True)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--skip_modulo', type=int, default=2)
    parser.add_argument('--lmdb_folder', type=str)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--pretrained_chkp_dir', type=str)
    parser.add_argument('--movinet_chkp_dir', type=str)
    parser.add_argument('--output_dir', type=str,
                        default='gs://mgb-home/alalusim/droid-af/artifacts/training_runs/',
                        help='Base directory for all run artifacts (checkpoints, TensorBoard logs, params). '
                             'May be a local path or a gs:// bucket path. A unique per-run subfolder named '
                             'from the model parameters and date/time is created under it.')
    parser.add_argument('--splits_file')
    parser.add_argument('--adam', default=None, type=float)
    parser.add_argument('--scale_outputs', action='store_true')
    parser.add_argument('--es_patience', default=3, type=int,
                        help='Number of epochs with no change before early stopping.')
    parser.add_argument('--es_loss2monitor', default='val_loss', type=str,
                        help='Loss on which early stopping is based: "val_loss", "val_echolab_loss", "val_cls_COLUMN-NAME_loss", or "val_survival_NAME_loss".')
    parser.add_argument('--randomize_start_frame', action='store_true')
    # ---------- Adaptation for regression + classification ---------- #
    parser.add_argument('--output_labels_types', default='r', type=str,
                        help='A string indicating task types: r for regression, c for classification. Should be of length 1 or the same length of the specified output_labels variable, e.g. "r" or "rrcr".')
    parser.add_argument('--add_separate_dense_reg', action='store_true',
                        help='Adds an additional dense layer trained separately for the regression head')
    parser.add_argument('--add_separate_dense_cls', action='store_true',
                        help='Adds an additional dense layer trained separately for the classification head')
    parser.add_argument('-lw', '--loss_weights', action='append', type=float,
                        help='Loss weights in model-output order: regression (if any), classification tasks, then survival tasks.')
    parser.add_argument(
        '--survival_task',
        type=str,
        help='Name for the discrete-time survival output. Requires all --survival_* arguments below.',
    )
    parser.add_argument('--survival_event_column', type=str,
                        help='Binary (0/1) event column for --survival_task.')
    parser.add_argument('--survival_follow_up_days_column', type=str,
                        help='Follow-up duration in days for --survival_task.')
    parser.add_argument('--survival_intervals', type=int,
                        help='Number of discrete prediction intervals for --survival_task.')
    parser.add_argument('--survival_days_window', type=float,
                        help='Prediction horizon in days for --survival_task.')
    parser.add_argument('--survival_blanking_days', type=float, default=0,
                        help='Optional blanking period in days before the survival horizon.')
    parser.add_argument('--survival_prevalent_policy', choices=['first_interval', 'exclude'], default='first_interval',
                        help='How to handle events at or before the blanking period.')
    # ---------------------------------------------------------------- #
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    model_params_dict = {}
    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: {value}")
        model_params_dict[arg] = value

    main(
        n_input_frames=args.n_input_frames,
        output_labels=args.output_labels,
        wide_file=args.wide_file,
        splits_file=args.splits_file,
        selected_views=args.selected_views,
        selected_doppler=args.selected_doppler,
        selected_quality=args.selected_quality,
        selected_canonical=args.selected_canonical,
        n_train_patients=args.n_train_patients,
        batch_size=args.batch_size,
        epochs=args.epochs,
        skip_modulo=args.skip_modulo,
        lmdb_folder=args.lmdb_folder,
        fine_tune=args.fine_tune,
        model_params=model_params_dict,
        pretrained_chkp_dir=args.pretrained_chkp_dir,
        movinet_chkp_dir=args.movinet_chkp_dir,
        output_dir=args.output_dir,
        adam=args.adam,
        scale_outputs=args.scale_outputs,
        es_patience=args.es_patience,
        es_loss2monitor=args.es_loss2monitor,
        # ---------- Adaptation for regression + classification ---------- #
        output_labels_types=args.output_labels_types,
        add_separate_dense_reg=args.add_separate_dense_reg,
        add_separate_dense_cls=args.add_separate_dense_cls,
        loss_weights=args.loss_weights,
        # ---------------------------------------------------------------- #
        randomize_start_frame=args.randomize_start_frame,
        survival_task=args.survival_task,
        survival_event_column=args.survival_event_column,
        survival_follow_up_days_column=args.survival_follow_up_days_column,
        survival_intervals=args.survival_intervals,
        survival_days_window=args.survival_days_window,
        survival_blanking_days=args.survival_blanking_days,
        survival_prevalent_policy=args.survival_prevalent_policy,
    )
