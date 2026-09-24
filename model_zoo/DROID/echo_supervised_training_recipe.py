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
from data_descriptions.echo_dataset import make_dataset
from data_descriptions.transforms import AUGMENTATIONS
from data_descriptions.wide_file import EcholabDataDescription
from echo_defines import category_dictionaries
from model_descriptions.droid_model import (
    CLASS_MAPPING_FILE, OUTPUT_SCALING_FILE, build_encoder, build_model, head_spec, load_trained_model, read_json,
    run_dir_from_checkpoint, split_output_labels, validate_survival_tasks,
)
from model_descriptions.echo import train_model

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

USER = os.getenv('USER')


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay.

    LR ramps linearly from ``warmup_start_lr`` to ``peak_lr`` over ``warmup_steps``,
    then cosine-decays from ``peak_lr`` down to ``alpha * peak_lr`` over the remaining
    ``total_steps - warmup_steps`` steps. Implemented with basic ops so it does not
    depend on the ``warmup_target``/``warmup_steps`` kwargs of Keras' built-in
    CosineDecay, which are absent on the legacy tf.keras stack DROID runs on
    (TF_USE_LEGACY_KERAS=1).
    """

    def __init__(self, peak_lr, total_steps, warmup_steps=0, warmup_start_lr=0.0, alpha=0.0, name=None):
        super().__init__()
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.alpha = alpha
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        peak_lr = tf.cast(self.peak_lr, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_start_lr = tf.cast(self.warmup_start_lr, tf.float32)
        alpha = tf.cast(self.alpha, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = warmup_start_lr + (peak_lr - warmup_start_lr) * (step / tf.maximum(1.0, warmup_steps))

        decay_steps = tf.maximum(1.0, total_steps - warmup_steps)
        progress = tf.clip_by_value((step - warmup_steps) / decay_steps, 0.0, 1.0)
        cosine = 0.5 * (1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * progress))
        cosine_lr = peak_lr * ((1.0 - alpha) * cosine + alpha)

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            'peak_lr': self.peak_lr,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'warmup_start_lr': self.warmup_start_lr,
            'alpha': self.alpha,
            'name': self.name,
        }


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
        optimizer_name,
        learning_rate,
        lr_schedule,
        warmup_epochs,
        warmup_start_frac,
        alpha,
        weight_decay,
        scale_outputs,
        es_patience,
        es_loss2monitor,
        output_labels_types,
        add_separate_dense_reg,
        add_separate_dense_cls,
        loss_weights,
        randomize_start_frame,
        transforms=None,
        transform_prob=0.05,
        survival_task=None,
        survival_event_column=None,
        survival_follow_up_days_column=None,
        survival_intervals=None,
        survival_days_window=None,
        survival_blanking_days=0,
        survival_prevalent_policy='first_interval',
        run_validation_inference=False,
        loader_workers=4,
        video_decode_mode='auto',
        video_decode_threads=1,
        prefetch_batches=1,
        disable_survival_metrics_callback=False,
):

    if loader_workers < 1 or video_decode_threads < 0 or prefetch_batches < 0:
        raise ValueError('loader_workers must be positive; decode threads/prefetch must be non-negative')
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
    model_params.update({
        'loader_workers': loader_workers,
        'video_decode_mode': video_decode_mode,
        'video_decode_threads': video_decode_threads,
        'prefetch_batches': prefetch_batches,
    })
    lmdb_vois = '_'.join(selected_views)
    olabels = '_'.join(output_labels) or 'survival'

    # ---------- Adaptation for regression + classification ---------- #
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

    # ---------------------------------------------------------------- #
    output_labels, output_reg_len, cls_output_names = split_output_labels(output_labels, output_labels_types,
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
        n_train_patients_value = float(n_train_patients)
        if 0 < n_train_patients_value <= 1:
            patient_train = patient_train[:int(len(patient_train) * n_train_patients_value)]
            patient_valid = patient_valid[:int(len(patient_valid) * n_train_patients_value)]
        else:
            patient_train = patient_train[:int(n_train_patients_value * 0.9)]
            patient_valid = patient_valid[:int(n_train_patients_value * 0.1)]

    train_ids = [t for t in working_ids if int(t.split('_')[0]) in patient_train]
    valid_ids = [t for t in working_ids if int(t.split('_')[0]) in patient_valid]
    print(f"train_ids: {len(train_ids)}") 
    print(f"valid_ids: {len(valid_ids)}") 

    # If scale_outputs, normalize by summary stats of training set
    output_scaling = {}
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
            output_scaling = {
                label: {'mean': float(mean), 'std': float(std)}
                for label, mean, std in zip(output_labels_to_scale, mean_outputs, std_outputs)
            }

    valid_ids = list(set(valid_ids).intersection(set(working_ids)))
    print(f"valid_ids: {len(valid_ids)}") 

    # ---------- Adaptation for regression + classification ---------- #
    cls_category_map_dicts, cls_category_len_dict = process_class_categories(wide_df_selected, cls_output_names,
                                                                             var_type='output_labels')

    if pretrained_chkp_dir:
        cls_lbl_map_path = os.path.join(run_dir_from_checkpoint(pretrained_chkp_dir), CLASS_MAPPING_FILE)
        define_new_heads = False
        if tf.io.gfile.exists(cls_lbl_map_path):
            cls_category_signature_map_dicts = read_json(cls_lbl_map_path)
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

    # Each requested transform is applied independently with probability
    # transform_prob; its own defaults control how it is applied.
    unknown_transforms = [t for t in (transforms or []) if t not in AUGMENTATIONS]
    if unknown_transforms:
        raise ValueError(
            f"Unknown transforms {unknown_transforms}; choose from {sorted(AUGMENTATIONS)}.")
    train_transforms = [AUGMENTATIONS[t](p=transform_prob) for t in (transforms or [])]
    INPUT_DD_TRAIN = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        train_transforms,
        n_input_frames,
        skip_modulo,
        randomize_start_frame=randomize_start_frame,
        decode_mode=video_decode_mode,
        decode_threads=video_decode_threads,
    )
    
    INPUT_DD_VALID = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo,
        randomize_start_frame=False,
        decode_mode=video_decode_mode,
        decode_threads=video_decode_threads,
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
        workers=loader_workers,
    ).repeat().prefetch(prefetch_batches)

    io_valid_ds = make_dataset(
        INPUT_DD_VALID,
        OUTPUT_DD,
        valid_ids,
        batch_size,
        output_signatures,
        shuffle=False,
        workers=loader_workers,
    ).prefetch(prefetch_batches)

    logging.info('Video loader: workers=%d, decode=%s, decode_threads=%d, prefetch_batches=%d',
                 loader_workers, video_decode_mode, video_decode_threads, prefetch_batches)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        encoder = build_encoder(n_input_frames, batch_size, movinet_chkp_dir, freeze_backbone=fine_tune)
        spec = head_spec(output_reg_len, cls_category_map_dicts, survival_tasks,
                         add_separate_dense_reg, add_separate_dense_cls)
        model = build_model(encoder, spec, n_input_frames, trainable=not fine_tune)

        if pretrained_chkp_dir:
            # Loading also restores the shared `encoder`, so `model` keeps the pretrained encoder
            # with new heads when the outputs differ.
            pretrained_model, pretrained = load_trained_model(
                pretrained_chkp_dir, encoder, n_input_frames, trainable=not fine_tune)
            logging.info(f'output_labels of loaded model: {pretrained.output_labels}')
            if (output_labels != pretrained.output_labels) or (pretrained.n_regression != output_reg_len) or (
                    (pretrained.head_spec['category_order'] or []) != cls_output_names) or (
                    survival_tasks != pretrained.survival_tasks) or define_new_heads:
                logging.info('Redefining regression and/or classification heads due to differences in outputs used')
            else:
                model = pretrained_model

        # Peak LR: prefer explicit --learning_rate; fall back to the legacy --adam
        # value, then to the historical RMSprop default scaled by batch size.
        peak_lr = learning_rate if learning_rate is not None else adam
        if peak_lr is None:
            peak_lr = 0.00005 * batch_size

        total_steps = n_train_steps * epochs
        warmup_steps = int(warmup_epochs * n_train_steps)

        if lr_schedule == 'cosine':
            lr = WarmupCosineDecay(
                peak_lr=peak_lr,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                warmup_start_lr=peak_lr * warmup_start_frac,
                alpha=alpha,
            )
        else:  # 'constant'
            lr = peak_lr

        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer_name == 'adamw':
            adamw_cls = getattr(tf.keras.optimizers, 'AdamW', None)
            if adamw_cls is None:
                experimental_optimizers = getattr(tf.keras.optimizers, 'experimental', None)
                if experimental_optimizers is not None:
                    adamw_cls = getattr(experimental_optimizers, 'AdamW', None)
            if adamw_cls is None:
                raise ValueError('AdamW is not available in this TensorFlow/Keras build; use --optimizer adam or rmsprop.')
            optimizer = adamw_cls(learning_rate=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(
                lr,
                rho=0.9,
                momentum=0.9,
                epsilon=1.0,
                clipnorm=1.0
            )
        else:
            raise ValueError(f'Unknown optimizer {optimizer_name!r}.')

        def make_classification_metrics():
            # Fresh metric instances per output: Keras rejects the same metric
            # object being attached to more than one output.
            return [
                tf.keras.metrics.CategoricalAccuracy(),
                tf.keras.metrics.AUC(name='AUROC'),
                tf.keras.metrics.AUC(curve="PR", name='AUPRC')
            ]

        loss = {'cls_' + k: tf.keras.losses.CategoricalCrossentropy() for k in cls_category_len_dict.keys()}
        metrics = {'cls_' + k: make_classification_metrics() for k in cls_category_len_dict.keys()}
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
    # Training-set mean/std of each scaled regression label; predictions are in these scaled units
    # (original = prediction * std + mean).
    if output_scaling:
        with tf.io.gfile.GFile(f'{output_folder}/{OUTPUT_SCALING_FILE}', 'w') as json_file:
            json.dump(output_scaling, json_file)

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
    trained_model = train_model(
        model,
        io_train_ds,
        io_valid_ds,
        epochs,
        n_train_steps,
        n_valid_steps,
        output_folder,
        es_flags,
        batch_size=batch_size,
        valid_ids=valid_ids,
        output_labels=output_labels,
        output_reg_len=output_reg_len,
        cls_category_map_dicts=cls_category_map_dicts,
        survival_tasks=survival_tasks,
        run_summary=run_summary,
        disable_survival_metrics_callback=disable_survival_metrics_callback,
        run_validation_inference_flag=run_validation_inference,
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
    parser.add_argument('--loader_workers', type=int, default=4,
                        help='Parallel clip loaders (start with 4 on an 8-12 CPU host).')
    parser.add_argument('--video_decode_mode', choices=['auto', 'sequential'], default='auto',
                        help='auto skips unselected MJPEG frames; other codecs decode sequentially.')
    parser.add_argument('--video_decode_threads', type=int, default=1,
                        help='Threads per generic decoder; selective MJPEG always uses one. 0 = FFmpeg auto.')
    parser.add_argument('--prefetch_batches', type=int, default=1,
                        help='Bounded batch prefetch; 0 disables prefetch.')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--pretrained_chkp_dir', type=str)
    parser.add_argument('--movinet_chkp_dir', type=str)
    parser.add_argument('--output_dir', type=str,
                        default='gs://mgb-home/alalusim/droid-af/artifacts/training_runs/',
                        help='Base directory for all run artifacts (checkpoints, TensorBoard logs, params). '
                             'May be a local path or a gs:// bucket path. A unique per-run subfolder named '
                             'from the model parameters and date/time is created under it.')
    parser.add_argument('--splits_file')
    parser.add_argument('--adam', default=None, type=float,
                        help='Legacy: constant Adam learning rate. Prefer --learning_rate with --optimizer/--lr_schedule.')
    parser.add_argument('--optimizer', dest='optimizer_name', choices=['adam', 'adamw', 'rmsprop'], default='adam',
                        help='Optimizer to use.')
    parser.add_argument('--learning_rate', default=None, type=float,
                        help='Peak learning rate (warmup target). Falls back to --adam if unset.')
    parser.add_argument('--lr_schedule', choices=['constant', 'cosine'], default='constant',
                        help='constant keeps a fixed learning rate; cosine enables warmup + cosine decay.')
    parser.add_argument('--warmup_epochs', default=0.0, type=float,
                        help='Length of linear warmup in epochs (0 disables warmup).')
    parser.add_argument('--warmup_start_frac', default=0.0, type=float,
                        help='Warmup start learning rate as a fraction of peak LR.')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='Final learning rate as a fraction of peak LR at the end of cosine decay.')
    parser.add_argument('--weight_decay', default=0.1, type=float,
                        help='Weight decay for the adamw optimizer.')
    parser.add_argument('--scale_outputs', action='store_true')
    parser.add_argument('--es_patience', default=3, type=int,
                        help='Number of epochs with no change before early stopping.')
    parser.add_argument('--es_loss2monitor', default='val_loss', type=str,
                        help='Loss on which early stopping is based: "val_loss", "val_echolab_loss", "val_cls_COLUMN-NAME_loss", or "val_survival_NAME_loss".')
    parser.add_argument('--randomize_start_frame', action='store_true')
    parser.add_argument('--transforms', action='append', choices=sorted(AUGMENTATIONS.keys()),
                        help='Training-clip augmentations to enable (repeatable). '
                             'Each enabled transform is applied independently with --transform_prob.')
    parser.add_argument('--transform_prob', default=0.05, type=float,
                        help='Probability that any individual enabled transform is applied to a training clip.')
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
    parser.add_argument('--disable_survival_metrics_callback', action='store_true',
                        help='Disable periodic full-validation survival AUROC/concordance metrics during model.fit. '
                             'The survival head and validation loss remain enabled.')
    # ---------------------------------------------------------------- #
    parser.add_argument('--run_validation_inference', action='store_true',
                        help='Run a post-training validation inference pass over the best checkpoint '
                             'and save predictions/metrics under output_folder/validation_inference.')
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
        optimizer_name=args.optimizer_name,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        warmup_start_frac=args.warmup_start_frac,
        alpha=args.alpha,
        weight_decay=args.weight_decay,
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
        transforms=args.transforms,
        transform_prob=args.transform_prob,
        survival_task=args.survival_task,
        survival_event_column=args.survival_event_column,
        survival_follow_up_days_column=args.survival_follow_up_days_column,
        survival_intervals=args.survival_intervals,
        survival_days_window=args.survival_days_window,
        survival_blanking_days=args.survival_blanking_days,
        survival_prevalent_policy=args.survival_prevalent_policy,
        disable_survival_metrics_callback=args.disable_survival_metrics_callback,
        run_validation_inference=args.run_validation_inference,
        loader_workers=args.loader_workers,
        video_decode_mode=args.video_decode_mode,
        video_decode_threads=args.video_decode_threads,
        prefetch_batches=args.prefetch_batches,
    )
