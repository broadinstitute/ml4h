"""Train study-level DROID heads from the embeddings of a completed DROID run.

The source run supplies labels, survival tasks, selected views, and patient
splits. The matching DROID inference output supplies one embedding per clip.
"""

import argparse
import datetime
import io
import json
import logging
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from ml4h.metrics import survival_likelihood_loss
from data_descriptions.study_embeddings import (
    assign_splits, embedding_files, join_embeddings_wide, make_study_dataset,
    make_study_records, read_embeddings,
)
from droid_callbacks import _survival_metrics_report
from echo_supervised_training_recipe import WarmupCosineDecay
from model_descriptions.droid_model import (
    OUTPUT_SCALING_FILE, TrainedRun, head_spec, read_json, read_trained_run,
    split_output_labels, validate_survival_tasks,
)
from model_descriptions.echo import train_model
from model_descriptions.study_attention import build_study_attention_model

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)


def _write_json(path, value):
    with tf.io.gfile.GFile(path, 'w') as stream:
        json.dump(value, stream, indent=2)


def _write_parquet(path, table):
    buffer = io.BytesIO()
    table.to_parquet(buffer, index=False)
    with tf.io.gfile.GFile(path, 'wb') as stream:
        stream.write(buffer.getvalue())


def _default_embedding_root(source_run_dir):
    marker = '/artifacts/training_runs/'
    if marker not in source_run_dir:
        raise ValueError('Cannot derive the inference output path; supply --embeddings_dir.')
    prefix = source_run_dir.split(marker, 1)[0]
    return f'{prefix}/artifacts/inference/{os.path.basename(source_run_dir.rstrip("/"))}'


def _inherit_training_arguments(args, params):
    defaults = {
        'epochs': ('epochs', 50), 'batch_size': ('batch_size', 16),
        'optimizer': ('optimizer_name', 'adam'),
        'learning_rate': ('learning_rate', 1e-4),
        'lr_schedule': ('lr_schedule', 'constant'),
        'warmup_epochs': ('warmup_epochs', 0.0),
        'warmup_start_frac': ('warmup_start_frac', 0.0),
        'alpha': ('alpha', 0.0), 'weight_decay': ('weight_decay', 0.1),
        'es_patience': ('es_patience', 3),
        'es_loss2monitor': ('es_loss2monitor', 'val_loss'),
    }
    for argument, (source_key, fallback) in defaults.items():
        if getattr(args, argument) is None:
            source_value = params.get(source_key)
            if argument == 'learning_rate' and source_value is None:
                source_value = params.get('adam')
            setattr(args, argument, source_value if source_value is not None else fallback)
    for argument in ('run_validation_inference', 'disable_survival_metrics_callback'):
        if getattr(args, argument) is None:
            setattr(args, argument, bool(params.get(argument, False)))
    if args.loss_weights is None:
        args.loss_weights = params.get('loss_weights')


def _configured_run(source, args):
    if args.output_labels is None:
        labels = source.output_labels
        n_regression = source.n_regression
        cls_names = source.head_spec['category_order'] or []
        if args.output_labels_types is not None:
            raise ValueError('--output_labels_types requires --output_labels.')
    else:
        types = args.output_labels_types
        if types is None:
            source_classes = set(source.head_spec['category_order'] or [])
            types = ''.join('c' if label in source_classes else 'r'
                            for label in args.output_labels)
        labels, n_regression, cls_names = split_output_labels(args.output_labels, types)
    tasks = validate_survival_tasks(
        [json.loads(value) for value in args.survival_task_json]
        if args.survival_task_json is not None else source.survival_tasks,
    )
    if not tasks:
        raise ValueError('The study model needs an AF survival task from the source run or --survival_task_json.')
    unknown_classes = set(cls_names) - set(source.cls_category_map_dicts)
    if unknown_classes:
        raise ValueError(f'No source-run category mapping for: {sorted(unknown_classes)}')
    mappings = {name: source.cls_category_map_dicts[name] for name in cls_names}
    if cls_names:
        mappings['cls_output_order'] = cls_names
    spec = head_spec(n_regression, mappings, tasks)
    if not (labels or tasks):
        raise ValueError('Configure at least one regression, classification, or survival output.')
    return TrainedRun(source.params, labels, n_regression, mappings, tasks, spec)


def _output_widths(run):
    widths = []
    if run.n_regression:
        widths.append(run.n_regression)
    widths.extend(len(run.cls_category_map_dicts[name]) for name in
                  run.head_spec['category_order'] or [])
    widths.extend(task['intervals'] * 2 for task in run.survival_tasks)
    return widths


def _optimizer(args, steps_per_epoch):
    peak = args.learning_rate
    if args.lr_schedule == 'cosine':
        lr = WarmupCosineDecay(
            peak_lr=peak, total_steps=steps_per_epoch * args.epochs,
            warmup_steps=int(steps_per_epoch * args.warmup_epochs),
            warmup_start_lr=peak * args.warmup_start_frac, alpha=args.alpha,
        )
    else:
        lr = peak
    if args.optimizer == 'adamw':
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=args.weight_decay)
    if args.optimizer == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def _predict_split(model, dataset, records, run, output_folder, name):
    """Write one row per study and survival metrics on held-out splits."""
    predictions = model.predict(dataset, verbose=1)
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]
    table = pd.DataFrame({
        'study_id': [record['study_id'] for record in records],
        'patient_id': [record['patient_id'] for record in records],
        'n_clips': [len(record['sample_ids']) for record in records],
    })
    metrics = {}
    output_names = model.output_names
    for index, (output_name, values) in enumerate(zip(output_names, predictions)):
        for column in range(values.shape[1]):
            table[f'{output_name}_{column}'] = values[:, column]
        if output_name.startswith('survival_'):
            table[f'{output_name}_cumulative'] = np.prod(values, axis=1)
            task = next(t for t in run.survival_tasks if output_name == f"survival_{t['name']}")
            truth = np.stack([record['targets'][index] for record in records])
            loss = survival_likelihood_loss(task['intervals'])(
                tf.constant(truth), tf.constant(values),
            ).numpy()
            report = {'mean_negative_log_likelihood': float(np.mean(loss))}
            try:
                report.update(_survival_metrics_report(truth, values, task))
            except Exception:
                logging.exception('Could not compute all %s survival metrics.', name)
            metrics[task['name']] = report
    folder = f'{output_folder}/{name}_inference'
    tf.io.gfile.makedirs(folder)
    _write_parquet(f'{folder}/predictions.pq', table)
    _write_json(f'{folder}/metrics.json', metrics)


def main(args):
    source_dir = args.source_run_dir.rstrip('/')
    source = read_trained_run(f'{source_dir}/model/chkp')
    run = _configured_run(source, args)
    params = source.params
    _inherit_training_arguments(args, params)
    split_path = args.splits_file or params['splits_file']
    splits = read_json(split_path)
    train_subset = args.n_train_patients or params.get('n_train_patients', 'all')

    selected_path = f'{source_dir}/wide_df_selected.pq'
    wide_files = args.wide_file or [selected_path]
    embedding_dirs = args.embeddings_dir or [_default_embedding_root(source_dir)]
    paths = embedding_files(embedding_dirs)
    embeddings, columns = read_embeddings(paths)
    scaling_path = f'{source_dir}/{OUTPUT_SCALING_FILE}'
    output_scaling = read_json(scaling_path) if tf.io.gfile.exists(scaling_path) else {}
    merged = join_embeddings_wide(
        embeddings, wide_files, params, run.output_labels, run.survival_tasks,
        source_selected_path=selected_path, output_scaling=output_scaling,
    )
    records = make_study_records(merged, columns, run)
    del embeddings, merged
    assigned = assign_splits(records, splits, train_subset)
    for name, values in assigned.items():
        logging.info('%s: %d studies, %d patients, %d clips', name, len(values),
                     len({record['patient_id'] for record in values}),
                     sum(len(record['sample_ids']) for record in values))
    if not assigned['train'] or not assigned['valid']:
        raise ValueError('Training and validation each need at least one eligible study.')

    batch_size = args.batch_size
    widths = _output_widths(run)
    embedding_dim = len(columns)
    train_dataset = make_study_dataset(
        assigned['train'], embedding_dim, widths, batch_size, shuffle=True,
    ).repeat()
    valid_dataset = make_study_dataset(
        assigned['valid'], embedding_dim, widths, batch_size,
    )
    train_steps = math.ceil(len(assigned['train']) / batch_size)
    valid_steps = math.ceil(len(assigned['valid']) / batch_size)

    model = build_study_attention_model(
        embedding_dim, run.head_spec, args.attention_heads,
        args.attention_key_dim, args.dropout,
    )
    losses = {}
    if run.n_regression:
        losses['echolab'] = tf.keras.losses.MeanSquaredError()
    for name in run.head_spec['category_order'] or []:
        losses[f'cls_{name}'] = tf.keras.losses.CategoricalCrossentropy()
    for task in run.survival_tasks:
        losses[f"survival_{task['name']}"] = survival_likelihood_loss(task['intervals'])
    if args.loss_weights is not None and len(args.loss_weights) != len(losses):
        raise ValueError('--loss_weights must have one value per model output.')
    model.compile(optimizer=_optimizer(args, train_steps), loss=losses,
                  loss_weights=args.loss_weights)

    if args.output_dir:
        output_root = args.output_dir.rstrip('/')
    elif '/artifacts/training_runs/' in source_dir:
        output_root = source_dir.split('/artifacts/training_runs/', 1)[0] + '/artifacts/study_attention_runs'
    else:
        output_root = f'{source_dir}/study_attention_runs'
    run_name = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{os.path.basename(source_dir)}_study_attention'
    output_folder = f'{output_root}/{run_name}'
    tf.io.gfile.makedirs(output_folder)
    _write_json(f'{output_folder}/model_params.json', {
        **vars(args), 'source_run_dir': source_dir, 'embedding_files': paths,
        'wide_files': wide_files, 'splits_file': split_path,
        'n_train_patients': train_subset, 'output_labels': run.output_labels,
        'survival_task': run.survival_tasks, 'embedding_dim': embedding_dim,
    })
    _write_json(f'{output_folder}/study_counts.json', {
        name: {'studies': len(values), 'patients': len({r['patient_id'] for r in values}),
               'clips': sum(len(r['sample_ids']) for r in values)}
        for name, values in assigned.items()
    })

    model = train_model(
        model, train_dataset, valid_dataset, args.epochs, train_steps, valid_steps,
        output_folder, {'es_patience': args.es_patience,
                        'es_loss2monitor': args.es_loss2monitor},
        batch_size=batch_size,
        valid_ids=[record['study_id'] for record in assigned['valid']],
        output_labels=run.output_labels,
        output_reg_len=run.n_regression,
        cls_category_map_dicts=run.cls_category_map_dicts,
        survival_tasks=run.survival_tasks,
        run_summary={'run_name': run_name, 'source_run_dir': source_dir},
        run_validation_inference_flag=args.run_validation_inference,
        disable_survival_metrics_callback=args.disable_survival_metrics_callback,
    )
    for name in ('valid', 'internal_test', 'test'):
        if assigned[name]:
            dataset = make_study_dataset(assigned[name], embedding_dim, widths, batch_size)
            _predict_split(model, dataset, assigned[name], run, output_folder, name)
    logging.info('Saved study-attention run to %s', output_folder)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_run_dir', required=True,
                        help='Completed DROID run containing model_params.json and wide_df_selected.pq.')
    parser.add_argument('--embeddings_dir', action='append',
                        help='Inference root, inference_embeddings_* folder, or prediction_*.pq file. '
                             'Repeat for train/valid and test outputs. Defaults to the matching inference run.')
    parser.add_argument('--wide_file', action='append',
                        help='Wide Parquet input. Repeat to combine train/valid and test. '
                             'Defaults to source_run_dir/wide_df_selected.pq.')
    parser.add_argument('--splits_file', help='Defaults to the source run splits file.')
    parser.add_argument('--n_train_patients',
                        help='Defaults to the source run training subset; use all for all training patients.')
    parser.add_argument('--output_labels', action='append',
                        help='Override source run regression/classification outputs (repeatable).')
    parser.add_argument('--output_labels_types',
                        help='Output types, r/c, matching --output_labels. Defaults to source run.')
    parser.add_argument('--survival_task_json', action='append',
                        help='Override source survival tasks with a JSON task object (repeatable).')
    parser.add_argument('--loss_weights', action='append', type=float,
                        help='Repeat once per output; defaults to source run.')
    parser.add_argument('--batch_size', type=int, help='Studies per batch; defaults to source batch size.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--attention_heads', type=int, default=4)
    parser.add_argument('--attention_key_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'rmsprop'])
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--lr_schedule', choices=['constant', 'cosine'])
    parser.add_argument('--warmup_epochs', type=float)
    parser.add_argument('--warmup_start_frac', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--es_patience', type=int)
    parser.add_argument('--es_loss2monitor')
    parser.add_argument('--disable_survival_metrics_callback',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--run_validation_inference',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--output_dir', help='Base directory for this study-attention run.')
    return parser


if __name__ == '__main__':
    main(make_parser().parse_args())
