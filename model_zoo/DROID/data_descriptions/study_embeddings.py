"""Join DROID clip embeddings to wide-file labels and group them by study."""

import io
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from data_descriptions.wide_file import EcholabDataDescription
from echo_defines import category_dictionaries


def read_parquet(path):
    with tf.io.gfile.GFile(path, 'rb') as stream:
        return pd.read_parquet(io.BytesIO(stream.read()))


def embedding_files(directories):
    """Accept inference roots, inference_embeddings_* folders, or .pq files."""
    files = []
    for directory in directories:
        if directory.endswith('.pq'):
            matches = [directory] if tf.io.gfile.exists(directory) else []
        else:
            root = directory.rstrip('/')
            matches = tf.io.gfile.glob(f'{root}/prediction_*.pq')
            matches += tf.io.gfile.glob(f'{root}/inference_embeddings_*/prediction_*.pq')
        if not matches:
            raise FileNotFoundError(f'No embedding Parquet files under {directory}')
        files.extend(matches)
    return sorted(set(files))


def read_embeddings(paths):
    tables = [read_parquet(path) for path in paths]
    df = pd.concat(tables, ignore_index=True)
    columns = sorted(
        (c for c in df if c.startswith('embedding_') and c[10:].isdigit()),
        key=lambda c: int(c[10:]),
    )
    if not columns or columns != [f'embedding_{i}' for i in range(len(columns))]:
        raise ValueError('Embedding files need contiguous embedding_0...embedding_N columns.')
    if df['sample_id'].isna().any() or df['sample_id'].duplicated().any():
        raise ValueError('Embedding sample_id values must be present and unique across all files.')
    values = df[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Embedding files contain non-finite values.')
    return df[['sample_id', *columns]], columns


def select_source_views(wide, params):
    for name, column in (
        ('view', 'view_prediction'), ('doppler', 'doppler_prediction'),
        ('quality', 'quality_prediction'), ('canonical', 'canonical_prediction'),
    ):
        selected = params.get(f'selected_{name}') or params.get(f'selected_{name}s')
        if selected:
            allowed = [category_dictionaries[name][value] for value in selected]
            wide = wide.loc[wide[column].isin(allowed)]
    return wide.copy()


def join_embeddings_wide(embeddings, wide_files, params, output_labels, survival_tasks,
                         source_selected_path=None, output_scaling=None):
    tables = []
    for path in wide_files:
        table = read_parquet(path)
        table = select_source_views(table, params)
        if output_scaling and path != source_selected_path:
            for label, stats in output_scaling.items():
                if label in table:
                    table[label] = (table[label] - stats['mean']) / stats['std']
        tables.append(table)
    wide = pd.concat(tables, ignore_index=True)
    needed = list(dict.fromkeys(
        ['sample_id', *output_labels, *(
            column for task in survival_tasks
            for column in (task['event_column'], task['follow_up_days_column'])
        )],
    ))
    missing = sorted(set(needed) - set(wide))
    if missing:
        raise ValueError(f'Wide files are missing columns: {missing}')
    if wide['sample_id'].duplicated().any():
        raise ValueError('Wide files contain duplicate sample_id values.')
    wide = wide.dropna(subset=needed)
    for task in survival_tasks:
        events = pd.to_numeric(wide[task['event_column']], errors='raise')
        days = pd.to_numeric(wide[task['follow_up_days_column']], errors='raise')
        if not events.isin([0, 1]).all() or not np.isfinite(days).all():
            raise ValueError(f"Invalid event or follow-up values for {task['name']}.")
        keep = ~((events == 0) & (days < 0))
        if task['prevalent_policy'] == 'exclude':
            keep &= ~((events == 1) & (days <= task['blanking_days']))
        wide = wide.loc[keep].copy()
    missing_embeddings = set(wide['sample_id']) - set(embeddings['sample_id'])
    if missing_embeddings:
        examples = sorted(map(str, missing_embeddings))[:5]
        raise ValueError(
            f'{len(missing_embeddings)} eligible wide-file clips have no embedding; '
            f'check inference shard coverage. Examples: {examples}',
        )
    merged = embeddings.merge(wide, on='sample_id', how='inner', validate='one_to_one')
    if merged.empty:
        raise ValueError('No sample_id values match between embeddings and wide files.')
    logging.info('Matched %d of %d embeddings to eligible wide-file rows.', len(merged), len(embeddings))
    return merged


def _classification_index(mapping, value):
    if value in mapping:
        return mapping[value]
    if str(value) in mapping:
        return mapping[str(value)]
    try:
        numeric_value = float(value)
        matches = [index for key, index in mapping.items() if float(key) == numeric_value]
        if len(matches) == 1:
            return matches[0]
    except (TypeError, ValueError):
        pass
    raise ValueError(f'Classification value {value!r} is absent from the source run mapping.')


def make_study_records(merged, embedding_columns, run):
    """Return records with one label tuple and every eligible clip per study."""
    pieces = merged['sample_id'].astype(str).str.split('_', n=2, expand=True)
    if pieces.shape[1] != 3 or pieces.isna().any().any():
        raise ValueError('sample_id must have patient_study_view format.')
    merged = merged.copy()
    merged['patient_id'] = pd.to_numeric(pieces[0], errors='raise').astype(int)
    merged['study_id'] = pieces[0] + '_' + pieces[1]
    merged = merged.sort_values('sample_id')
    label_columns = list(dict.fromkeys([
        *run.output_labels,
        *(column for task in run.survival_tasks
          for column in (task['event_column'], task['follow_up_days_column'])),
    ]))
    records = []
    cls_names = run.head_spec['category_order'] or []
    for study_id, study in merged.groupby('study_id', sort=True):
        if any(study[column].nunique(dropna=False) != 1 for column in label_columns):
            raise ValueError(f'Inconsistent labels across clips in study {study_id}.')
        row = study.iloc[0]
        targets = []
        if run.n_regression:
            targets.append(row[run.output_labels[:run.n_regression]].to_numpy(dtype=np.float32))
        for name in cls_names:
            mapping = run.cls_category_map_dicts[name]
            one_hot = np.zeros(len(mapping), np.float32)
            one_hot[_classification_index(mapping, row[name])] = 1
            targets.append(one_hot)
        for task in run.survival_tasks:
            targets.append(EcholabDataDescription._survival_tensor_from_row(row, task))
        records.append({
            'study_id': study_id,
            'patient_id': int(row['patient_id']),
            'sample_ids': study['sample_id'].astype(str).tolist(),
            'embeddings': study[embedding_columns].to_numpy(dtype=np.float32),
            'targets': tuple(targets),
        })
    return records


def assign_splits(records, splits, n_train_patients='all'):
    train = list(splits['patient_train'])
    valid = list(splits['patient_valid'])
    if n_train_patients != 'all':
        value = float(n_train_patients)
        if 0 < value <= 1:
            train = train[:int(len(train) * value)]
            valid = valid[:int(len(valid) * value)]
        else:
            train = train[:int(value * 0.9)]
            valid = valid[:int(value * 0.1)]
    split_patients = {
        'train': set(map(int, train)), 'valid': set(map(int, valid)),
        'internal_test': set(map(int, splits.get('patient_internal_test', []))),
        'test': set(map(int, splits.get('patient_test', []))),
    }
    names = list(split_patients)
    if any(split_patients[names[i]] & split_patients[names[j]]
           for i in range(len(names)) for j in range(i + 1, len(names))):
        raise ValueError('Patient IDs overlap between train, validation, and test splits.')
    assigned = {name: [] for name in split_patients}
    for record in records:
        for name, patients in split_patients.items():
            if record['patient_id'] in patients:
                assigned[name].append(record)
                break
    return assigned


def make_study_dataset(records, embedding_dim, output_widths, batch_size, shuffle=False):
    """Pad to the longest study in each batch, retaining all clips and studies."""
    if not records:
        raise ValueError('Cannot make a dataset without studies.')
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    target_signature = tuple(tf.TensorSpec((width,), tf.float32) for width in output_widths)

    def generate():
        order = np.arange(len(records))
        if shuffle:
            np.random.default_rng().shuffle(order)
        for index in order:
            record = records[index]
            clips = record['embeddings']
            yield ({'embeddings': clips, 'mask': np.ones(len(clips), dtype=bool)},
                   tuple(record['targets']))

    dataset = tf.data.Dataset.from_generator(
        generate,
        output_signature=(
            {'embeddings': tf.TensorSpec((None, embedding_dim), tf.float32),
             'mask': tf.TensorSpec((None,), tf.bool)},
            target_signature,
        ),
    )
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            {'embeddings': [None, embedding_dim], 'mask': [None]},
            tuple([width] for width in output_widths),
        ),
        drop_remainder=False,
    )
    return dataset.prefetch(1)
