"""Validate a synthetic DROID-RV dataset against the format the recipes expect.

Checks the wide file columns, the splits file, and that every LMDB entry decodes
back into a video of the expected shape. Also replays the recipe's own filtering
and patient-partitioning logic to confirm the train/valid sets are non-empty at a
given batch size, so a failure here is caught before spending GPU time.

Does not require TensorFlow or the MoViNet checkpoints. If `ml4ht` is installed,
the LMDB round-trip additionally runs through the real
`LmdbEchoStudyVideoDataDescription` loader.

Example:
    python validate_synthetic_data.py --data_dir ./data
"""
import argparse
import io
import itertools
import json
import os
import sys

import av
import lmdb
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'DROID'))
from echo_defines import category_dictionaries  # noqa: E402

REQUIRED_METADATA_COLUMNS = [
    'sample_id',
    'view_prediction',
    'doppler_prediction',
    'quality_prediction',
    'canonical_prediction',
]
OUTCOME_COLUMNS = ['age', 'sex', 'rvedd', 'rv_size', 'rv_function', 'rvef', 'rvedv', 'rvesv']
CLASSIFICATION_COLUMNS = ['sex', 'rv_size', 'rv_function']

SELECTED_VIEWS = ['A4C', 'RV_focused']


class Checker:
    """Collects pass/fail results so every check runs before we report."""

    def __init__(self):
        self.failures = []

    def check(self, condition, description, detail=''):
        if condition:
            print(f'  PASS  {description}')
        else:
            print(f'  FAIL  {description}' + (f' -- {detail}' if detail else ''))
            self.failures.append(description)
        return bool(condition)


def validate_wide_file(checker, wide_path):
    print(f'\nWide file: {wide_path}')
    df = pd.read_parquet(wide_path)

    missing = [c for c in REQUIRED_METADATA_COLUMNS + OUTCOME_COLUMNS if c not in df.columns]
    checker.check(not missing, 'all required columns present', f'missing {missing}')
    print(f'  INFO  {len(df)} rows, {df["sample_id"].nunique()} unique sample_ids')

    checker.check(df['sample_id'].is_unique, 'sample_id is unique')
    checker.check(
        not df[OUTCOME_COLUMNS].isna().any().any(),
        'no nulls in outcome columns',
        f'{df[OUTCOME_COLUMNS].isna().sum().to_dict()}',
    )

    # `LmdbEchoStudyVideoDataDescription.get_raw_data` does
    # `_, study, view = sample_id.split('_')`, so exactly three parts are required.
    parts = df['sample_id'].str.split('_')
    checker.check((parts.str.len() == 3).all(), "sample_id splits into exactly 3 '_'-delimited parts")

    # The recipe partitions with `int(sample_id.split('_')[0]) in patient_train`.
    try:
        parts.str[0].astype(int)
        patient_ids_are_ints = True
    except ValueError:
        patient_ids_are_ints = False
    checker.check(patient_ids_are_ints, 'sample_id patient prefix parses as int')

    for column, dictionary in [
        ('view_prediction', 'view'),
        ('doppler_prediction', 'doppler'),
        ('quality_prediction', 'quality'),
        ('canonical_prediction', 'canonical'),
    ]:
        valid_codes = set(category_dictionaries[dictionary].values())
        checker.check(
            set(df[column].unique()).issubset(valid_codes),
            f'{column} values are valid {dictionary} codes',
            f'saw {sorted(set(df[column].unique()) - valid_codes)}',
        )

    for column in CLASSIFICATION_COLUMNS:
        n_classes = df[column].nunique()
        checker.check(n_classes >= 2, f'{column} has >= 2 classes', f'{n_classes} class(es)')
        print(f'  INFO  {column}: {df[column].value_counts().to_dict()}')

    return df


def validate_splits(checker, splits_path, df):
    print(f'\nSplits file: {splits_path}')
    with open(splits_path, 'r') as json_file:
        splits = json.load(json_file)

    checker.check(
        all(k in splits for k in ['patient_train', 'patient_valid', 'patient_test']),
        'patient_train / patient_valid / patient_test keys present',
        f'saw {sorted(splits.keys())}',
    )

    train, valid, test = splits['patient_train'], splits['patient_valid'], splits['patient_test']
    print(f'  INFO  train={len(train)} valid={len(valid)} test={len(test)} patients')

    checker.check(
        all(isinstance(p, int) for p in train + valid + test),
        'all patient ids are JSON ints',
    )
    checker.check(len(valid) > 0, 'patient_valid is non-empty')

    sets = {'train': set(train), 'valid': set(valid), 'test': set(test)}
    for a, b in itertools.combinations(sets, 2):
        overlap = sets[a] & sets[b]
        checker.check(not overlap, f'{a} and {b} do not overlap', f'{len(overlap)} shared patients')

    wide_patients = set(df['sample_id'].str.split('_').str[0].astype(int))
    all_split_patients = sets['train'] | sets['valid'] | sets['test']
    checker.check(
        all_split_patients == wide_patients,
        'splits cover exactly the patients in the wide file',
        f'{len(wide_patients - all_split_patients)} unsplit, '
        f'{len(all_split_patients - wide_patients)} not in wide file',
    )
    return splits


def validate_recipe_selection(checker, df, splits, batch_size):
    """Replay the recipe's filtering and partitioning to confirm usable batches."""
    print(f'\nRecipe selection (batch_size={batch_size}):')
    selected = df[
        df['view_prediction'].isin([category_dictionaries['view'][v] for v in SELECTED_VIEWS])
        & (df['doppler_prediction'] == category_dictionaries['doppler']['standard'])
        & (df['quality_prediction'] == category_dictionaries['quality']['good'])
        & (df['canonical_prediction'] == category_dictionaries['canonical']['on_axis'])
    ].dropna(subset=OUTCOME_COLUMNS)

    working_ids = selected['sample_id'].tolist()
    print(f'  INFO  {len(working_ids)} of {len(df)} rows survive view/doppler/quality/canonical filtering')

    train_ids = [t for t in working_ids if int(t.split('_')[0]) in splits['patient_train']]
    valid_ids = [t for t in working_ids if int(t.split('_')[0]) in splits['patient_valid']]
    print(f'  INFO  train_ids={len(train_ids)} valid_ids={len(valid_ids)}')

    # The recipe batches with drop_remainder=True, so a split smaller than one
    # batch silently yields zero steps and training fails on an empty dataset.
    checker.check(
        len(working_ids) // batch_size > 0,
        f'training set yields >= 1 batch of {batch_size}',
        f'{len(working_ids)} rows',
    )
    checker.check(
        len(valid_ids) // batch_size > 0,
        f'validation set yields >= 1 batch of {batch_size}',
        f'{len(valid_ids)} rows -- lower --batch_size or raise --n_patients',
    )
    return selected


def decode_lmdb_video(lmdb_folder, video_id):
    """Decode a stored video the way `LmdbEchoStudyVideoDataDescription` does.

    Uses `to_ndarray` rather than the loader's `np.array(frame.to_image())` so this
    check does not require Pillow; the two are equivalent for an rgb24 frame. The
    loader's exact PIL-based path is covered by `validate_real_loader`.
    """
    env = lmdb.open(lmdb_folder, readonly=True, lock=False)
    with env.begin(buffers=True) as txn:
        raw = txn.get(video_id.encode('utf-8'))
        if raw is None:
            env.close()
            return None
        container = av.open(io.BytesIO(bytes(raw)), metadata_errors='ignore')
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        container.close()
    env.close()
    return frames


def validate_lmdbs(checker, df, lmdb_folder, n_frames, sample_studies):
    print(f'\nLMDB folder: {lmdb_folder}')
    study_ids = sorted(df['study_id'].unique())

    missing_dirs = [s for s in study_ids if not os.path.isfile(os.path.join(lmdb_folder, f'{s}.lmdb', 'data.mdb'))]
    checker.check(not missing_dirs, f'all {len(study_ids)} study LMDBs exist', f'missing {missing_dirs[:5]}')

    manifest_problems = []
    for study_id in study_ids:
        log_path = os.path.join(lmdb_folder, f'{study_id}.lmdb', f'log_{study_id}.pq')
        if not os.path.isfile(log_path):
            manifest_problems.append(f'{study_id}: no log parquet')
            continue
        log = pd.read_parquet(log_path)
        expected = set(df[df['study_id'] == study_id]['video_id'])
        stored = set(log[log['stored']]['view'])
        if stored != expected:
            manifest_problems.append(f'{study_id}: manifest lists {sorted(stored)}, wide file has {sorted(expected)}')
    checker.check(
        not manifest_problems,
        'every log_{study}.pq lists exactly the wide file videos as stored=True',
        '; '.join(manifest_problems[:3]),
    )

    # Decode every video in a few studies rather than one video overall: a codec
    # or key-encoding problem would otherwise be easy to miss.
    checked_studies = study_ids[:sample_studies]
    print(f'  INFO  decoding all videos in {len(checked_studies)} of {len(study_ids)} studies')
    decode_problems = []
    n_decoded = 0
    for study_id in checked_studies:
        for video_id in df[df['study_id'] == study_id]['video_id']:
            frames = decode_lmdb_video(os.path.join(lmdb_folder, f'{study_id}.lmdb'), video_id)
            if frames is None:
                decode_problems.append(f'{study_id}/{video_id}: key missing from LMDB')
            elif len(frames) != n_frames:
                decode_problems.append(f'{study_id}/{video_id}: decoded {len(frames)} frames, expected {n_frames}')
            elif frames[0].shape != (224, 224, 3):
                decode_problems.append(f'{study_id}/{video_id}: frame shape {frames[0].shape}, expected (224, 224, 3)')
            else:
                n_decoded += 1
    checker.check(
        not decode_problems,
        f'all {n_decoded} sampled videos decode to {n_frames}x224x224x3',
        '; '.join(decode_problems[:3]),
    )


def validate_real_loader(checker, df, lmdb_folder, n_input_frames, skip_modulo):
    """Run one sample through the actual DROID loader, if ml4ht is importable."""
    print('\nReal loader (data_descriptions.echo.LmdbEchoStudyVideoDataDescription):')
    try:
        from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
    except ImportError as e:
        print(f'  SKIP  ml4ht not available in this environment ({e})')
        return

    dd = LmdbEchoStudyVideoDataDescription(
        lmdb_folder, 'image', [], n_input_frames, skip_modulo, randomize_start_frame=False,
    )
    sample_id = df['sample_id'].iloc[0]
    tensor = dd.get_raw_data(sample_id)
    expected = (n_input_frames, 224, 224, 3)
    checker.check(
        tensor.shape == expected,
        f'loader returns {expected} for {sample_id}',
        f'got {tensor.shape}',
    )
    checker.check(
        tensor.dtype == np.float32 and 0.0 <= tensor.min() and tensor.max() <= 1.0,
        'loader output is float32 scaled to [0, 1]',
        f'dtype={tensor.dtype} range=[{tensor.min():.3f}, {tensor.max():.3f}]',
    )

    # get_loading_options reads log_{study}.pq, which the training recipe does not
    # exercise but the inference recipe path does.
    checker.check(
        len(dd.get_loading_options(sample_id)) == 1,
        'get_loading_options resolves the video via log_{study}.pq',
    )


def main(data_dir, n_frames, n_input_frames, skip_modulo, batch_size, sample_studies):
    checker = Checker()
    df = validate_wide_file(checker, os.path.join(data_dir, 'wide_file.parquet'))
    splits = validate_splits(checker, os.path.join(data_dir, 'splits.json'), df)
    validate_recipe_selection(checker, df, splits, batch_size)

    lmdb_folder = os.path.join(data_dir, 'lmdb')
    validate_lmdbs(checker, df, lmdb_folder, n_frames, sample_studies)
    validate_real_loader(checker, df, lmdb_folder, n_input_frames, skip_modulo)

    if checker.failures:
        print(f'\n{len(checker.failures)} check(s) FAILED:')
        for failure in checker.failures:
            print(f'  - {failure}')
        return 1
    print('\nAll checks passed.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--n_frames', type=int, default=16, help='Frames per generated video.')
    parser.add_argument('--n_input_frames', type=int, default=16, help='Frames the model consumes.')
    parser.add_argument('--skip_modulo', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_studies', type=int, default=3, help='Studies to fully decode.')
    args = parser.parse_args()

    sys.exit(main(
        data_dir=args.data_dir,
        n_frames=args.n_frames,
        n_input_frames=args.n_input_frames,
        skip_modulo=args.skip_modulo,
        batch_size=args.batch_size,
        sample_studies=args.sample_studies,
    ))
