"""Generate a synthetic DROID-RV dataset for end-to-end pipeline testing.

Produces the three artifacts the DROID training/inference recipes require:

  1. wide_file.parquet  one row per video, with sample identifier, view/doppler/
                        quality/canonical predictions, and outcome columns.
  2. splits.json        patient_train / patient_valid / patient_test lists.
  3. lmdb/              one `{study_id}.lmdb` per study, keyed by video id, plus a
                        `log_{study_id}.pq` manifest, matching `echo_to_lmdb.py`.

The videos are random noise, not echocardiograms. The point is that every shape,
dtype, key and filename matches what `model_zoo/DROID/echo_supervised_training_recipe.py`
and `data_descriptions/echo.py` expect, so the pipeline can be exercised without
access to real data.

Example:
    python generate_synthetic_data.py --output_dir ./data
"""
import argparse
import json
import logging
import os
import sys
import tempfile

import cv2
import lmdb
import numpy as np
import pandas as pd

# `echo_defines` lives in the base DROID model zoo entry; reuse its category
# dictionaries rather than duplicating the view/doppler/quality/canonical codes.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'DROID'))
from echo_defines import category_dictionaries  # noqa: E402

# Views that satisfy the DROID-RV training recipe's `--selected_views` arguments.
IN_FILTER_VIEWS = ['A4C', 'RV_focused']

# Videos that fail the recipe's selection filter, so that filtering is actually
# exercised rather than being a no-op. Each entry violates at least one criterion.
OUT_OF_FILTER_METADATA = [
    {'view': 'PLAX', 'doppler': 'standard', 'quality': 'good', 'canonical': 'on_axis'},
    {'view': 'A2C', 'doppler': 'standard', 'quality': 'good', 'canonical': 'on_axis'},
    {'view': 'PSAX_AV', 'doppler': 'standard', 'quality': 'good', 'canonical': 'on_axis'},
    {'view': 'RV_inflow', 'doppler': 'standard', 'quality': 'good', 'canonical': 'on_axis'},
    {'view': 'A4C', 'doppler': 'doppler', 'quality': 'good', 'canonical': 'on_axis'},
    {'view': 'RV_focused', 'doppler': 'standard', 'quality': 'unusable', 'canonical': 'on_axis'},
    {'view': 'A4C', 'doppler': 'standard', 'quality': 'good', 'canonical': 'off_axis'},
    {'view': 'RV_focused', 'doppler': '3-D', 'quality': 'good', 'canonical': 'on_axis'},
]

FRAME_SIZE = (224, 224)
FPS = 30


def _random_ids(rng, n, low, high):
    """n unique integers drawn from [low, high)."""
    ids = set()
    while len(ids) < n:
        ids.update(rng.integers(low, high, size=n - len(ids)).tolist())
    return sorted(ids)


def _random_hex_ids(rng, n, n_chars=8):
    """n unique lowercase hex strings. No underscores: sample_id is split on '_'."""
    ids = set()
    while len(ids) < n:
        ids.add(''.join(rng.choice(list('0123456789abcdef'), size=n_chars)))
    return sorted(ids)


def build_patient_outcomes(rng, patient_ids):
    """One row of clinical outcomes per patient.

    Classification labels are assigned in a balanced, alternating fashion so that
    every class is guaranteed to appear (the training recipe errors out on
    constant-valued classification columns). Regression values are then drawn
    conditional on the class label, so the synthetic outcomes are mutually
    consistent rather than independent noise.
    """
    n = len(patient_ids)
    rows = []
    for i, patient_id in enumerate(patient_ids):
        sex = 'Female' if i % 2 == 0 else 'Male'
        rv_size = 'Dilated' if (i // 2) % 2 == 0 else 'Not Dilated'
        rv_function = 'Hypokinetic' if (i // 4) % 2 == 0 else 'Not Hypokinetic'

        # RV end-diastolic diameter (mm, the unit DROID-RV predicts), larger when
        # dilated.
        if rv_size == 'Dilated':
            rvedd = rng.normal(46.0, 5.0)
        else:
            rvedd = rng.normal(33.0, 4.0)

        # RV ejection fraction (%), lower when hypokinetic.
        if rv_function == 'Hypokinetic':
            rvef = rng.normal(35.0, 6.0)
        else:
            rvef = rng.normal(55.0, 5.0)

        # RV volumes (mL); end-systolic follows from EF.
        rvedv = rng.normal(140.0 if rv_size == 'Dilated' else 95.0, 20.0)
        rvedv = float(np.clip(rvedv, 40.0, 300.0))
        rvef = float(np.clip(rvef, 10.0, 75.0))
        rvesv = rvedv * (1.0 - rvef / 100.0)

        rows.append({
            'patient_id': patient_id,
            'age': round(float(np.clip(rng.normal(62.0, 14.0), 18.0, 95.0)), 1),
            'sex': sex,
            'rvedd': round(float(np.clip(rvedd, 15.0, 70.0)), 1),
            'rv_size': rv_size,
            'rv_function': rv_function,
            'rvef': round(rvef, 1),
            'rvedv': round(rvedv, 1),
            'rvesv': round(float(rvesv), 1),
        })
    assert len(rows) == n
    return pd.DataFrame(rows)


def build_metadata_df(rng, n_patients, videos_per_patient):
    """One row per video: identifiers, per-video metadata, per-patient outcomes.

    `sample_id` is `{patient_id}_{study_id}_{video_id}`, which is what
    `LmdbEchoStudyVideoDataDescription.get_raw_data` splits on to locate the LMDB
    and the key within it. One study per patient, so `study_id == patient_id`.

    Roughly 70% of videos are given metadata that passes the recipe's selection
    filter. Each patient gets at least three passing videos, so no patient drops
    out of the dataset entirely once filtering is applied.
    """
    patient_ids = _random_ids(rng, n_patients, 100_000, 1_000_000)
    outcomes = build_patient_outcomes(rng, patient_ids).set_index('patient_id')

    rows = []
    for i, patient_id in enumerate(patient_ids):
        study_id = patient_id
        video_ids = _random_hex_ids(rng, videos_per_patient)

        # Alternate 3 and 4 in-filter videos per patient to land near 70% overall
        # while guaranteeing a floor of 3 per patient.
        n_in_filter = min(videos_per_patient, 3 + (i % 2))

        for j, video_id in enumerate(video_ids):
            if j < n_in_filter:
                meta = {
                    'view': str(rng.choice(IN_FILTER_VIEWS)),
                    'doppler': 'standard',
                    'quality': 'good',
                    'canonical': 'on_axis',
                }
            else:
                meta = OUT_OF_FILTER_METADATA[rng.integers(len(OUT_OF_FILTER_METADATA))]

            rows.append({
                'sample_id': f'{patient_id}_{study_id}_{video_id}',
                'patient_id': patient_id,
                'study_id': study_id,
                'video_id': video_id,
                'view_prediction': category_dictionaries['view'][meta['view']],
                'doppler_prediction': category_dictionaries['doppler'][meta['doppler']],
                'quality_prediction': category_dictionaries['quality'][meta['quality']],
                'canonical_prediction': category_dictionaries['canonical'][meta['canonical']],
                **outcomes.loc[patient_id].to_dict(),
            })

    return pd.DataFrame(rows)


def build_splits(rng, patient_ids, test_frac=0.3, valid_frac_of_train=0.15):
    """Patient-level train/valid/test split.

    The recipe requires all three keys and builds a validation dataset from
    `patient_valid`, so the validation patients are carved out of the training
    pool rather than left empty. Patient ids must be JSON ints: the recipe tests
    membership with `int(sample_id.split('_')[0]) in patient_train`.
    """
    shuffled = list(patient_ids)
    rng.shuffle(shuffled)

    n_test = int(round(test_frac * len(shuffled)))
    test = shuffled[:n_test]
    train_pool = shuffled[n_test:]

    n_valid = max(1, int(round(valid_frac_of_train * len(train_pool))))
    valid = train_pool[:n_valid]
    train = train_pool[n_valid:]

    return {
        'patient_train': [int(p) for p in train],
        'patient_valid': [int(p) for p in valid],
        'patient_test': [int(p) for p in test],
    }


def random_video_bytes(rng, tmp_dir, n_frames):
    """Encode `n_frames` of random noise as an MJPG avi and return its bytes.

    Uses the same encoding path as `array_to_cropped_avi` in
    `model_zoo/DROID/echo_to_lmdb.py` (cv2.VideoWriter, MJPG fourcc, .avi) so the
    bytes decode through `av.open` exactly like real stored echoes do.

    The noise is generated at low resolution and upsampled before fine noise is
    added, which keeps JPEG-compressed frames small; full-resolution white noise
    is nearly incompressible and would bloat the LMDBs.
    """
    video_path = os.path.join(tmp_dir, 'video.avi')
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS, FRAME_SIZE)
    if not writer.isOpened():
        raise RuntimeError('cv2.VideoWriter could not be opened for the MJPG codec')

    for _ in range(n_frames):
        coarse = rng.integers(0, 256, size=(56, 56), dtype=np.uint8)
        frame = cv2.resize(coarse, FRAME_SIZE, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        frame += rng.normal(0.0, 12.0, size=FRAME_SIZE)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        # Stored echoes are grayscale replicated across three channels.
        writer.write(cv2.merge([frame, frame, frame]))
    writer.release()

    with open(video_path, 'rb') as avi:
        video_bytes = avi.read()
    os.remove(video_path)

    if not video_bytes:
        raise RuntimeError(f'cv2 wrote an empty avi to {video_path}')
    return video_bytes


def write_lmdbs(rng, df, lmdb_folder, n_frames):
    """One LMDB per study, keyed by video id, plus a `log_{study_id}.pq` manifest.

    The manifest column is named `view` even though it holds video ids; that is
    the convention `echo_to_lmdb.py` established and `get_loading_options` reads.
    """
    os.makedirs(lmdb_folder, exist_ok=True)
    n_videos = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        for study_id, study_df in df.groupby('study_id', sort=True):
            study_lmdb = os.path.join(lmdb_folder, f'{study_id}.lmdb')
            env = lmdb.open(study_lmdb, map_size=2 ** 32 - 1)
            log_dic = {'study': [], 'view': [], 'log': [], 'stored': []}
            with env.begin(write=True) as txn:
                for video_id in study_df['video_id']:
                    txn.put(
                        key=video_id.encode('utf-8'),
                        value=random_video_bytes(rng, tmp_dir, n_frames),
                    )
                    log_dic['study'].append(study_id)
                    log_dic['view'].append(video_id)
                    log_dic['log'].append('')
                    log_dic['stored'].append(True)
                    n_videos += 1
            env.close()
            pd.DataFrame(log_dic).to_parquet(os.path.join(study_lmdb, f'log_{study_id}.pq'))
            logging.info(f'Wrote {len(study_df)} videos to {study_lmdb}')

    return n_videos


def main(n_patients, videos_per_patient, n_frames, output_dir, seed):
    rng = np.random.default_rng(seed)

    df = build_metadata_df(rng, n_patients, videos_per_patient)
    splits = build_splits(rng, df['patient_id'].unique())

    os.makedirs(output_dir, exist_ok=True)
    wide_path = os.path.join(output_dir, 'wide_file.parquet')
    splits_path = os.path.join(output_dir, 'splits.json')
    lmdb_folder = os.path.join(output_dir, 'lmdb')

    df.to_parquet(wide_path)
    with open(splits_path, 'w') as json_file:
        json.dump(splits, json_file, indent=2)
    n_videos = write_lmdbs(rng, df, lmdb_folder, n_frames)

    in_filter = df[
        df['view_prediction'].isin([category_dictionaries['view'][v] for v in IN_FILTER_VIEWS])
        & (df['doppler_prediction'] == category_dictionaries['doppler']['standard'])
        & (df['quality_prediction'] == category_dictionaries['quality']['good'])
        & (df['canonical_prediction'] == category_dictionaries['canonical']['on_axis'])
    ]

    print(f"""
Synthetic DROID-RV dataset written to {os.path.abspath(output_dir)}

  wide file    {wide_path}
               {len(df)} rows ({n_patients} patients x {videos_per_patient} videos)
               {len(in_filter)} rows pass the A4C/RV_focused + standard + good + on_axis filter
  splits file  {splits_path}
               train={len(splits['patient_train'])} valid={len(splits['patient_valid'])} test={len(splits['patient_test'])} patients
  lmdb folder  {lmdb_folder}
               {n_patients} study LMDBs, {n_videos} videos of {n_frames} frames at 224x224x3

Outcome columns: age (r), sex (c), rvedd (r), rv_size (c), rv_function (c), rvef (r), rvedv (r), rvesv (r)

Next: python validate_synthetic_data.py --data_dir {output_dir}
""")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_patients', type=int, default=20, help='One study per patient.')
    parser.add_argument('--videos_per_patient', type=int, default=5)
    parser.add_argument(
        '--n_frames', type=int, default=16,
        help='Frames per video. Must be at least n_input_frames * skip_modulo to '
             'avoid the loader cycling back to the first frame.',
    )
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()

    main(
        n_patients=args.n_patients,
        videos_per_patient=args.videos_per_patient,
        n_frames=args.n_frames,
        output_dir=args.output_dir,
        seed=args.seed,
    )
