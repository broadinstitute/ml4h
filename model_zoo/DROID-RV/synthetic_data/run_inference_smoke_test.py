"""Run the real DROID-RV and DROID-RVEF checkpoints over the synthetic dataset.

Loads videos out of the synthetic LMDBs through the same
`LmdbEchoStudyVideoDataDescription` loader the DROID recipes use, runs them through
both published checkpoints, rescales the regression heads back into physical units,
and writes predictions to parquet.

The synthetic videos are random noise, so the predictions are meaningless as
measurements. What this tests is that the stored data is loadable, the checkpoints
restore into the expected head layout, and a forward pass produces finite,
correctly shaped outputs in a plausible range.

Requires the git-lfs checkpoints:
    git lfs pull --include model_zoo/DROID-RV/droid_rv_checkpoint/*
    git lfs pull --include model_zoo/DROID-RV/droid_rvef_checkpoint/*
    git lfs pull --include model_zoo/DROID-RV/movinet_a2_base/*

Example:
    python run_inference_smoke_test.py --data_dir ./data --split test
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
DROID_RV_DIR = os.path.join(HERE, '..')
DROID_DIR = os.path.join(HERE, '..', '..', 'DROID')
sys.path.append(DROID_RV_DIR)
sys.path.append(DROID_DIR)

from data_descriptions.echo import LmdbEchoStudyVideoDataDescription  # noqa: E402
from droid_rv_model_description import (  # noqa: E402
    DROID_RV_REGRESSION_SCALING,
    DROID_RVEF_REGRESSION_SCALING,
    create_movinet_classifier,
    create_regressor_classifier,
    rescale_droid_rv_outputs,
    rescale_droid_rvef_outputs,
)
from echo_defines import category_dictionaries  # noqa: E402
from validate_synthetic_data import Checker  # noqa: E402

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

SELECTED_VIEWS = ['A4C', 'RV_focused']

# Head layouts, taken from the readme's documented model outputs. `classes` lists
# the class names in the order the softmax units were trained, which is the
# alphabetical ordering the training recipe assigns.
DROID_RV_SPEC = {
    'name': 'DROID-RV',
    'checkpoint': 'droid_rv_checkpoint/chkp',
    'regression': DROID_RV_REGRESSION_SCALING,
    'categories': {'RV_size': 2, 'RV_function': 2, 'Sex': 2},
    'category_order': ['RV_size', 'RV_function', 'Sex'],
    'classes': {
        'RV_size': ['Dilated', 'Not Dilated'],
        'RV_function': ['Hypokinetic', 'Not Hypokinetic'],
        'Sex': ['Female', 'Male'],
    },
    'rescale': rescale_droid_rv_outputs,
    # Generous physiologic windows: a correctly restored checkpoint predicts
    # within these even on noise, whereas a failed weight load does not.
    'plausible': {'Age': (0.0, 120.0), 'RVEDD': (5.0, 90.0)},
}

DROID_RVEF_SPEC = {
    'name': 'DROID-RVEF',
    'checkpoint': 'droid_rvef_checkpoint/chkp',
    'regression': DROID_RVEF_REGRESSION_SCALING,
    'categories': {'Sex': 2},
    'category_order': ['Sex'],
    'classes': {'Sex': ['Female', 'Male']},
    'rescale': rescale_droid_rvef_outputs,
    'plausible': {
        'RVEF': (-10.0, 110.0),
        'RV End-Diastolic Volume': (0.0, 500.0),
        'RV End-Systolic Volume': (0.0, 400.0),
        'Age': (0.0, 120.0),
    },
}


def select_sample_ids(data_dir, split, n_samples):
    """Filtered, split-restricted sample ids, mirroring the recipes' selection."""
    wide_df = pd.read_parquet(os.path.join(data_dir, 'wide_file.parquet'))
    with open(os.path.join(data_dir, 'splits.json'), 'r') as json_file:
        splits = json.load(json_file)

    selected = wide_df[
        wide_df['view_prediction'].isin([category_dictionaries['view'][v] for v in SELECTED_VIEWS])
        & (wide_df['doppler_prediction'] == category_dictionaries['doppler']['standard'])
        & (wide_df['quality_prediction'] == category_dictionaries['quality']['good'])
        & (wide_df['canonical_prediction'] == category_dictionaries['canonical']['on_axis'])
    ]

    if split == 'all':
        patients = splits['patient_train'] + splits['patient_valid'] + splits['patient_test']
    else:
        patients = splits[f'patient_{split}']

    sample_ids = sorted(
        s for s in selected['sample_id'].tolist() if int(s.split('_')[0]) in patients
    )
    if not sample_ids:
        raise ValueError(f'No samples in split "{split}" survive view/doppler/quality/canonical filtering')

    if n_samples and n_samples < len(sample_ids):
        logging.info(f'Limiting to the first {n_samples} of {len(sample_ids)} samples')
        sample_ids = sample_ids[:n_samples]
    return sample_ids


def load_videos(input_dd, sample_ids):
    """Stack videos into (n, n_input_frames, 224, 224, 3) float32."""
    videos = []
    for i, sample_id in enumerate(sample_ids, start=1):
        videos.append(input_dd.get_raw_data(sample_id))
        if i % 10 == 0 or i == len(sample_ids):
            logging.info(f'Loaded {i}/{len(sample_ids)} videos')
    return np.stack(videos).astype(np.float32)


def build_model(spec, encoder, n_input_frames):
    model = create_regressor_classifier(
        encoder,
        input_shape=(n_input_frames, 224, 224, 3),
        n_output_features=len(spec['regression']),
        categories=spec['categories'],
        category_order=spec['category_order'],
    )
    checkpoint = os.path.join(DROID_RV_DIR, spec['checkpoint'])
    if not tf.io.gfile.glob(f'{checkpoint}.index'):
        raise FileNotFoundError(
            f'{checkpoint}.index not found. Pull it with: '
            f'git lfs pull --include model_zoo/DROID-RV/{os.path.dirname(spec["checkpoint"])}/*'
        )
    model.load_weights(checkpoint)
    return model


def predict(model, videos, batch_size):
    """Batched forward pass, returning a list of arrays, one per output head."""
    batches = []
    for start in range(0, len(videos), batch_size):
        batch = model.predict(videos[start:start + batch_size], verbose=0)
        # A single-head model returns one array rather than a list.
        batches.append(batch if isinstance(batch, list) else [batch])
    return [np.concatenate([b[head] for b in batches], axis=0) for head in range(len(batches[0]))]


def run_model(checker, spec, encoder, videos, sample_ids, n_input_frames, batch_size, output_dir):
    print(f'\n{spec["name"]}')
    model = build_model(spec, encoder, n_input_frames)
    outputs = predict(model, videos, batch_size)

    n_heads = 1 + len(spec['category_order'])
    if not checker.check(
        len(outputs) == n_heads,
        f'{spec["name"]}: model produced {n_heads} output heads',
        f'got {len(outputs)}',
    ):
        return None

    checker.check(
        outputs[0].shape == (len(sample_ids), len(spec['regression'])),
        f'{spec["name"]}: regression head shape is ({len(sample_ids)}, {len(spec["regression"])})',
        f'got {outputs[0].shape}',
    )
    checker.check(
        all(np.isfinite(o).all() for o in outputs),
        f'{spec["name"]}: all outputs are finite',
    )

    # Rescale in place, so this must run after the raw-output checks above.
    outputs = spec['rescale'](outputs)

    df = pd.DataFrame({'sample_id': sample_ids})
    for i, (label, _, _) in enumerate(spec['regression']):
        values = outputs[0][:, i]
        df[label] = values
        low, high = spec['plausible'][label]
        checker.check(
            bool(((values >= low) & (values <= high)).all()),
            f'{spec["name"]}: {label} within [{low}, {high}]',
            f'range [{values.min():.1f}, {values.max():.1f}]',
        )
        print(f'  INFO  {label}: mean {values.mean():.1f}, range [{values.min():.1f}, {values.max():.1f}]')

    for head, category in enumerate(spec['category_order'], start=1):
        probs = outputs[head]
        class_names = spec['classes'][category]
        checker.check(
            probs.shape == (len(sample_ids), len(class_names)),
            f'{spec["name"]}: {category} head shape is ({len(sample_ids)}, {len(class_names)})',
            f'got {probs.shape}',
        )
        checker.check(
            np.allclose(probs.sum(axis=1), 1.0, atol=1e-4),
            f'{spec["name"]}: {category} probabilities sum to 1',
        )
        predicted = [class_names[i] for i in probs.argmax(axis=1)]
        df[category] = predicted
        for i, class_name in enumerate(class_names):
            df[f'{category}_p_{class_name}'] = probs[:, i]
        counts = pd.Series(predicted).value_counts().to_dict()
        print(f'  INFO  {category}: {counts}')

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'predictions_{spec["name"].lower().replace("-", "_")}.pq')
    df.to_parquet(out_path)
    print(f'  INFO  wrote {out_path}')
    return df


def main(data_dir, split, n_samples, batch_size, n_input_frames, skip_modulo, movinet_chkp_dir, output_dir, models):
    checker = Checker()
    sample_ids = select_sample_ids(data_dir, split, n_samples)
    print(f'Running inference on {len(sample_ids)} samples from split "{split}"')

    input_dd = LmdbEchoStudyVideoDataDescription(
        os.path.join(data_dir, 'lmdb'), 'image', [], n_input_frames, skip_modulo, randomize_start_frame=False,
    )
    videos = load_videos(input_dd, sample_ids)
    checker.check(
        videos.shape == (len(sample_ids), n_input_frames, 224, 224, 3),
        f'loaded videos have shape ({len(sample_ids)}, {n_input_frames}, 224, 224, 3)',
        f'got {videos.shape}',
    )
    checker.check(
        float(videos.min()) >= 0.0 and float(videos.max()) <= 1.0,
        'loaded videos are scaled to [0, 1]',
        f'range [{videos.min():.3f}, {videos.max():.3f}]',
    )

    # One shared MoViNet encoder feeds both heads, as in droid_rv_inference.py.
    _, backbone = create_movinet_classifier(
        n_input_frames=n_input_frames,
        batch_size=batch_size,
        num_classes=600,
        checkpoint_dir=movinet_chkp_dir,
    )
    backbone_output = backbone.layers[-1].output[0]
    flatten = tf.keras.layers.Flatten()(backbone_output)
    encoder = tf.keras.Model(inputs=[backbone.input], outputs=[flatten])

    specs = {'rv': [DROID_RV_SPEC], 'rvef': [DROID_RVEF_SPEC], 'both': [DROID_RV_SPEC, DROID_RVEF_SPEC]}[models]
    for spec in specs:
        run_model(checker, spec, encoder, videos, sample_ids, n_input_frames, batch_size, output_dir)

    if checker.failures:
        print(f'\n{len(checker.failures)} check(s) FAILED:')
        for failure in checker.failures:
            print(f'  - {failure}')
        return 1
    print('\nAll inference checks passed. Predictions are on random noise and are not meaningful measurements.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test', 'all'])
    parser.add_argument('--models', type=str, default='both', choices=['rv', 'rvef', 'both'])
    parser.add_argument('--n_samples', type=int, default=0, help='0 runs every sample in the split.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_input_frames', type=int, default=16)
    parser.add_argument('--skip_modulo', type=int, default=1)
    parser.add_argument('--movinet_chkp_dir', type=str, default=os.path.join(DROID_RV_DIR, 'movinet_a2_base'))
    parser.add_argument('--output_dir', type=str, default='./inference_smoke_test_output')
    args = parser.parse_args()

    sys.exit(main(
        data_dir=args.data_dir,
        split=args.split,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        n_input_frames=args.n_input_frames,
        skip_modulo=args.skip_modulo,
        movinet_chkp_dir=args.movinet_chkp_dir,
        output_dir=args.output_dir,
        models=args.models,
    ))
