import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from echo_defines import category_dictionaries
from model_descriptions.echo import DDGenerator, create_movinet_classifier, create_regressor

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)


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
        split_idx,
        n_splits,
        batch_size,
        skip_modulo,
        lmdb_folder,
        pretrained_ckpt_dir,
        movinet_ckpt_dir,
        output_dir,
        extract_embeddings,
        start_beat
):
    # Hide devices based on split
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([physical_devices[split_idx % 4]], 'GPU')

    wide_df = pd.read_parquet(wide_file)

    # Select only view(s) of interest
    selected_views_idx = [category_dictionaries['view'][v] for v in selected_views]
    selected_doppler_idx = [category_dictionaries['doppler'][v] for v in selected_doppler]
    selected_quality_idx = [category_dictionaries['quality'][v] for v in selected_quality]
    selected_canonical_idx = [category_dictionaries['canonical'][v] for v in selected_canonical]
    wide_df_selected = wide_df[
        (wide_df['view_prediction'].isin(selected_views_idx)) &
        (wide_df['doppler_prediction'].isin(selected_doppler_idx)) &
        (wide_df['quality_prediction'].isin(selected_quality_idx)) &
        (wide_df['canonical_prediction'].isin(selected_canonical_idx))
        ]

    # Fill entries without measurements and get all sample_ids
    for olabel in output_labels:
        wide_df_selected.loc[wide_df_selected[olabel].isna(), olabel] = -1
    working_ids = wide_df_selected['sample_id'].values.tolist()

    # Read splits and partition dataset
    with open(splits_file, 'r') as json_file:
        splits = json.load(json_file)

    patient_train = splits['patient_train']
    patient_valid = splits['patient_valid']

    if n_train_patients != 'all':
        patient_train = patient_train[:int(int(n_train_patients) * 0.9)]
        patient_valid = patient_valid[:int(int(n_train_patients) * 0.1)]

    if 'trainvalid' in lmdb_folder:
        patient_inference = patient_train + patient_valid
        if 'patient_internal_test' in splits:
            patient_inference = patient_inference + splits['patient_internal_test']
    else:
        patient_inference = splits['patient_test']

    inference_ids = sorted([t for t in working_ids if int(t.split('_')[0]) in patient_inference])

    INPUT_DD = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo,
        start_beat=start_beat
    )

    inference_ids_split = np.array_split(inference_ids, n_splits)[split_idx]
    body_inference_ids = tf.data.Dataset.from_tensor_slices(inference_ids_split).batch(batch_size, drop_remainder=False)
    n_inference_steps = len(inference_ids_split) // batch_size + int((len(inference_ids_split) % batch_size) > 0.5)

    io_inference_ds = body_inference_ids.interleave(
        lambda sample_ids: tf.data.Dataset.from_generator(
            DDGenerator(
                INPUT_DD,
                None
            ),
            output_signature=(
                tf.TensorSpec(shape=(None, n_input_frames, 224, 224, 3), dtype=tf.float32),
            ),
            args=(sample_ids,)
        )
    )

    model, backbone = create_movinet_classifier(
        n_input_frames,
        batch_size,
        num_classes=600,
        checkpoint_dir=movinet_ckpt_dir,
    )

    backbone_output = backbone.layers[-1].output[0]
    flatten = tf.keras.layers.Flatten()(backbone_output)
    encoder = tf.keras.Model(inputs=[backbone.input], outputs=[flatten])
    model_plus_head = create_regressor(
        encoder,
        input_shape=(n_input_frames, 224, 224, 3),
        n_output_features=len(output_labels)
    )
    model_plus_head.load_weights(pretrained_ckpt_dir)

    vois = '_'.join(selected_views)
    ufm = 'conv7'
    if extract_embeddings:
        output_folder = os.path.join(output_dir,
                                     f'inference_embeddings_{vois}_{ufm}_{lmdb_folder.split("/")[-1]}_{splits_file}_{start_beat}')
    else:
        output_folder = os.path.join(output_dir,
                                     f'inference_{vois}_{ufm}_{lmdb_folder.split("/")[-1]}_{splits_file}_{start_beat}')
    os.makedirs(output_folder, exist_ok=True)

    if extract_embeddings:
        embeddings = encoder.predict(io_inference_ds, steps=n_inference_steps, verbose=1)
        df = pd.DataFrame()
        df['sample_id'] = inference_ids_split
        for j, _ in enumerate(range(embeddings.shape[1])):
            df[f'embedding_{j}'] = embeddings[:, j]

        df.to_parquet(os.path.join(output_folder, f'prediction_{split_idx}.pq'))
    else:
        predictions = model_plus_head.predict(io_inference_ds, steps=n_inference_steps, verbose=1)
        df = pd.DataFrame()
        df['sample_id'] = inference_ids_split
        for j, _ in enumerate(range(predictions.shape[1])):
            df[f'prediction_{j}'] = predictions[:, j]

        df.to_parquet(os.path.join(output_folder, f'prediction_{split_idx}.pq'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_input_frames', type=int, default=50)
    parser.add_argument('-o', '--output_labels', action='append')
    parser.add_argument('--wide_file', type=str)
    parser.add_argument('--splits_file')
    parser.add_argument('-v', '--selected_views', action='append', choices=category_dictionaries['view'].keys(),
                        required=True)
    parser.add_argument('-d', '--selected_doppler', action='append', choices=category_dictionaries['doppler'].keys(),
                        required=True)
    parser.add_argument('-q', '--selected_quality', action='append', choices=category_dictionaries['quality'].keys(),
                        required=True)
    parser.add_argument('-c', '--selected_canonical', action='append',
                        choices=category_dictionaries['canonical'].keys(), required=True)
    parser.add_argument('-n', '--n_train_patients', default='all')
    parser.add_argument('--split_idx', type=int, choices=range(4))
    parser.add_argument('--n_splits', type=int, default=4)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--skip_modulo', type=int, default=1)
    parser.add_argument('--lmdb_folder')
    parser.add_argument('--pretrained_ckpt_dir', type=str)
    parser.add_argument('--movinet_ckpt_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--extract_embeddings', action='store_true')
    parser.add_argument('--start_beat', type=int, default=0)

    args = parser.parse_args()
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: {value}")

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
        split_idx=args.split_idx,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        skip_modulo=args.skip_modulo,
        lmdb_folder=args.lmdb_folder,
        pretrained_ckpt_dir=args.pretrained_ckpt_dir,
        movinet_ckpt_dir=args.movinet_ckpt_dir,
        output_dir=args.output_dir,
        extract_embeddings=args.extract_embeddings,
        start_beat=args.start_beat
    )
