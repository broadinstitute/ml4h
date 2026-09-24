import argparse
import io
import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from data_descriptions.echo_dataset import make_inference_dataset
from echo_defines import category_dictionaries
from model_descriptions.droid_model import build_encoder, build_model, read_json, read_trained_run, with_embeddings

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

SAVE_ONEHOT_DF_FOR_EACH_CLASS = True


def _write_parquet(df, path):
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    with tf.io.gfile.GFile(path, 'wb') as pq_file:
        pq_file.write(buffer.getvalue())


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
        pretrained_chkp_dir,
        movinet_chkp_dir,
        output_dir,
        extract_embeddings,
        start_beat,
        loader_workers=4,
        video_decode_mode='auto',
        video_decode_threads=1,
        prefetch_batches=1,
):
    if not 0 <= split_idx < n_splits:
        raise ValueError(f'split_idx must be in [0, {n_splits})')
    if loader_workers < 1 or video_decode_threads < 0 or prefetch_batches < 0:
        raise ValueError('loader_workers must be positive; decode threads/prefetch must be non-negative')

    # Loading information on saved model:
    run = read_trained_run(pretrained_chkp_dir)
    model_params = run.params
    if output_labels and output_labels != run.output_labels:
        logging.warning(f'Ignoring --output_labels {output_labels}; the checkpoint predicts {run.output_labels}')
    output_labels = run.output_labels
    selected_views = selected_views or model_params['selected_views']
    selected_doppler = selected_doppler or model_params['selected_doppler']
    selected_quality = selected_quality or model_params['selected_quality']
    selected_canonical = selected_canonical or model_params['selected_canonical']
    n_input_frames = n_input_frames or model_params.get('n_input_frames')
    skip_modulo = skip_modulo or model_params.get('skip_modulo')
    if not (n_input_frames and skip_modulo):
        raise ValueError('The checkpoint does not record n_input_frames/skip_modulo; pass them explicitly.')
    logging.info(f'Loaded model with output labels: {output_labels}, '
                 f'classification heads: {run.head_spec["category_order"]}, '
                 f'survival heads: {list(run.head_spec["survival_heads"])}, views: {selected_views}, '
                 f'doppler: {selected_doppler}, quality: {selected_quality}, canonical: {selected_canonical}, '
                 f'frames: {n_input_frames}, skip_modulo: {skip_modulo}')

    # One GPU per split so that splits can run side by side on a multi-GPU host
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.set_visible_devices([physical_devices[split_idx % len(physical_devices)]], 'GPU')

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
    ].copy()

    # Fill entries without measurements and get all sample_ids
    for olabel in output_labels:
        if olabel in wide_df_selected:
            wide_df_selected.loc[wide_df_selected[olabel].isna(), olabel] = -1
    working_ids = wide_df_selected['sample_id'].values.tolist()

    # Read splits and partition dataset
    splits = read_json(splits_file)

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

    if 'trainvalid' in lmdb_folder:
        patient_inference = patient_train + patient_valid
        if 'patient_internal_test' in splits:
            patient_inference = patient_inference + splits['patient_internal_test']
    else:
        patient_inference = splits['patient_test']
    patient_inference = set(patient_inference)

    inference_ids = sorted([t for t in working_ids if int(t.split('_')[0]) in patient_inference])
    if len(inference_ids) == 0:
        logging.error(f'No matches found between {wide_file} indices and the {splits_file} indices!')
        sys.exit(1)

    INPUT_DD = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo,
        start_frame=start_beat,
        decode_mode=video_decode_mode,
        decode_threads=video_decode_threads,
    )

    inference_ids_split = [str(i) for i in np.array_split(inference_ids, n_splits)[split_idx]]
    io_inference_ds = make_inference_dataset(
        INPUT_DD,
        inference_ids_split,
        batch_size,
        (n_input_frames, 224, 224, 3),
        workers=loader_workers,
    ).prefetch(prefetch_batches)
    logging.info('Video loader: workers=%d, decode=%s, decode_threads=%d, prefetch_batches=%d',
                 loader_workers, video_decode_mode, video_decode_threads, prefetch_batches)

    encoder = build_encoder(n_input_frames, batch_size, movinet_chkp_dir)
    model_plus_head = build_model(encoder, run.head_spec, n_input_frames, trainable=False)
    model_plus_head.load_weights(pretrained_chkp_dir)

    vois = '_'.join(selected_views)
    ufm = 'conv7'
    folder_suffix = f'{vois}_{ufm}_{lmdb_folder.rstrip("/").split("/")[-1]}_{splits_file.split("/")[-1]}_{start_beat}'

    def make_output_folder(prefix):
        folder = os.path.join(output_dir, f'{prefix}_{folder_suffix}')
        tf.io.gfile.makedirs(folder)
        with tf.io.gfile.GFile(f'{folder}/wide_df_selected.csv', 'w') as csv_file:
            wide_df_selected.to_csv(csv_file)
        return folder

    output_folder = make_output_folder('inference')

    def prediction_path(fname_suffix='', folder=output_folder):
        return os.path.join(folder, f'prediction_{split_idx}' + fname_suffix + '.pq')

    def columns_df(pred, column_prefix='prediction'):
        df = pd.DataFrame({'sample_id': inference_ids_split})
        for i_p in range(pred.shape[1]):
            df[f'{column_prefix}_{i_p}'] = pred[:, i_p]
        return df

    # One pass over the clips; with --extract_embeddings the encoder output is returned alongside the heads.
    if extract_embeddings:
        predictions = with_embeddings(model_plus_head, encoder).predict(io_inference_ds, verbose=1)
        embeddings, predictions = predictions[0], predictions[1:]
        _write_parquet(columns_df(embeddings, 'embedding'),
                       prediction_path(folder=make_output_folder('inference_embeddings')))
    else:
        predictions = model_plus_head.predict(io_inference_ds, verbose=1)
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

    # Outputs follow the model: regression ('echolab'), one 'cls_<label>' per classification
    # task, then one 'survival_<task>' per survival task.
    df = pd.DataFrame({'sample_id': inference_ids_split})
    for output_name, pred in zip(model_plus_head.output_names, predictions):
        if output_name == 'echolab':
            for i_p in range(pred.shape[1]):
                df[f'prediction_{i_p}'] = pred[:, i_p]
        elif output_name.startswith('cls_'):
            # Class labels go in the joint file; per-class probabilities in a separate file (flag dependent)
            cls_name = output_name[len('cls_'):]
            if SAVE_ONEHOT_DF_FOR_EACH_CLASS:
                _write_parquet(columns_df(pred), prediction_path('_one_hot_' + cls_name))
            cls_map_inv = {v: k for k, v in run.cls_category_map_dicts[cls_name].items()}
            df[cls_name] = [cls_map_inv[i] for i in pred.argmax(axis=1)]
        elif output_name.startswith('survival_'):
            # Conditional survival probability per interval and cumulative survival over the whole window
            for i_p in range(pred.shape[1]):
                df[f'{output_name}_{i_p}'] = pred[:, i_p]
            df[f'{output_name}_cumulative'] = np.prod(pred, axis=1)
        else:
            raise ValueError(f'Unexpected model output {output_name}')
    _write_parquet(df, prediction_path())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_input_frames', type=int,
                        help='Frames per clip; defaults to the value the checkpoint was trained with.')
    parser.add_argument('-o', '--output_labels', action='append', required=False,
                        help='Deprecated: outputs are read from the checkpoint.')
    parser.add_argument('--wide_file', type=str)
    parser.add_argument('--splits_file')

    parser.add_argument('-v', '--selected_views', action='append', choices=category_dictionaries['view'].keys(),
                        required=False)
    parser.add_argument('-d', '--selected_doppler', action='append', choices=category_dictionaries['doppler'].keys(),
                        required=False)
    parser.add_argument('-q', '--selected_quality', action='append', choices=category_dictionaries['quality'].keys(),
                        required=False)
    parser.add_argument('-c', '--selected_canonical', action='append',
                        choices=category_dictionaries['canonical'].keys(), required=False)

    parser.add_argument('-n', '--n_train_patients', default='all')
    parser.add_argument('--split_idx', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=4)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--skip_modulo', type=int,
                        help='Frame stride; defaults to the value the checkpoint was trained with.')
    parser.add_argument('--lmdb_folder', type=str)
    parser.add_argument('--loader_workers', type=int, default=4,
                        help='Parallel clip loaders (start with 4 on an 8-12 CPU host).')
    parser.add_argument('--video_decode_mode', choices=['auto', 'sequential'], default='auto',
                        help='auto skips unselected MJPEG frames; other codecs decode sequentially.')
    parser.add_argument('--video_decode_threads', type=int, default=1,
                        help='Threads per generic decoder; selective MJPEG always uses one. 0 = FFmpeg auto.')
    parser.add_argument('--prefetch_batches', type=int, default=1,
                        help='Bounded batch prefetch; 0 disables prefetch.')
    parser.add_argument('--pretrained_chkp_dir', type=str)
    parser.add_argument('--movinet_chkp_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--extract_embeddings', action='store_true',
                        help='Also save encoder embeddings, computed in the same pass as the predictions.')
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
        pretrained_chkp_dir=args.pretrained_chkp_dir,
        movinet_chkp_dir=args.movinet_chkp_dir,
        output_dir=args.output_dir,
        extract_embeddings=args.extract_embeddings,
        start_beat=args.start_beat,
        loader_workers=args.loader_workers,
        video_decode_mode=args.video_decode_mode,
        video_decode_threads=args.video_decode_threads,
        prefetch_batches=args.prefetch_batches,
    )
