import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from echo_defines import category_dictionaries
from model_descriptions.echo import DDGenerator, create_movinet_classifier, create_regressor, create_regressor_classifier

import wandb
from wandb.integration.keras import WandbMetricsLogger

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

SAVE_ONEHOT_DF_FOR_EACH_CLASS = True


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
        start_beat
):
    # Loading information on saved model:
    model_param_path = os.path.join(os.path.split(os.path.dirname(pretrained_chkp_dir))[0], 'model_params.json')
    with open(model_param_path, 'r') as json_file:
        model_params = json.load(json_file)

    output_labels = model_params['output_labels'] if not output_labels else output_labels
    selected_views = model_params['selected_views'] if not selected_views else selected_views
    selected_doppler = model_params['selected_doppler'] if not selected_doppler else selected_doppler
    selected_quality = model_params['selected_quality'] if not selected_quality else selected_quality
    selected_canonical = model_params['selected_canonical'] if not selected_canonical else selected_canonical
    logging.info(f'Loaded model with output labels: {output_labels}, views: {selected_views}, doppler: {selected_doppler}, quality: {selected_quality}, canonical: {selected_canonical}')

    # ---------- Adaptation for regression + classification ---------- #
    if ('output_labels_types' in model_params.keys()) and ('c' in model_params['output_labels_types'].lower()):
        cls_lbl_map_path = os.path.join(os.path.split(os.path.dirname(pretrained_chkp_dir))[0],
                                        'classification_class_label_mapping_per_output.json')
        with open(cls_lbl_map_path, 'r') as json_file:
            cls_category_map_dicts = json.load(json_file)
        cls_category_len_dict = {}
        for c_lbl in cls_category_map_dicts['cls_output_order']:
            cls_category_len_dict[c_lbl] = len(cls_category_map_dicts[c_lbl])
        # Reordering output labels to fit the regression-classification output order during training (assuming correct
        # output_labels that include all saved classification output names - if not, the classification output names
        # are added next anyway):
        output_labels = ([i for i in output_labels if i not in cls_category_map_dicts['cls_output_order']] +
                        cls_category_map_dicts['cls_output_order'])
        logging.info(f'Loaded model contains classification heads. Updated output_label_order: {output_labels}, with classification heads for: {cls_category_map_dicts["cls_output_order"]}')
        output_reg_len = len(output_labels) - len(cls_category_map_dicts['cls_output_order'])
        add_separate_dense_reg = cls_category_map_dicts['add_separate_dense_reg']
        add_separate_dense_cls = cls_category_map_dicts['add_separate_dense_cls']
    else:
        logging.info(f'Loaded model contains only regression variables.')
        output_reg_len = len(output_labels)
        cls_category_len_dict = {}
        add_separate_dense_reg = model_params[
            'add_separate_dense_reg'] if 'add_separate_dense_reg' in model_params.keys() else False
        add_separate_dense_cls = model_params[
            'add_separate_dense_cls'] if 'add_separate_dense_cls' in model_params.keys() else False
    # ---------------------------------------------------------------- #

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
    wide_df_selected.to_csv(f'/home/wide_df_selected.csv')
    
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

    # Testing a random subset of IDs (for speed) to see if there are matching ids in working_ids and chosen split
    random_ids_for_test = np.random.permutation(len(working_ids))[:min(len(working_ids), 500)]
    working_ids_subset_test = [working_ids[i] for i in random_ids_for_test]
    inference_ids_match_test = [t for t in working_ids_subset_test if int(t.split('_')[0]) in patient_inference]
    if len(inference_ids_match_test) == 0:
        logging.warning(
            f'A random test of indices showed no match between {wide_file} indices and {splits_file} indices. It is possible that there are still matches, but please verify file names. This process might take a long time to break if there are no matches, consider forcing it to stop.')

    inference_ids = sorted([t for t in working_ids if int(t.split('_')[0]) in patient_inference])
    if len(inference_ids) == 0:
        logging.error(f'No matches found between {wide_file} indices and the {splits_file} indices!')
        sys.exit()

    INPUT_DD = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo,
        start_frame=start_beat
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
        ),
        cycle_length = 2,
        num_parallel_calls = 2
    ).prefetch(8)

    model, backbone = create_movinet_classifier(
        n_input_frames,
        batch_size,
        num_classes=600,
        checkpoint_dir=movinet_chkp_dir,
    )

    backbone_output = backbone.layers[-1].output[0]
    flatten = tf.keras.layers.Flatten()(backbone_output)
    encoder = tf.keras.Model(inputs=[backbone.input], outputs=[flatten])
    # ---------- Adaptation for regression + classification ---------- #
    # Organize regressor/classifier inputs:
    func_args = {'input_shape': (n_input_frames, 224, 224, 3),
                'n_output_features': output_reg_len,
                'categories': cls_category_len_dict,
                'category_order': cls_category_map_dicts['cls_output_order'] if cls_category_len_dict else None,
                'add_dense': {'regressor': add_separate_dense_reg, 'classifier': add_separate_dense_cls}}

    model_plus_head = create_regressor_classifier(encoder, **func_args)
    # ---------------------------------------------------------------- #
    model_plus_head.load_weights(pretrained_chkp_dir)

    vois = '_'.join(selected_views)
    ufm = 'conv7'
    if extract_embeddings:
        output_folder = os.path.join(output_dir,
                                    f'inference_embeddings_{vois}_{ufm}_{lmdb_folder.split("/")[-1]}_{splits_file.split("/")[-1]}_{start_beat}')
    else:
        output_folder = os.path.join(output_dir,
                                    f'inference_{vois}_{ufm}_{lmdb_folder.split("/")[-1]}_{splits_file.split("/")[-1]}_{start_beat}')
    os.makedirs(output_folder, exist_ok=True)

    wide_df_selected.to_csv(f'{output_folder}/wide_df_selected.csv')
    
    def save_model_pred_as_df(pred, fname_suffix='', pred_col_names=[]):
        save_df = pd.DataFrame()
        save_df['sample_id'] = inference_ids_split
        if len(pred_col_names) == pred.shape[1]:
            use_pred_col_names = True
        else:
            use_pred_col_names = False
        for i_p in range(pred.shape[1]):
            if use_pred_col_names:
                save_df[pred_col_names[i_p]] = pred[:, i_p]
            else:
                save_df[f'prediction_{i_p}'] = pred[:, i_p]

        save_df.to_parquet(os.path.join(output_folder, f'prediction_{split_idx}' + fname_suffix + '.pq'))

    run = wandb.init(
        project = "echo_mvp"
    )

    if extract_embeddings:
        embeddings = encoder.predict(io_inference_ds, steps=n_inference_steps, verbose=1, callbacks=[WandbMetricsLogger(log_freq=10)])
        df = pd.DataFrame()
        df['sample_id'] = inference_ids_split
        for j, _ in enumerate(range(embeddings.shape[1])):
            df[f'embedding_{j}'] = embeddings[:, j]

        df.to_parquet(os.path.join(output_folder, f'prediction_{split_idx}.pq'))
    else:
        predictions = model_plus_head.predict(io_inference_ds, steps=n_inference_steps, verbose=1, callbacks=[WandbMetricsLogger(log_freq=10)])
        # predictions is a list of length = number of outputs in list, where all regression variables are in a single
        # list element and each classification task has a separate list element.
        # Each list element is of size:
        # len(inference_ids_split) X number of output variables (total number of regression vars or number of classes)
        if len(cls_category_len_dict) > 0:
            # Case: regression + classification or classification only
            # Currently saving actual class predictions jointly with the regression variables if exist
            # and for each class one-hot predictions are saved in a separate pq file (flag dependent)
            if output_reg_len > 0:
                reg_pred = predictions[0]
                cls_pred = predictions[1:]
            else:
                reg_pred = np.zeros((0, 0))
                cls_pred = predictions     
                if len(cls_category_len_dict) == 1: 
                    cls_pred = [predictions]
            df = pd.DataFrame()
            df['sample_id'] = inference_ids_split
            for i_p in range(reg_pred.shape[1]):
                df[f'prediction_{i_p}'] = reg_pred[:, i_p]
            for i in range(len(cls_pred)):
                curr_cls_name = cls_category_map_dicts['cls_output_order'][i]
                if SAVE_ONEHOT_DF_FOR_EACH_CLASS:
                    save_model_pred_as_df(cls_pred[i], fname_suffix='_one_hot_' + curr_cls_name)
                cls_pred_vals_curr = cls_pred[i].argmax(axis=1)
                cls_map_inv = {v: k for k, v in zip(cls_category_map_dicts[curr_cls_name].keys(),
                                                    cls_category_map_dicts[curr_cls_name].values())}
                df[cls_category_map_dicts['cls_output_order'][i]] = cls_pred_vals_curr
                df.replace({curr_cls_name: cls_map_inv}, inplace=True)

            df.to_parquet(os.path.join(output_folder, f'prediction_{split_idx}.pq'))
        else:
            # Case: regression only
            save_model_pred_as_df(predictions)
    run.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_input_frames', type=int, default=50)
    parser.add_argument('-o', '--output_labels', action='append', required=False)
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
    parser.add_argument('--split_idx', type=int, choices=range(4))
    parser.add_argument('--n_splits', type=int, default=4)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--skip_modulo', type=int, default=1)
    parser.add_argument('--lmdb_folder', type=str)
    parser.add_argument('--pretrained_chkp_dir', type=str)
    parser.add_argument('--movinet_chkp_dir', type=str)
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
        pretrained_chkp_dir=args.pretrained_chkp_dir,
        movinet_chkp_dir=args.movinet_chkp_dir,
        output_dir=args.output_dir,
        extract_embeddings=args.extract_embeddings,
        start_beat=args.start_beat,
    )
