import argparse
import datetime
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from data_descriptions.wide_file import EcholabDataDescription
from echo_defines import category_dictionaries
from model_descriptions.echo import create_movinet_classifier, create_regressor, create_regressor_classifier, train_model, DDGenerator

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

USER = os.getenv('USER')


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
        output_signature_labels,
        output_labels_types,
        add_separate_dense_reg,
        add_separate_dense_cls
):
    lmdb_vois = '_'.join(selected_views)
    olabels = '_'.join(output_labels)

    # ---------- Adaptation for regression + classification ---------- #
    # ---- Processing input values and handling incorrect inputs ----- #
    # Specify parameters for regression and classification heads:
    if len(output_labels_types) == len(output_labels):
        # Number of task types labels (regression/classification) is equal to the number of output variables
        unq_lbl_types = set([ch for ch in output_labels_types.lower()])
    elif len(output_labels_types) == 1:
        # Only one task type label (regression/classification) is given for all output variables
        unq_lbl_types = output_labels_types.lower()
    else:
        # A wrong number of task type labels was given (empty or different from 1 or 'len(output_labels)')
        logging.info(
            "Reverting to a regression head only since the lengths of 'output_labels' and 'output_labels_types' do not match.")
        unq_lbl_types = 'r'
    if ('r' not in unq_lbl_types) and ('c' not in unq_lbl_types):
        # Wrong task type labels were given (letters other than 'r' for regression and 'c' for classification)
        logging.info(
            "Reverting to a regression head only since the lengths of 'output_labels_types' contains unrecognized commands.")
        unq_lbl_types = 'r'
    if pretrained_chkp_dir and set(output_labels) == set(output_signature_labels):
        # Currently not supporting this case. TODO: check what are the 'output_signature_labels', can we support this?
        logging.info(
            "Reverting to a regression head only since the code currently does not support loading from checkpoint with 'output_signature_labels'")
        unq_lbl_types = 'r'

    # Grouping regression tasks together and classification tasks together
    # and computing output lengths (number of regression variables and number of classification tasks)
    if len(unq_lbl_types) > 1:
        # Both regression and classification tasks were specified
        output_label_types_int = [0 if (ch=='r') else 1 for ch in output_labels_types.lower()]
        output_reg_len = len(output_label_types_int) - sum(output_label_types_int)
        cls_output_names = [output_labels[i_c] for i_c, c in enumerate(output_label_types_int) if c == 1]
        output_order = np.argsort(output_label_types_int)
        output_labels = [output_labels[i] for i in output_order]
    elif 'r' in unq_lbl_types:
        # Only one task type specified - regression
        output_reg_len = len(output_labels)
        cls_output_names = []
    else:
        # Only one task type - classification
        output_reg_len = 0
        cls_output_names = output_labels
    # ---------------------------------------------------------------- #

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

    # Drop entries without echolab measurements and get all sample_ids
    wide_df_selected = wide_df_selected.dropna(subset=output_labels)
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

    # If scale_outputs, normalize by summary stats of training set
    if scale_outputs:
        wide_df_train = wide_df_selected[wide_df_selected['sample_id'].isin(train_ids)]
        mean_outputs = np.mean(wide_df_train[output_labels].values, axis=0)
        std_outputs = np.std(wide_df_train[output_labels].values, axis=0)
        wide_df_selected.loc[:, output_labels] = (wide_df_selected[output_labels].values - mean_outputs) / std_outputs
        logging.info(mean_outputs)
        logging.info(std_outputs)

    train_ids = list(set(train_ids).intersection(set(working_ids)))
    valid_ids = list(set(valid_ids).intersection(set(working_ids)))

    # ---------- Adaptation for regression + classification ---------- #
    # Creating dictionaries specifying number of classes for each output_label name
    # and mapping between wide_file values to class labels:
    cls_category_map_dicts = {}
    cls_category_len_dict = {}
    for c_lbl in cls_output_names:
        all_cls_vals = np.sort(wide_df_selected[c_lbl].drop_duplicates().tolist())
        val2clsind_map_dict = {val: c_ind for val, c_ind in zip(all_cls_vals, range(len(all_cls_vals)))}
        cls_category_map_dicts[c_lbl] = val2clsind_map_dict
        cls_category_len_dict[c_lbl] = len(wide_df_selected[c_lbl].drop_duplicates())
        if cls_category_len_dict[c_lbl] < 2:
            logging.info(f'Error: Output variable {c_lbl} has a constant value in the train and validation sets - will cause errors for in the classifier')
    cls_category_map_dicts['cls_output_order'] = cls_output_names
    # ---------------------------------------------------------------- #

    INPUT_DD = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo
    )

    OUTPUT_DD = EcholabDataDescription(
        wide_df=wide_df_selected[['sample_id'] + output_labels].drop_duplicates(),
        sample_id_column='sample_id',
        column_names=output_labels,
        name='echolab',
        # ---------- Adaptation for regression + classification ---------- #
        cls_categories_map=cls_category_map_dicts if cls_output_names else None
        # ---------------------------------------------------------------- #
    )

    body_train_ids = tf.data.Dataset.from_tensor_slices(train_ids).shuffle(len(train_ids),
                                                                           reshuffle_each_iteration=True).batch(
        batch_size, drop_remainder=True)
    body_valid_ids = tf.data.Dataset.from_tensor_slices(valid_ids).shuffle(len(valid_ids),
                                                                           reshuffle_each_iteration=True).batch(
        batch_size, drop_remainder=True)

    n_train_steps = len(train_ids) // batch_size
    n_valid_steps = len(valid_ids) // batch_size

    # ---------- Adaptation for regression + classification ---------- #
    # Adapting tensor output sizes for classification heads
    if cls_output_names:
        num_classes = [output_reg_len] + list(cls_category_len_dict.values())
        output_signatures = (
            tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32),
            tuple([tf.TensorSpec(shape=(batch_size, n_c), dtype=tf.float32)
                   for n_c in num_classes])
        )
    else:
        num_classes = len(output_labels)
        output_signatures = (
            tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, num_classes) if num_classes > 1 else (batch_size,), dtype=tf.float32)
        )
    # ---------------------------------------------------------------- #

    io_train_ds = body_train_ids.interleave(
        lambda sample_ids: tf.data.Dataset.from_generator(
            DDGenerator(
                INPUT_DD,
                OUTPUT_DD
            ),
            output_signature=output_signatures,
            args=(sample_ids,)
        )
    ).repeat(epochs).prefetch(8)

    io_valid_ds = body_valid_ids.interleave(
        lambda sample_ids: tf.data.Dataset.from_generator(
            DDGenerator(
                INPUT_DD,
                OUTPUT_DD
            ),
            output_signature=output_signatures,
            args=(sample_ids,)
        )
    ).repeat(epochs).prefetch(8)

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
                     'category_order': cls_category_map_dicts['cls_output_order'] if cls_category_len_dict else None,
                     'add_dense': {'regressor': add_separate_dense_reg, 'classifier': add_separate_dense_cls}}

        model = create_regressor_classifier(encoder, **func_args)
        # ---------------------------------------------------------------- #

        if pretrained_chkp_dir:
            # TODO: Check if we can achieve compatability with output_signature_labels, currently left unchanged for that scenario
            model = create_regressor(
                encoder,
                input_shape=(n_input_frames, 224, 224, 3),
                n_output_features=len(output_signature_labels),
                trainable=not fine_tune
            )
            model.load_weights(pretrained_chkp_dir)

            if set(output_labels) != set(output_signature_labels):
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

        loss = {'cls_' + k: tf.keras.losses.CategoricalCrossentropy() for k in cls_category_len_dict.keys()}
        metrics = {'cls_' + k: tf.keras.metrics.CategoricalAccuracy() for k in cls_category_len_dict.keys()}
        if output_reg_len > 0:
            loss['echolab'] = tf.keras.losses.MeanSquaredError()
            metrics['echolab'] = tf.keras.metrics.MeanAbsoluteError()

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    io_train_ds = io_train_ds.with_options(options)
    io_valid_ds = io_valid_ds.with_options(options)

    fine_tune_string = f'_fine_tune' if fine_tune else ''
    output_folder = os.path.join(output_dir,
                                 f'{datetime.datetime.now().strftime("%Y%m%d%H%M")}_{lmdb_vois}_{olabels}_{n_input_frames}frames{fine_tune_string}_{n_train_patients}')

    os.makedirs(output_folder, exist_ok=True)
    with open(f'{output_folder}/model_params.json', 'w') as json_file:
        json.dump(model_params, json_file)

    wide_df_selected.to_parquet(f'{output_folder}/wide_df_selected.pq')

    # ---------- Adaptation for regression + classification ---------- #
    # Record output labels new order (after possible reordering of regression and classification):
    with open(f'{output_folder}/output_labels_final_ordering.json', 'w') as json_file:
        json.dump(output_labels, json_file)
    # Record output mapping for classification tasks (dictionary that contains column names as well):
    if cls_output_names:
        cls_category_map_dicts['add_separate_dense_cls'] = add_separate_dense_cls
        cls_category_map_dicts['add_separate_dense_reg'] = add_separate_dense_reg
        with open(f'{output_folder}/classification_class_label_mapping_per_output.json', 'w') as json_file:
            json.dump(cls_category_map_dicts, json_file)
    # ---------------------------------------------------------------- #

    logging.info(model.summary())
    trained_model = train_model(
        model,
        io_train_ds,
        io_valid_ds,
        epochs,
        n_train_steps,
        n_valid_steps,
        output_folder
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
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--splits_file')
    parser.add_argument('--adam', default=None, type=float)
    parser.add_argument('--scale_outputs', action='store_true')
    parser.add_argument('-os', '--output_signature_labels', action='append', type=str)
    # ---------- Adaptation for regression + classification ---------- #
    parser.add_argument('--output_labels_types', default='r', type=str,
                        help='A string indicating task types: r for regression, c for classification. Should be of length 1 or the same length of the specified output_labels variable, e.g. "r" or "rrcr".')
    parser.add_argument('--add_separate_dense_reg', action='store_true',
                        help='Adds an additional dense layer trained separately for the regression head')
    parser.add_argument('--add_separate_dense_cls', action='store_true',
                        help='Adds an additional dense layer trained separately for the classification head')
    # ---------------------------------------------------------------- #
    args = parser.parse_args()

    # TODO? add option to set number of classes parameter manually for classification (in case there are more classes than the ones in the data?)

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
        output_signature_labels=args.output_signature_labels,
        # ---------- Adaptation for regression + classification ---------- #
        output_labels_types=args.output_labels_types,
        add_separate_dense_reg=args.add_separate_dense_reg,
        add_separate_dense_cls=args.add_separate_dense_cls
        # ---------------------------------------------------------------- #
    )
