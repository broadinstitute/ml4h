import os
import json
import logging
import argparse
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from ml4h_ccds.data_descriptions.wide_file import EcholabDataDescription
from ml4h_ccds.data_descriptions.echo import LmdbEchoStudyVideoDataDescription

from ml4h_ccds.data_descriptions.util import upload_folder_to_s3
from ml4h_ccds.model_descriptions.echo import create_movinet_classifier, create_regressor, train_model, DDGenerator

from ml4h_ccds.echo_defines import category_dictionaries, DATA_FOLDERS
from ml4h_ccds.echo_utils import download_s3_folders

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
    fine_tune,
    model_params,
    model_signature,
    adam,
    scale_outputs,
    output_signature_labels
):

    lmdb_vois = '_'.join(selected_views)
    olabels = '_'.join(output_labels)
    DATA_FOLDERS['echo_lmdb']['source'] = f'stage3_trainvalid_{lmdb_vois}'
    DATA_FOLDERS['echo_lmdb']['target'] = f'{os.path.expanduser("~")}/data/stage3_trainvalid_{lmdb_vois}/'

    # Download echo folders to scratch space
    download_s3_folders(DATA_FOLDERS, USER)

    lmdb_folder = DATA_FOLDERS['echo_lmdb']['target']
    wide_folder = DATA_FOLDERS['wide']['target']
    splits_folder = DATA_FOLDERS['splits']['target']
    model_folder = DATA_FOLDERS['supervised_models']['target']

    wide_df = pd.read_parquet(f'{wide_folder}/{wide_file}')  

    # Select only view(s) of interest
    selected_views_idx = [category_dictionaries['view'][v] for v in selected_views]
    selected_doppler_idx = [category_dictionaries['doppler'][v] for v in selected_doppler]
    selected_quality_idx = [category_dictionaries['quality'][v] for v in selected_quality]
    selected_canonical_idx = [category_dictionaries['canonical'][v] for v in selected_canonical]
    wide_df_selected = wide_df[
        (wide_df['view_prediction'].isin(selected_views_idx)) & \
        (wide_df['doppler_prediction'].isin(selected_doppler_idx)) & \
        (wide_df['quality_prediction'].isin(selected_quality_idx)) & \
        (wide_df['canonical_prediction'].isin(selected_canonical_idx))    
    ]

    # Drop entries without echolab measurements and get all sample_ids
    wide_df_selected = wide_df_selected.dropna(subset=output_labels)
    working_ids = wide_df_selected['sample_id'].values.tolist() 

    # Read splits and partition dataset
    with open(f'{splits_folder}/{splits_file}', 'r') as json_file:
        splits = json.load(json_file)

    patient_train = splits['patient_train']
    patient_valid = splits['patient_valid']

    if n_train_patients != 'all':
        patient_train = patient_train[:int(int(n_train_patients)*0.9)]
        patient_valid = patient_valid[:int(int(n_train_patients)*0.1)]

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

    INPUT_DD = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo
    )
        
    OUTPUT_DD = EcholabDataDescription(
        wide_df = wide_df_selected[['sample_id'] + output_labels].drop_duplicates(),
        sample_id_column = 'sample_id',
        column_names = output_labels,
        name = 'echolab'
    )                     

    body_train_ids = tf.data.Dataset.from_tensor_slices(train_ids).shuffle(len(train_ids), reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)            
    body_valid_ids = tf.data.Dataset.from_tensor_slices(valid_ids).shuffle(len(valid_ids), reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)

    n_train_steps = len(train_ids) // batch_size
    n_valid_steps = len(valid_ids) // batch_size

    num_classes = len(output_labels)
    output_batch_shape = (batch_size, num_classes) if num_classes > 1 else (batch_size,)

    io_train_ds = body_train_ids.interleave(
        lambda sample_ids: tf.data.Dataset.from_generator(
            DDGenerator(
                INPUT_DD,
                OUTPUT_DD
            ),
            output_signature = (
                tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=output_batch_shape, dtype=tf.float32),
            ),
            args = (sample_ids,)
        )
    ).repeat(epochs).prefetch(8)

    io_valid_ds = body_valid_ids.interleave(
        lambda sample_ids: tf.data.Dataset.from_generator(
            DDGenerator(
                INPUT_DD,
                OUTPUT_DD
            ),
            output_signature = (
                tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=output_batch_shape, dtype=tf.float32),
            ),
            args = (sample_ids,)
        )
    ).repeat(epochs).prefetch(8)           
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        _, backbone = create_movinet_classifier(
            n_input_frames,
            batch_size,
            num_classes=600,
            checkpoint_dir=f'{model_folder}/movinet_a2_base',
            freeze_backbone=fine_tune
        )
        backbone_output = backbone.layers[-1].output[0]
        flatten = tf.keras.layers.Flatten()(backbone_output)
        encoder = tf.keras.Model(inputs=[backbone.input], outputs=[flatten])

        model = create_regressor(
            encoder,
            input_shape=(n_input_frames, 224, 224, 3),
            n_output_features=len(output_labels),
            trainable=not(fine_tune)
        )

        if model_signature:
            model = create_regressor(
                encoder,
                input_shape=(n_input_frames, 224, 224, 3),
                n_output_features=len(output_signature_labels),
                trainable=not(fine_tune)
            )
            model.load_weights(f'{DATA_FOLDERS["supervised_models"]["target"]}/{model_signature}/model/chkp')        

            if set(output_labels) != set(output_signature_labels):
                model = create_regressor(
                    encoder,
                    input_shape=(n_input_frames, 224, 224, 3),
                    n_output_features=len(output_labels),
                    trainable=not(fine_tune)
                )               

        if adam:
            optimizer = tf.keras.optimizers.Adam(learning_rate=adam)
        else:
            initial_learning_rate = 0.00005 * batch_size 
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate, 
                decay_steps = n_train_steps * epochs,
            )

            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate, 
                rho=0.9, 
                momentum=0.9, 
                epsilon=1.0, 
                clipnorm=1.0
            )

        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanAbsoluteError()]

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
    output_folder = f'{model_folder}{datetime.datetime.now().strftime("%Y%m%d%H%M")}_{lmdb_vois}_{olabels}_{n_input_frames}frames{fine_tune_string}_{n_train_patients}'
    
    os.makedirs(output_folder, exist_ok=True)
    with open(f'{output_folder}/model_params.json', 'w') as json_file:
        json.dump(model_params, json_file)

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

    # Upload model folder to the supervised model bucket
    upload_folder_to_s3(
        '2017P001650',
        f'echos/{DATA_FOLDERS["supervised_models"]["source"]}',
        output_folder
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_input_frames', type=int, default=32)
    parser.add_argument(
        '-o',
        '--output_labels', 
        action='append', 
        choices=[
            'LV_LVEF_Measurement', # LVEF BETTER
            'LV_RawPercent', # LVEF 
            'LV_DiastolicDimension', # LVEDD
            'LV_SystolicDimension', # LVESD
            'LV_SeptalThicknessMm', # IVS Thickness
            'LV_PosteriorWallThicknessMm', #PW thickness
            'RV_RV_Dimension',
            'converted_RVEDV',
            'converted_RVEF',
            'top3_discrepancy',
            'bottom3_discrepancy'
            'RV_RV_BasalDiameter',
            'LA_A_P',
            'RV_RV_FAC',
            'RV_CavitySize',
            'RV_SystolicFunction'
        ], 
        type=str
    )
    parser.add_argument('--wide_file', type=str, default='stage3_trainvalid_LVEF.pq')
    parser.add_argument('-v', '--selected_views', action='append', choices=category_dictionaries['view'].keys(), required=True)
    parser.add_argument('-d', '--selected_doppler', action='append', choices=category_dictionaries['doppler'].keys(), required=True)
    parser.add_argument('-q', '--selected_quality', action='append', choices=category_dictionaries['quality'].keys(), required=True)
    parser.add_argument('-c', '--selected_canonical', action='append', choices=category_dictionaries['canonical'].keys(), required=True)
    parser.add_argument('-n', '--n_train_patients', type=str, required=True)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--skip_modulo', type=int, default=2)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--model_signature', type=str)
    parser.add_argument('--splits_file', default='stage3_splits_050222.json')
    parser.add_argument('--adam', default=None, type=float)
    parser.add_argument('--scale_outputs', action='store_true')
    parser.add_argument(
        '-os',
        '--output_signature_labels', 
        action='append', 
        choices=[
            'LV_RawPercent', 
            'LV_DiastolicDimension', 
            'LV_SystolicDimension', 
            'LV_SeptalThicknessMm', 
            'LV_PosteriorWallThicknessMm',
            'RV_RV_Dimension',
            'converted_RVEDV',
            'converted_RVEF',
            'top3_discrepancy',
            'bottom3_discrepancy'
            'RV_RV_BasalDiameter',
            'LA_A_P',
            'RV_RV_FAC',
            'RV_CavitySize',
            'RV_SystolicFunction'
        ], 
        type=str
    )

    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    model_params = {}
    for arg, value in sorted(vars(args).items()):
        logging.info(f"Argument {arg}: {value}")
        model_params[arg] = value
    
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
        fine_tune=args.fine_tune,
        model_params=model_params,
        model_signature=args.model_signature,
        adam=args.adam,
        scale_outputs=args.scale_outputs,
        output_signature_labels=args.output_signature_labels
    )
