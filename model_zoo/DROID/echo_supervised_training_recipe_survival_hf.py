import argparse
import datetime
import json
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from data_descriptions.wide_file import EcholabDataDescription
from echo_defines import category_dictionaries
from model_descriptions.echo import create_movinet_classifier, create_regressor, create_regressor_classifier, train_model, DDGenerator
from model_descriptions.echo import survival_likelihood_loss
# from metrics import concordance_index  # , concordance_index_censored

logging.basicConfig(level=logging.INFO)
tf.get_logger().setLevel(logging.ERROR)

USER = os.getenv('USER')
BALANCED_SAMPLING_HF = False

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
        es_patience,
        es_loss2monitor,
        output_labels_types,
        add_separate_dense_reg,
        add_separate_dense_cls,
        loss_weights,
        hf_task,
        hf_diag_type,
        save_tag,
        class_weights,
        survival_var_names,
        survival_intervals,
        survival_days_window
):
    lmdb_vois = '_'.join(selected_views)
    olabels = '_'.join(output_labels)

    if not survival_var_names:
        survival_var_names = []
    
    if hf_task:
        if hf_task != 'survival':
            output_labels += [category_dictionaries['hf_diag_type'][hf_diag_type]]
            output_labels = list(set(output_labels))

            # ---------- Adaptation for regression + classification ---------- #
    def process_labels_types(o_lbls, o_lbls_types, var_type='output_labels'):
        # ---- Processing input values and handling incorrect inputs ----- #
        # Specify parameters for regression and classification heads:
        if len(o_lbls_types) == len(o_lbls):
            # Number of task types labels (regression/classification) is equal to the number of output variables
            unq_lbl_types = set([ch for ch in o_lbls_types.lower()])
        elif len(o_lbls_types) == 1:
            # Only one task type label (regression/classification) is given for all output variables
            unq_lbl_types = o_lbls_types.lower()
        else:
            # A wrong number of task type labels was given (empty or different from 1 or 'len(output_labels)')
            raise TypeError(
                f"The lengths of '{var_type}' and '{var_type}_types' do not match (should be equal or 'len({var_type}_types)=1').")
        if not set(unq_lbl_types) <= {'r', 'c'}:
            # Wrong task type labels were given (letters other than 'r' for regression and 'c' for classification)
            raise TypeError(f"'{var_type}_types' contains unrecognized letters (should include 'r' and/or 'c' only).")

        # Grouping regression tasks together and classification tasks together
        # and computing output lengths (number of regression variables and number of classification tasks)
        if len(unq_lbl_types) > 1:
            # Both regression and classification tasks were specified
            output_label_types_int = [0 if (ch == 'r') else 1 for ch in o_lbls_types.lower()]
            o_reg_len = len(output_label_types_int) - sum(output_label_types_int)
            cls_o_names = [o_lbls[i_c] for i_c, c in enumerate(output_label_types_int) if c == 1]
            output_order = np.argsort(output_label_types_int)
            o_lbls = [o_lbls[i] for i in output_order]
            if var_type == 'output_labels':
                logging.info('Training with regression and classification heads')
            else:
                logging.info('Loaded model has regression and classification heads')
            logging.info(f'Updated {var_type} order: {o_lbls}')
        elif 'r' in unq_lbl_types:
            # Only one task type specified - regression
            o_reg_len = len(o_lbls)
            cls_o_names = []
            if var_type == 'output_labels':
                logging.info('Training only with a regression head')
            else:
                logging.info('Loaded model has only a regression head')
        else:
            # Only one task type - classification
            o_reg_len = 0
            cls_o_names = o_lbls
            if var_type == 'output_labels':
                logging.info('Training only with a classification head')
            else:
                logging.info('Loaded model has only a classification head')

        return o_lbls, o_reg_len, cls_o_names

    def process_class_categories(df, cls_o_names, var_type='output_labels'):
        # Creating dictionaries specifying number of classes for each output_label name
        # and mapping between wide_file values to class labels:
        clsc_map_dicts = {}
        clsc_len_dict = {}
        for c_lbl in cls_o_names:
            all_cls_vals = np.sort(df[c_lbl].drop_duplicates().tolist())
            val2clsind_map_dict = {val: c_ind for val, c_ind in zip(all_cls_vals, range(len(all_cls_vals)))}
            clsc_map_dicts[c_lbl] = val2clsind_map_dict
            clsc_len_dict[c_lbl] = len(df[c_lbl].drop_duplicates())
            if clsc_len_dict[c_lbl] < 2:
                logging.error(
                    f'Error: Output variable {c_lbl} has a constant value in the train and validation sets - might cause errors in the classifier. Error raised when processing {var_type} related classification variables.')
        clsc_map_dicts['cls_output_order'] = cls_o_names
        return clsc_map_dicts, clsc_len_dict

    # ---------------------------------------------------------------- #
    output_labels, output_reg_len, cls_output_names = process_labels_types(output_labels, output_labels_types,
                                                                           var_type='output_labels')
    # ---------------------------------------------------------------- #
    wide_df = pd.read_parquet(wide_file)

    # Select only view(s) of interest
    selected_views_idx = [category_dictionaries['view'][v] for v in selected_views]
    selected_doppler_idx = [category_dictionaries['doppler'][v] for v in selected_doppler]
    selected_quality_idx = [category_dictionaries['quality'][v] for v in selected_quality]
    selected_canonical_idx = [category_dictionaries['canonical'][v] for v in selected_canonical]
    selected_hf_task_idx = category_dictionaries['hf_task'][hf_task] if hf_task else [0, 1, 2, 3]
    wide_df_selected = wide_df[
        (wide_df['view_prediction'].isin(selected_views_idx)) &
        (wide_df['doppler_prediction'].isin(selected_doppler_idx)) &
        (wide_df['quality_prediction'].isin(selected_quality_idx)) &
        (wide_df['canonical_prediction'].isin(selected_canonical_idx)) &
        (wide_df[category_dictionaries['hf_diag_type'][hf_diag_type]].isin(selected_hf_task_idx))
        ]
    
    if 'SexDSC' in output_labels:
        wide_df_selected.loc[np.logical_or(wide_df_selected['SexDSC'] == 'Male', wide_df_selected['SexDSC'] == 'M'), 'SexDSC'] = 'M'
        wide_df_selected.loc[np.logical_or(wide_df_selected['SexDSC'] == 'Female', wide_df_selected['SexDSC'] == 'F'), 'SexDSC'] = 'F'
        
    # Drop entries without echolab measurements and get all sample_ids
    wide_df_selected = wide_df_selected.dropna(subset=output_labels)
    working_ids = wide_df_selected['sample_id'].values.tolist()

    # Read splits and partition dataset
    with open(splits_file, 'r') as json_file:
        splits = json.load(json_file)

    if 'patient_train' in splits.keys():
        patient_train = splits['patient_train']
        patient_valid = splits['patient_valid']
    elif 'patient_train' in splits[list(splits.keys())[0]].keys():
        patient_train = splits['ewoc_mgh']['patient_train'] + splits['c3po_mgh']['patient_train']
        patient_valid = splits['ewoc_mgh']['patient_valid'] + splits['c3po_mgh']['patient_valid']
    else:
        print('Splits file is of wrong structure!')

    if n_train_patients != 'all':
        patient_train = patient_train[:int(int(n_train_patients) * 0.9)]
        patient_valid = patient_valid[:int(int(n_train_patients) * 0.1)]

    train_ids = [t for t in working_ids if int(t.split('_')[0]) in patient_train]
    valid_ids = [t for t in working_ids if int(t.split('_')[0]) in patient_valid]

    # If scale_outputs, normalize by summary stats of training set
    if scale_outputs:
        wide_df_train = wide_df_selected[wide_df_selected['sample_id'].isin(train_ids)]
        output_labels_to_scale = np.array([l for l in output_labels if l not in cls_output_names])
        output_labels_to_scale = list(output_labels_to_scale[
                                          np.logical_and(wide_df_train[output_labels_to_scale].dtypes != 'object',
                                                         wide_df_train[output_labels_to_scale].dtypes != 'string')])
        logging.info(
            f'Not scaling classification or survival columns and columns containing strings/objects, unscaled columns: {[l for l in output_labels if l not in output_labels_to_scale]}')
        mean_outputs = np.mean(wide_df_train[output_labels_to_scale].values, axis=0)
        std_outputs = np.std(wide_df_train[output_labels_to_scale].values, axis=0)
        wide_df_selected.loc[:, output_labels_to_scale] = (wide_df_selected[output_labels_to_scale].values - mean_outputs) / std_outputs
        logging.info(mean_outputs)
        logging.info(std_outputs)

    train_ids = list(set(train_ids).intersection(set(working_ids)))
    valid_ids = list(set(valid_ids).intersection(set(working_ids)))

    # ---------- Adaptation for regression + classification ---------- #
    cls_category_map_dicts, cls_category_len_dict = process_class_categories(wide_df_selected, cls_output_names,
                                                                             var_type='output_labels')

    if pretrained_chkp_dir:
        cls_lbl_map_path = os.path.join(os.path.split(os.path.dirname(pretrained_chkp_dir))[0],
                                        'classification_class_label_mapping_per_output.json')
        define_new_heads = False
        if os.path.isfile(cls_lbl_map_path):
            with open(cls_lbl_map_path, 'r') as json_file:
                cls_category_signature_map_dicts = json.load(json_file)
            similar_cls = [c for c in cls_output_names if c in cls_category_signature_map_dicts.keys()]
            for c in similar_cls:
                if (len(cls_category_signature_map_dicts[c]) > len(cls_category_map_dicts[c])) and set(
                        cls_category_map_dicts[c].keys()).issubset(set(cls_category_signature_map_dicts[c].keys())):
                    cls_category_map_dicts[c] = cls_category_signature_map_dicts[c]
                    cls_category_len_dict[c] = len(cls_category_map_dicts[c])
                    logging.info(f'Using mapping from pretrained_chkp_dir for classification task on {c}')
                elif not set(
                        cls_category_map_dicts[c].keys()).issubset(set(cls_category_signature_map_dicts[c].keys())):
                    define_new_heads = True
        else:
            cls_category_signature_map_dicts = {}
            define_new_heads = True
    # ---------------------------------------------------------------- #

    INPUT_DD = LmdbEchoStudyVideoDataDescription(
        lmdb_folder,
        'image',
        [],
        n_input_frames,
        skip_modulo
    )

    if hf_task == 'survival':
        output_survival = [x for s_name in survival_var_names for x in [s_name+'_event', s_name+'_follow_up']]
    else:
        output_survival = []

    OUTPUT_DD = EcholabDataDescription(
        wide_df=wide_df_selected[['sample_id'] + output_labels + output_survival].drop_duplicates(),
        sample_id_column='sample_id',
        column_names=output_labels + output_survival,
        name='echolab',
        # ---------- Adaptation for regression + classification ---------- #
        cls_categories_map=cls_category_map_dicts if cls_output_names else None,
        # ---------- Adaptation for survival ---------- #
        survival_names=survival_var_names,
        survival_intervals=survival_intervals,
        survival_days_window=survival_days_window
        # ---------------------------------------------------------------- #
    )

    if BALANCED_SAMPLING_HF:
        selected_hf_task_idx
        wide_df_train0 = wide_df_selected.loc[np.logical_and(wide_df_selected['sample_id'].isin(train_ids), wide_df_selected[category_dictionaries['hf_diag_type'][hf_diag_type]] == selected_hf_task_idx[0])]
        train_ids0 = list(wide_df_train0['sample_id'])
        wide_df_train1 = wide_df_selected.loc[np.logical_and(wide_df_selected['sample_id'].isin(train_ids), wide_df_selected[category_dictionaries['hf_diag_type'][hf_diag_type]] == selected_hf_task_idx[1])]
        train_ids1 = list(wide_df_train1['sample_id'])
        # print(train_ids1)
        
        dataset_cls0 = tf.data.Dataset.from_tensor_slices(train_ids0)
        dataset_cls1 = tf.data.Dataset.from_tensor_slices(train_ids1)
        
        body_train_ids = tf.data.experimental.sample_from_datasets([dataset_cls0, dataset_cls1], weights=[0.5, 0.5]).shuffle(min(len(dataset_cls0),len(dataset_cls1)),
                                                                               reshuffle_each_iteration=True).batch(
            batch_size, drop_remainder=True)
    else:
        body_train_ids = tf.data.Dataset.from_tensor_slices(train_ids).shuffle(len(train_ids),
                                                                               reshuffle_each_iteration=True).batch(
            batch_size, drop_remainder=True)
    body_valid_ids = tf.data.Dataset.from_tensor_slices(valid_ids).shuffle(len(valid_ids),
                                                                           reshuffle_each_iteration=True).batch(
        batch_size, drop_remainder=True)

    n_train_steps = min(10000, len(train_ids) // batch_size)
    n_valid_steps = min(5000, len(valid_ids) // batch_size)

    # ---------- Adaptation for regression + classification ---------- #
    # Adapting tensor output sizes for classification heads
    num_classes = ([survival_intervals*2]*len(survival_var_names) + [output_reg_len] +
                   [cls_category_len_dict[c] if cls_category_len_dict[c]>2 else 1 for c in cls_category_map_dicts['cls_output_order']]) if (
            output_reg_len > 0) else ([survival_intervals*2]*len(survival_var_names) + [cls_category_len_dict[c] if cls_category_len_dict[c]>2 else 1 for c in cls_category_map_dicts['cls_output_order']])
    
    # print(num_classes)
    
    if len(num_classes) > 1:
        output_signatures = (
            tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32),
            tuple([tf.TensorSpec(shape=(batch_size, n_c), dtype=tf.float32)
                   for n_c in num_classes])
        )
    else:
        output_signatures = (
            tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, num_classes[0]) if num_classes[0] > 1 else (batch_size,), dtype=tf.float32)
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
        ),
        num_parallel_calls=tf.data.AUTOTUNE).repeat(epochs).prefetch(tf.data.AUTOTUNE)

    io_valid_ds = body_valid_ids.interleave(
        lambda sample_ids: tf.data.Dataset.from_generator(
            DDGenerator(
                INPUT_DD,
                OUTPUT_DD
            ),
            output_signature=output_signatures,
            args=(sample_ids,)
        ),
        num_parallel_calls=tf.data.AUTOTUNE).repeat(epochs).prefetch(tf.data.AUTOTUNE)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with (mirrored_strategy.scope()):
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
                     'add_dense': {'regressor': add_separate_dense_reg, 'classifier': add_separate_dense_cls},
                     'survival_shapes': {s_name: survival_intervals for s_name in survival_var_names} if survival_var_names else {}
                     }

        model = create_regressor_classifier(encoder, **func_args)
        # ---------------------------------------------------------------- #

        if pretrained_chkp_dir:
            signature_model_param_path = os.path.join(os.path.split(os.path.dirname(pretrained_chkp_dir))[0],
                                                      'model_params.json')
            if os.path.isfile(signature_model_param_path):
                f = open(signature_model_param_path)
                signature_model_params = json.load(f)
            else:
                signature_model_params = {}
                
            sig_add_separate_dense_reg = signature_model_params[
                'add_separate_dense_reg'] if 'add_separate_dense_reg' in signature_model_params.keys() else False
            sig_add_separate_dense_cls = signature_model_params[
                'add_separate_dense_cls'] if 'add_separate_dense_cls' in signature_model_params.keys() else False
            output_signature_labels_types = signature_model_params[
                'output_labels_types'] if 'output_labels_types' in signature_model_params.keys() else 'r'
            output_signature_labels = signature_model_params[
                'output_labels'] if 'output_labels' in signature_model_params.keys() else ['no_info']
            logging.info(f'output_labels of loaded model: {output_signature_labels}')

            output_signature_labels, output_signature_reg_len, cls_output_signature_names = process_labels_types(
                output_signature_labels, output_signature_labels_types, var_type='output_signature_labels')

            sig_survival_intervals = signature_model_params[
                'survival_intervals'] if 'survival_var_names' in signature_model_params.keys() else 0
            sig_survival_var_names = signature_model_params[
                'survival_var_names'] if 'survival_var_names' in signature_model_params.keys() else []


            if 'c' in output_signature_labels_types.lower():
                cls_category_signature_len_dict = {}
                for c_lbl in cls_category_signature_map_dicts['cls_output_order']:
                    cls_category_signature_len_dict[c_lbl] = len(cls_category_signature_map_dicts[c_lbl])
            else:
                cls_category_signature_len_dict = {}
                

            model = create_regressor_classifier(
                encoder,
                input_shape=(n_input_frames, 224, 224, 3),
                trainable=not fine_tune,
                n_output_features=output_signature_reg_len,
                categories=cls_category_signature_len_dict,
                category_order=cls_category_signature_map_dicts[
                    'cls_output_order'] if cls_category_signature_len_dict else None,
                add_dense={'regressor': sig_add_separate_dense_reg, 'classifier': sig_add_separate_dense_cls},
                survival_shapes={s_name: sig_survival_intervals for s_name in
                                 sig_survival_var_names} if sig_survival_var_names else {}
            )
            model.load_weights(pretrained_chkp_dir)

            if (output_labels != output_signature_labels) or (output_signature_reg_len != output_reg_len) or (
                    cls_output_signature_names != cls_output_names) or define_new_heads:
                logging.info('Redefining regression and/or classification heads due to differences in outputs used')
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
        # -------------------------------------------------------------------------------------

        loss = {'cls_' + k: tf.keras.losses.CategoricalCrossentropy() if cls_category_len_dict[k]>2 else tf.keras.losses.BinaryCrossentropy() for k in cls_category_len_dict.keys()}
        metrics = {'cls_' + k: tf.keras.metrics.CategoricalAccuracy() if cls_category_len_dict[k]>2 else [tf.keras.metrics.BinaryAccuracy(), 
                                # tf.keras.metrics.TruePositives(name='tp'),
                                # tf.keras.metrics.FalsePositives(name='fp'),
                                # tf.keras.metrics.TrueNegatives(name='tn'),
                                # tf.keras.metrics.FalseNegatives(name='fn'),
                                tf.keras.metrics.AUC(name='prc', curve='PR')] for k in cls_category_len_dict.keys()}
        if output_reg_len > 0:
            loss['echolab'] = tf.keras.losses.MeanSquaredError()
            metrics['echolab'] = tf.keras.metrics.MeanAbsoluteError()

        # def c_index_metric(y_true, y_pred):
        #     c_index, concordant, discordant, tied_risk, tied_time = concordance_index(y_pred, y_true)
        #     return c_index
        def auprc_survival_metric_param(intervals):
            def auprc_survival(y_true, y_pred):
                y_pred_val = 1 - tf.math.cumprod(y_pred[:, :intervals], axis=1)[:, -1]
                y_true_val = tf.math.cumsum(y_true[:, intervals:], axis=1)[:, -1]
                true_positives = K.sum(K.round(K.clip(y_true_val * y_pred_val, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred_val, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision    
            return auprc_survival
            
        if survival_var_names:
            # TODO: got to here with survival curve updates
            for s_name in survival_var_names:
                # print(survival_intervals)
                loss['survival_'+s_name] = survival_likelihood_loss(survival_intervals)
                metrics['survival_'+s_name] = auprc_survival_metric_param(survival_intervals)
                # metrics['survival_'+s_name] = c_index_metric

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights if loss_weights else None
        )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    io_train_ds = io_train_ds.with_options(options)
    io_valid_ds = io_valid_ds.with_options(options)

    fine_tune_string = f'_fine_tune' if fine_tune else ''
    output_folder = os.path.join(output_dir,
                                 f'{save_tag}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}_{lmdb_vois}_{olabels}_{n_input_frames}frames{fine_tune_string}_{n_train_patients}')

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
        
        cls_category_map_dicts_tmp = cls_category_map_dicts.copy()
        for el in cls_category_map_dicts_tmp.keys():
            if isinstance(cls_category_map_dicts_tmp[el], dict):
                cls_category_map_dicts_tmp[el] = {str(k):v for k,v in cls_category_map_dicts_tmp[el].items()}
        
        with open(f'{output_folder}/classification_class_label_mapping_per_output.json', 'w') as json_file:
            json.dump(cls_category_map_dicts_tmp, json_file)
    # ---------------------------------------------------------------- #
    
    if class_weights:
        class_weight = {0: class_weights[0], 1: class_weights[1]}
    else:
        class_weight = None
    
    es_flags = {'es_patience': es_patience, 'es_loss2monitor': es_loss2monitor}

    logging.info(model.summary())
    trained_model = train_model(
        model,
        io_train_ds,
        io_valid_ds,
        epochs,
        n_train_steps,
        n_valid_steps,
        output_folder,
        es_flags,
        class_weight=class_weight
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
    parser.add_argument('--lmdb_folder',  action='append', type=str)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--pretrained_chkp_dir', type=str)
    parser.add_argument('--movinet_chkp_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--splits_file')
    parser.add_argument('--adam', default=None, type=float)
    parser.add_argument('--scale_outputs', action='store_true')
    parser.add_argument('--es_patience', default=3, type=int,
                        help='Number of epochs with no change before early stopping.')
    parser.add_argument('--es_loss2monitor', default='val_loss', type=str,
                        help='Loss on which the early stopping will be based, options are "val_loss", "val_echolab_loss" for regression loss, or "val_cls_COLUMN-NAME_loss" for classification loss.')
    # ---------- Adaptation for regression + classification ---------- #
    parser.add_argument('--output_labels_types', default='r', type=str,
                        help='A string indicating task types: r for regression, c for classification. Should be of length 1 or the same length of the specified output_labels variable, e.g. "r" or "rrcr".')
    parser.add_argument('--add_separate_dense_reg', action='store_true',
                        help='Adds an additional dense layer trained separately for the regression head')
    parser.add_argument('--add_separate_dense_cls', action='store_true',
                        help='Adds an additional dense layer trained separately for the classification head')
    parser.add_argument('-lw', '--loss_weights', action='append', type=float,
                        help='Loss weights, number of weights to specify should be: No. classification tasks (columns) + 1 if there are regression variables. For example, for output_labels_types="rrcc", the length should be 2+1=3.')
    # ---------------------------------------------------------------- #
    # ---------------------- Echo 2 HF related ----------------------- #
    parser.add_argument('-hct', '--hf_task', type=str, default=None, choices=category_dictionaries['hf_task'].keys())
    parser.add_argument('-hdt', '--hf_diag_type', type=str, default='hf_nlp', choices=category_dictionaries['hf_diag_type'].keys())
    parser.add_argument('--save_tag', type=str, default=None)
    parser.add_argument('-cw', '--class_weights', action='append', type=float, default=None)  # Currently for binary classification only!
    # ---------------------------------------------------------------- #
    # ---------------------- Survival curve related ----------------------- #
    parser.add_argument('--survival_var_names', action='append', type=str)
    parser.add_argument('--survival_intervals', type=int, default=25)
    parser.add_argument('--survival_days_window', type=int, default=3650)
    # ---------------------------------------------------------------- #
    args = parser.parse_args()

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
        es_patience=args.es_patience,
        es_loss2monitor=args.es_loss2monitor,
        # ---------- Adaptation for regression + classification ---------- #
        output_labels_types=args.output_labels_types,
        add_separate_dense_reg=args.add_separate_dense_reg,
        add_separate_dense_cls=args.add_separate_dense_cls,
        loss_weights=args.loss_weights,
        # ---------------------------------------------------------------- #
        # ---------------------- Echo 2 HF related ----------------------- #
        hf_task=args.hf_task,
        hf_diag_type=args.hf_diag_type,
        save_tag=args.save_tag,
        class_weights=args.class_weights,
        # ---------------------------------------------------------------- #
        # ---------------------- Survival curve related ----------------------- #
        survival_var_names=args.survival_var_names,
        survival_intervals=args.survival_intervals,
        survival_days_window=args.survival_days_window
        # ---------------------------------------------------------------- #
    )
