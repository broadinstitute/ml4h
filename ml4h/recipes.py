# recipes.py

# Imports
import os
import csv
import copy
import h5py
import glob
import logging
import datetime
import numpy as np
import pandas as pd

from functools import reduce
from google.cloud import storage
from timeit import default_timer as timer
from collections import Counter, defaultdict

from ml4h.arguments import parse_args
from ml4h.models.inspect import saliency_map
from ml4h.optimizers import find_learning_rate
from ml4h.defines import TENSOR_EXT, MODEL_EXT
from ml4h.models.train import train_model_from_generators
from ml4h.tensormap.tensor_map_maker import write_tensor_maps
from ml4h.tensorize.tensor_writer_mgb import write_tensors_mgb
from ml4h.models.model_factory import make_multimodal_multitask_model
from ml4h.ml4ht_integration.tensor_generator import TensorMapDataLoader2
from ml4h.explorations import test_labels_to_label_map, infer_with_pixels
from ml4h.tensor_generators import BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX
from ml4h.explorations import test_labels_to_label_map, infer_with_pixels, latent_space_dataframe, infer_stats_from_segmented_regions
from ml4h.explorations import mri_dates, ecg_dates, predictions_to_pngs, sample_from_language_model
from ml4h.plots import plot_reconstruction, plot_saliency_maps, plot_partners_ecgs, plot_ecg_rest_mp
from ml4h.explorations import plot_while_learning, plot_histograms_of_tensors_in_pdf, explore, pca_on_tsv
from ml4h.models.legacy_models import get_model_inputs_outputs, make_shallow_model, make_hidden_layer_model
from ml4h.tensor_generators import TensorGenerator, test_train_valid_tensor_generators, big_batch_from_minibatch_generator
from ml4h.data_descriptions import dataframe_data_description_from_tensor_map, ECGDataDescription, DataFrameDataDescription
from ml4h.metrics import get_roc_aucs, get_precision_recall_aucs, get_pearson_coefficients, log_aucs, log_pearson_coefficients
from ml4h.plots import evaluate_predictions, plot_scatters, plot_rocs, plot_precision_recalls, subplot_roc_per_class, plot_tsne, plot_survival, plot_dice
from ml4h.plots import plot_reconstruction, plot_hit_to_miss_transforms, plot_saliency_maps, plot_partners_ecgs, plot_ecg_rest_mp

from ml4h.plots import subplot_rocs, subplot_comparison_rocs, subplot_scatters, subplot_comparison_scatters, plot_prediction_calibrations
from ml4h.models.legacy_models import make_character_model_plus, embed_model_predict, make_siamese_model, legacy_multimodal_multitask_model
from ml4h.plots import evaluate_predictions, plot_scatters, plot_rocs, plot_precision_recalls, subplot_roc_per_class, plot_tsne, plot_survival
from ml4h.tensorize.tensor_writer_ukbb import write_tensors, append_fields_from_csv, append_gene_csv, write_tensors_from_dicom_pngs, write_tensors_from_ecg_pngs

from ml4ht.data.util.date_selector import DATE_OPTION_KEY
from ml4ht.data.sample_getter import DataDescriptionSampleGetter
from ml4ht.data.data_loader import SampleGetterIterableDataset, shuffle_get_epoch

from torch.utils.data import DataLoader

def run(args):
    start_time = timer()  # Keep track of elapsed execution time
    try:
        if 'tensorize' == args.mode:
            write_tensors(
                args.id, args.xml_folder, args.zip_folder, args.output_folder, args.tensors, args.dicoms, args.mri_field_ids, args.xml_field_ids,
                args.write_pngs, args.min_sample_id, args.max_sample_id, args.min_values,
            )
        elif 'tensorize_pngs' == args.mode:
            write_tensors_from_dicom_pngs(args.tensors, args.dicoms, args.app_csv, args.dicom_series, args.min_sample_id, args.max_sample_id, args.x, args.y)
        elif 'tensorize_ecg_pngs' == args.mode:
            write_tensors_from_ecg_pngs(args.tensors, args.xml_folder, args.min_sample_id, args.max_sample_id)
        elif 'tensorize_partners' == args.mode:
            write_tensors_mgb(args.xml_folder, args.tensors, args.num_workers)
        elif 'explore' == args.mode:
            explore(args)
        elif 'train' == args.mode:
            train_multimodal_multitask(args)
        elif 'train_legacy' == args.mode:
            train_legacy(args)
        elif 'train_xdl' == args.mode:
            train_xdl(args)
        elif 'test' == args.mode:
            test_multimodal_multitask(args)
        elif 'compare' == args.mode:
            compare_multimodal_multitask_models(args)
        elif 'infer' == args.mode:
            infer_multimodal_multitask(args)
        elif 'infer_hidden' == args.mode:
            infer_hidden_layer_multimodal_multitask(args)
        elif 'infer_pixels' == args.mode:
            infer_with_pixels(args)
        elif 'infer_stats_from_segmented_regions' == args.mode:
            infer_stats_from_segmented_regions(args)
        elif 'infer_encoders' == args.mode:
            infer_encoders_block_multimodal_multitask(args)
        elif 'infer_xdl' == args.mode:
            infer_xdl(args)
        elif 'test_scalar' == args.mode:
            test_multimodal_scalar_tasks(args)
        elif 'compare_scalar' == args.mode:
            compare_multimodal_scalar_task_models(args)
        elif 'plot_predictions' == args.mode:
            plot_predictions(args)
        elif 'plot_while_training' == args.mode:
            plot_while_training(args)
        elif 'plot_saliency' == args.mode:
            saliency_maps(args)
        elif 'plot_mri_dates' == args.mode:
            mri_dates(args.tensors, args.output_folder, args.id)
        elif 'plot_ecg_dates' == args.mode:
            ecg_dates(args.tensors, args.output_folder, args.id)
        elif 'plot_histograms' == args.mode:
            plot_histograms_of_tensors_in_pdf(args.id, args.tensors, args.output_folder, args.max_samples)
        elif 'plot_resting_ecgs' == args.mode:
            plot_ecg_rest_mp(args.tensors, args.min_sample_id, args.max_sample_id, args.output_folder, args.num_workers)
        elif 'plot_partners_ecgs' == args.mode:
            plot_partners_ecgs(args)
        elif 'train_shallow' == args.mode:
            train_shallow_model(args)
        elif 'train_char' == args.mode:
            train_char_model(args)
        elif 'train_siamese' == args.mode:
            train_siamese_model(args)
        elif 'write_tensor_maps' == args.mode:
            write_tensor_maps(args)
        elif 'append_continuous_csv' == args.mode:
            append_fields_from_csv(args.tensors, args.app_csv, 'continuous', ',')
        elif 'append_categorical_csv' == args.mode:
            append_fields_from_csv(args.tensors, args.app_csv, 'categorical', ',')
        elif 'append_continuous_tsv' == args.mode:
            append_fields_from_csv(args.tensors, args.app_csv, 'continuous', '\t')
        elif 'append_categorical_tsv' == args.mode:
            append_fields_from_csv(args.tensors, args.app_csv, 'categorical', '\t')
        elif 'append_gene_csv' == args.mode:
            append_gene_csv(args.tensors, args.app_csv, ',')
        elif 'pca_on_hidden_inference' == args.mode:
            pca_on_hidden_inference(args)
        elif 'find_learning_rate' == args.mode:
            _find_learning_rate(args)
        elif 'find_learning_rate_and_train' == args.mode:
            args.learning_rate = _find_learning_rate(args)
            if not args.learning_rate:
                raise ValueError('Could not find learning rate.')
            train_legacy(args)
        else:
            raise ValueError('Unknown mode:', args.mode)

    except Exception as e:
        logging.exception(e)
        
    if args.gcs_cloud_bucket is not None:
         save_to_google_cloud(args)
    
    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info("Executed the '{}' operation in {:.2f} seconds".format(args.mode, elapsed_time))

def save_to_google_cloud(args):

    """
    Function to transfer all files and result from output folder to designated location google cloud bucket.
    Args:
        args.output_folder(str): Local folder path where all files and results generated from run are saved:
        args.gcs_cloud_bucket (str): Google cloud bucket folder path where all files will be stored
    """
    #Instantiate a storage client
    client = storage.Client()

    #get the bucket
    split_path = args.gcs_cloud_bucket.split("/")
    bucket_name = split_path[0]
    blob_path = "/".join(split_path[1:])

    bucket = client.get_bucket(bucket_name)

    # uploading all files from local to server
    for root,_,files in os.walk(args.output_folder):
        for filename in files:
            local_file_path = os.path.join(root,filename)
            blob = bucket.blob(blob_path+filename)
            blob.upload_from_filename(local_file_path)
            print("Uploaded from local file path {} to {} in google cloud bucket".format(local_file_path,filename))





def _find_learning_rate(args) -> float:
    schedule = args.learning_rate_schedule
    args.learning_rate_schedule = None  # learning rate schedule interferes with setting lr done by find_learning_rate
    generate_train, _, _ = test_train_valid_tensor_generators(**args.__dict__)
    model = legacy_multimodal_multitask_model(**args.__dict__)
    lr = find_learning_rate(model, generate_train, args.training_steps, os.path.join(args.output_folder, args.id))
    args.learning_rate_schedule = schedule
    return lr


def train_legacy(args):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = legacy_multimodal_multitask_model(**args.__dict__)
    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size, args.epochs,
        args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels, args.tensor_maps_out,
        save_last_model=args.save_last_model,
    )

    out_path = os.path.join(args.output_folder, args.id + '/')
    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    return _predict_and_evaluate(
        model, test_data, test_labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, out_path, test_paths, args.embed_visualization, args.alpha,
        args.dpi, args.plot_width, args.plot_height,
    )


def train_multimodal_multitask(args):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model, encoders, decoders, merger = make_multimodal_multitask_model(**args.__dict__)
    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size, args.epochs,
        args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels, args.tensor_maps_out,
        save_last_model=args.save_last_model,
    )
    for tm in encoders:
        encoders[tm].save(f'{args.output_folder}{args.id}/encoder_{tm.name}.h5')
    for tm in decoders:
        decoders[tm].save(f'{args.output_folder}{args.id}/decoder_{tm.name}.h5')
    if merger:
        merger.save(f'{args.output_folder}{args.id}/merger.h5')

    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    performance_metrics = _predict_and_evaluate(
        model, test_data, test_labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, os.path.join(args.output_folder, args.id + '/'), test_paths,
        args.embed_visualization, args.alpha, args.dpi, args.plot_width, args.plot_height,
    )

    predictions_list = model.predict(test_data)
    samples = min(args.test_steps * args.batch_size, 12)
    out_path = os.path.join(args.output_folder, args.id, 'reconstructions/')
    if len(args.tensor_maps_out) == 1:
        predictions_list = [predictions_list]
    predictions_dict = {name: pred for name, pred in zip(model.output_names, predictions_list)}
    logging.info(f'Predictions and shapes are: {[(p, predictions_dict[p].shape) for p in predictions_dict]}')

    for i, etm in enumerate(encoders):
        embed = encoders[etm].predict(test_data[etm.input_name()])
        if etm.output_name() in predictions_dict:
            plot_reconstruction(etm, test_data[etm.input_name()], predictions_dict[etm.output_name()], out_path, test_paths, samples)
        for dtm in decoders:
            reconstruction = decoders[dtm].predict(embed)
            logging.info(f'{dtm.name} has prediction shape: {reconstruction.shape} from embed shape: {embed.shape}')
            my_out_path = os.path.join(out_path, f'decoding_{dtm.name}_from_{etm.name}/')
            os.makedirs(os.path.dirname(my_out_path), exist_ok=True)
            if dtm.axes() > 1:
                plot_reconstruction(dtm, test_labels[dtm.output_name()], reconstruction, my_out_path, test_paths, samples)
            else:
                evaluate_predictions(
                    dtm, reconstruction, test_labels[dtm.output_name()], {}, dtm.name, my_out_path,
                    test_paths, dpi=args.dpi, width=args.plot_width, height=args.plot_height,
                )
    return performance_metrics


def test_multimodal_multitask(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = legacy_multimodal_multitask_model(**args.__dict__)
    out_path = os.path.join(args.output_folder, args.id + '/')
    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    return _predict_and_evaluate(
        model, data, labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected, args.batch_size,
        args.hidden_layer, out_path, paths, args.embed_visualization, args.alpha,
        args.dpi, args.plot_width, args.plot_height,
    )


def test_multimodal_scalar_tasks(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = legacy_multimodal_multitask_model(**args.__dict__)
    p = os.path.join(args.output_folder, args.id + '/')
    return _predict_scalars_and_evaluate_from_generator(
        model, generate_test, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected, args.test_steps,
        args.hidden_layer, p, args.alpha, args.dpi, args.plot_width, args.plot_height,
    )


def compare_multimodal_multitask_models(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    models_inputs_outputs = get_model_inputs_outputs(args.model_files, args.tensor_maps_in, args.tensor_maps_out)
    input_data, output_data, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    common_outputs = _get_common_outputs(models_inputs_outputs, 'output')
    predictions = _get_predictions(args, models_inputs_outputs, input_data, common_outputs, 'input', 'output')
    _calculate_and_plot_prediction_stats(args, predictions, output_data, paths)


def compare_multimodal_scalar_task_models(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    models_io = get_model_inputs_outputs(args.model_files, args.tensor_maps_in, args.tensor_maps_out)
    outs = _get_common_outputs(models_io, "output")
    predictions, labels, paths = _scalar_predictions_from_generator(args, models_io, generate_test, args.test_steps, outs, "input", "output")
    _calculate_and_plot_prediction_stats(args, predictions, labels, paths)


def standardize_by_sample_ecg(ecg, _):
    """Z Score ECG"""
    return (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)


def train_xdl(args):
    mrn_df = pd.read_csv(args.app_csv)
    if 'start_fu_age' in mrn_df:
        mrn_df['age_in_days'] = pd.to_timedelta(mrn_df.start_fu_age).dt.days
    elif 'start_fu' in mrn_df:
        mrn_df['age_in_days'] = pd.to_timedelta(mrn_df.start_fu).dt.days
    for ot in args.tensor_maps_out:
        mrn_df = mrn_df[mrn_df[ot.name].notna()]
    mrn_df = mrn_df.set_index('sample_id')

    output_dds = [dataframe_data_description_from_tensor_map(tmap, mrn_df) for tmap in args.tensor_maps_out]

    ecg_dd = ECGDataDescription(
        args.tensors,
        name=args.tensor_maps_in[0].input_name(),
        ecg_len=5000,  # all ECGs will be linearly interpolated to be this length
        transforms=[standardize_by_sample_ecg],  # these will be applied in order
        # data will be automatically localized from s3
    )

    def option_picker(sample_id, data_descriptions):
        ecg_dts = ecg_dd.get_loading_options(sample_id)
        htn_dt = output_dds[0].get_loading_options(sample_id)[0]['start_fu_datetime']
        min_ecg_dt = htn_dt - pd.to_timedelta("1095d")
        max_ecg_dt = htn_dt + pd.to_timedelta("1095d")
        dates = []
        for dt in ecg_dts:
            if min_ecg_dt <= dt[DATE_OPTION_KEY] <= max_ecg_dt:
                dates.append(dt)
        if len(dates) == 0:
            raise ValueError('No matching dates')
        chosen_dt = np.random.choice(dates)
        chosen_dt['day_delta'] = (htn_dt - chosen_dt[DATE_OPTION_KEY]).days
        return {dd: chosen_dt for dd in data_descriptions}

    sg = DataDescriptionSampleGetter(
        input_data_descriptions=[ecg_dd],  # what we want a model to use as input data
        output_data_descriptions=output_dds,  # what we want a model to predict from the input data
        option_picker=option_picker,
    )
    model, encoders, decoders, merger = make_multimodal_multitask_model(**args.__dict__)

    train_ids = list(mrn_df[mrn_df.split == 'train'].index)
    valid_ids = list(mrn_df[mrn_df.split == 'valid'].index)
    test_ids = list(mrn_df[mrn_df.split == 'test'].index)

    train_dataset = SampleGetterIterableDataset(sample_ids=list(train_ids), sample_getter=sg,
                                                get_epoch=shuffle_get_epoch)
    valid_dataset = SampleGetterIterableDataset(sample_ids=list(valid_ids), sample_getter=sg,
                                                get_epoch=shuffle_get_epoch)

    num_train_workers = int(args.training_steps / (args.training_steps + args.validation_steps) * args.num_workers) or (1 if args.num_workers else 0)
    num_valid_workers = int(args.validation_steps / (args.training_steps + args.validation_steps) * args.num_workers) or (1 if args.num_workers else 0)

    generate_train = TensorMapDataLoader2(
        batch_size=args.batch_size, input_maps=args.tensor_maps_in, output_maps=args.tensor_maps_out,
        dataset=train_dataset,
        num_workers=num_train_workers,
    )
    generate_valid = TensorMapDataLoader2(
        batch_size=args.batch_size, input_maps=args.tensor_maps_in, output_maps=args.tensor_maps_out,
        dataset=valid_dataset,
        num_workers=num_valid_workers,
    )

    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size, args.epochs,
        args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels, args.tensor_maps_out,
        save_last_model=args.save_last_model,
    )
    for tm in encoders:
        encoders[tm].save(f'{args.output_folder}{args.id}/encoder_{tm.name}.h5')
    for tm in decoders:
        decoders[tm].save(f'{args.output_folder}{args.id}/decoder_{tm.name}.h5')
    if merger:
        merger.save(f'{args.output_folder}{args.id}/merger.h5')


def datetime_to_float(d):
    return pd.to_datetime(d, utc=True).timestamp()


def float_to_datetime(fl):
    return pd.to_datetime(fl, unit='s', utc=True)


def infer_from_dataloader(dataloader, model, tensor_maps_out, max_batches=125000):
    dataloader_iterator = iter(dataloader)
    space_dict = defaultdict(list)
    for i in range(max_batches):
        try:
            data, target = next(dataloader_iterator)
            for k in data:
                data[k] = np.array(data[k])
            prediction = model.predict(data, verbose=0)
            if len(model.output_names) == 1:
                prediction = [prediction]
            predictions_dict = {name: pred for name, pred in zip(model.output_names, prediction)}
            for b in range(prediction[0].shape[0]):
                for otm in tensor_maps_out:
                    y = predictions_dict[otm.output_name()]
                    if otm.is_categorical():
                        space_dict[f'{otm.name}_prediction'].append(y[b, 1])
                    elif otm.is_continuous():
                        space_dict[f'{otm.name}_prediction'].append(y[b, 0])
                    elif otm.is_survival_curve():
                        intervals = otm.shape[-1] // 2
                        days_per_bin = 1 + (2*otm.days_window) // intervals
                        predicted_survivals = np.cumprod(y[:, :intervals], axis=1)
                        space_dict[f'{otm.name}_prediction'].append(str(1 - predicted_survivals[0, -1]))
                        sick = np.sum(target[otm.output_name()][:, intervals:], axis=-1)
                        follow_up = np.cumsum(target[otm.output_name()][:, :intervals], axis=-1)[:, -1] * days_per_bin
                        space_dict[f'{otm.name}_event'].append(str(sick[0]))
                        space_dict[f'{otm.name}_follow_up'].append(str(follow_up[0]))
                for k in target:
                    if k in ['MRN', 'linker_id', 'is_c3po', 'output_age_in_days_continuous' ]:
                        space_dict[f'{k}'].append(target[k][b].numpy())
                    elif k in ['datetime']:
                        space_dict[f'{k}'].append(float_to_datetime(int(target[k][b].numpy())))
                    else:
                        space_dict[f'{k}'].append(target[k][b, -1].numpy())
            if i % 100 == 0:
                logging.info(f'Inferred on {i} batches, {len(space_dict[k])} rows')
        except StopIteration:
            logging.info(f'Inferred on all {i} batches.')
            break
    return pd.DataFrame.from_dict(space_dict)


def infer_xdl(args):
    mrn_df = pd.read_csv(args.app_csv)
    if 'start_fu_age' in mrn_df:
        mrn_df['age_in_days'] = pd.to_timedelta(mrn_df.start_fu_age).dt.days
    elif 'start_fu' in mrn_df:
        mrn_df['age_in_days'] = pd.to_timedelta(mrn_df.start_fu).dt.days
    mrn_df = mrn_df.rename(columns={'Dem.Gender.no_filter_x': 'sex', 'Dem.Gender.no_filter': 'sex'})
    mrn_df['is_c3po'] = mrn_df.cohort == 'c3po'
    for ot in args.tensor_maps_out:
        mrn_df = mrn_df[mrn_df[ot.name].notna()]
    mrn_df = mrn_df.set_index('sample_id')

    output_dds = [dataframe_data_description_from_tensor_map(tmap, mrn_df) for tmap in args.tensor_maps_out]

    output_dds.append(DataFrameDataDescription(mrn_df, col="MRN"))
    output_dds.append(DataFrameDataDescription(mrn_df, col="linker_id"))
    output_dds.append(DataFrameDataDescription(mrn_df, col="is_c3po"))
    output_dds.append(DataFrameDataDescription(mrn_df, col="datetime", process_col=datetime_to_float))

    ecg_dd = ECGDataDescription(
        args.tensors,
        name=args.tensor_maps_in[0].input_name(),
        ecg_len=5000,  # all ECGs will be linearly interpolated to be this length
        transforms=[standardize_by_sample_ecg],  # these will be applied in order
    )

    def test_option_picker(sample_id, data_descriptions):
        ecg_dts = ecg_dd.get_loading_options(sample_id)
        htn_dt = output_dds[0].get_loading_options(sample_id)[0]['start_fu_datetime']
        min_ecg_dt = htn_dt - pd.to_timedelta("1095d")
        max_ecg_dt = htn_dt - pd.to_timedelta("1d")
        dates = []
        for dt in ecg_dts:
            if min_ecg_dt <= dt[DATE_OPTION_KEY] <= max_ecg_dt:
                dates.append(dt)
        if len(dates) == 0:
            raise ValueError('No matching dates')
        chosen_dt = dates[-1]
        return {dd: chosen_dt for dd in data_descriptions}

    sg = DataDescriptionSampleGetter(
        input_data_descriptions=[ecg_dd],  # what we want a model to use as input data
        output_data_descriptions=output_dds,  # what we want a model to predict from the input data
        option_picker=test_option_picker,
    )

    dataset = SampleGetterIterableDataset(sample_ids=list(mrn_df.index), sample_getter=sg, get_epoch=shuffle_get_epoch)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size)

    model, _, _, _ = make_multimodal_multitask_model(**args.__dict__)
    infer_df = infer_from_dataloader(dataloader, model, args.tensor_maps_out)
    if 'mgh' in args.tensors:
        hospital = 'mgh'
        infer_df = infer_df.rename(columns={'MRN': 'MGH_MRN'})
        infer_df.MGH_MRN = infer_df.MGH_MRN.astype(int)
    else:
        hospital = 'bwh'
        infer_df = infer_df.rename(columns={'MRN': 'BWH_MRN'})
        infer_df.BWH_MRN = infer_df.BWH_MRN.astype(int)

    infer_df.linker_id = infer_df.linker_id.astype(int)
    names = '_'.join([otm.name for otm in args.tensor_maps_out])
    now_string = datetime.datetime.now().strftime('%Y_%m_%d')
    out_file = f'{args.output_folder}/{args.id}/infer_{names}_{hospital}_v{now_string}.tsv'
    infer_df.to_csv(out_file, sep='\t', index=False)
    logging.info(f'Infer dataframe head: {infer_df.head()}  \n\n Saved inferences to: {out_file}')


def _make_tmap_nan_on_fail(tmap):
    """
    Builds a copy TensorMap with a tensor_from_file that returns nans on errors instead of raising an error
    """
    new_tmap = copy.deepcopy(tmap)
    new_tmap.validator = lambda _, x, hd5: x  # prevent failure caused by validator

    def _tff(tm, hd5, dependents=None):
        try:
            return tmap.tensor_from_file(tm, hd5, dependents)
        except (IndexError, KeyError, ValueError, OSError, RuntimeError):
            return np.full(shape=tm.shape, fill_value=np.nan)

    new_tmap.tensor_from_file = _tff
    return new_tmap


def inference_file_name(output_folder: str, id_: str) -> str:
    return os.path.join(output_folder, id_, 'inference_' + id_ + '.tsv')


def infer_multimodal_multitask(args):
    stats = Counter()
    tensor_paths_inferred = set()
    inference_tsv = inference_file_name(args.output_folder, args.id)
    tsv_style_is_genetics = 'genetics' in args.tsv_style

    model = legacy_multimodal_multitask_model(**args.__dict__)
    no_fail_tmaps_out = [_make_tmap_nan_on_fail(tmap) for tmap in args.tensor_maps_out]
    tensor_paths = _tensor_paths_from_sample_csv(args.tensors, args.sample_csv)
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    generate_test = TensorGenerator(
        1, args.tensor_maps_in, no_fail_tmaps_out, tensor_paths, num_workers=0,
        cache_size=0, keep_paths=True, mixup_alpha=args.mixup_alpha,
    )

    logging.info(f"Found {len(tensor_paths)} tensor paths.")
    generate_test.set_worker_paths(tensor_paths)
    output_maps = {tm.output_name(): tm for tm in no_fail_tmaps_out}
    with open(inference_tsv, mode='w') as inference_file:
        # TODO: csv.DictWriter is much nicer for this
        inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['sample_id']
        if tsv_style_is_genetics:
            header = ['FID', 'IID']
        for ot in model.output_names:
            otm = output_maps[ot]
            logging.info(f"Got ot  {ot} and otm {otm}  ot and otm {otm.name} ot  and otm {otm.channel_map} channel_map and otm {otm.interpretation}.")
            if len(otm.shape) == 1 and otm.is_continuous():
                header.extend([otm.name + '_prediction', otm.name + '_actual'])
            elif len(otm.shape) == 1 and otm.is_categorical():
                channel_columns = []
                for k in otm.channel_map:
                    channel_columns.append(otm.name + '_' + k + '_prediction')
                    channel_columns.append(otm.name + '_' + k + '_actual')
                header.extend(channel_columns)
            elif otm.is_survival_curve():
                header.extend([otm.name + '_prediction', otm.name + '_actual', otm.name + '_follow_up'])
        inference_writer.writerow(header)

        while True:
            batch = next(generate_test)
            input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
            if tensor_paths[0] in tensor_paths_inferred:
                next(generate_test)  # this prints end of epoch info
                logging.info(f"Inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                break
            prediction = model.predict(input_data, verbose=0)
            if len(no_fail_tmaps_out) == 1:
                prediction = [prediction]
            predictions_dict = {name: pred for name, pred in zip(model.output_names, prediction)}
            sample_id = os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')
            csv_row = [sample_id]
            if tsv_style_is_genetics:
                csv_row *= 2
            for ot in model.output_names:
                y = predictions_dict[ot]
                otm = output_maps[ot]
                if len(otm.shape) == 1 and otm.is_continuous():
                    csv_row.append(str(otm.rescale(y)[0][0]))  # first index into batch then index into the 1x1 structure
                    if ((otm.sentinel is not None and otm.sentinel == output_data[otm.output_name()][0][0])
                            or np.isnan(output_data[otm.output_name()][0][0])):
                        csv_row.append("NA")
                    else:
                        csv_row.append(str(otm.rescale(output_data[otm.output_name()])[0][0]))
                elif len(otm.shape) == 1 and otm.is_categorical():
                    for k, i in otm.channel_map.items():
                        try:
                            csv_row.append(str(y[0][otm.channel_map[k]]))
                            actual = output_data[otm.output_name()][0][i]
                            csv_row.append("NA" if np.isnan(actual) else str(actual))
                        except IndexError:
                            logging.debug(f'index error at {otm.name} item {i} key {k} with cm: {otm.channel_map} y is {y.shape} y is {y}')
                elif otm.is_survival_curve():
                    intervals = otm.shape[-1] // 2
                    days_per_bin = 1 + otm.days_window // intervals
                    predicted_survivals = np.cumprod(y[:, :intervals], axis=1)
                    #predicted_survivals = np.cumprod(y[:, :10], axis=1) 2 year probability
                    csv_row.append(str(1 - predicted_survivals[0, -1]))
                    sick = np.sum(output_data[otm.output_name()][:, intervals:], axis=-1)
                    follow_up = np.cumsum(output_data[otm.output_name()][:, :intervals], axis=-1)[:, -1] * days_per_bin
                    csv_row.extend([str(sick[0]), str(follow_up[0])])
                elif otm.axes() > 1:
                    hd5_path = os.path.join(args.output_folder, args.id, 'inferred_hd5s', f'{sample_id}{TENSOR_EXT}')
                    os.makedirs(os.path.dirname(hd5_path), exist_ok=True)
                    with h5py.File(hd5_path, 'a') as hd5:
                        hd5.create_dataset(f'{otm.name}_truth', data=otm.rescale(output_data[otm.output_name()][0]),
                                           compression='gzip')
                        if otm.path_prefix == 'ukb_ecg_rest':
                            for lead in otm.channel_map:
                                hd5.create_dataset(f'/ukb_ecg_rest/{lead}/instance_0',
                                                   data=otm.rescale(y[0, otm.channel_map[lead]]),
                                                   compression='gzip')
            inference_writer.writerow(csv_row)
            tensor_paths_inferred.add(tensor_paths[0])
            stats['count'] += 1
            if stats['count'] % 250 == 0:
                logging.info(f"Wrote:{stats['count']} rows of inference.  Last tensor:{tensor_paths[0]}")


def _tensor_paths_from_sample_csv(tensors, sample_csv):
    sample_set = None
    if sample_csv is not None:
        with open(sample_csv, 'r') as csv_file:
            sample_ids = [row[0] for row in csv.reader(csv_file)]
            sample_set = set(sample_ids[1:])
    tensor_paths = [
        file for file in glob.glob(os.path.join(tensors, f"*{TENSOR_EXT}"))
        if sample_set is None or os.path.splitext(os.path.basename(file))[0] in sample_set
    ]
    return tensor_paths


def _hidden_file_name(output_folder: str, prefix_: str, id_: str, extension_: str) -> str:
    return os.path.join(output_folder, id_, f'hidden_{prefix_}_{id_}{extension_}')


def infer_hidden_layer_multimodal_multitask(args):
    stats = Counter()
    args.num_workers = 0
    inference_tsv = _hidden_file_name(args.output_folder, args.hidden_layer, args.id, '.tsv')
    tsv_style_is_genetics = 'genetics' in args.tsv_style
    tensor_paths = _tensor_paths_from_sample_csv(args.tensors, args.sample_csv)
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    generate_test = TensorGenerator(
        1, args.tensor_maps_in, args.tensor_maps_out, tensor_paths, num_workers=0,
        cache_size=args.cache_size, keep_paths=True, mixup_alpha=args.mixup_alpha,
    )
    generate_test.set_worker_paths(tensor_paths)
    full_model = legacy_multimodal_multitask_model(**args.__dict__)
    embed_model = make_hidden_layer_model(full_model, args.tensor_maps_in, args.hidden_layer)
    embed_model.save(_hidden_file_name(args.output_folder, f'{args.hidden_layer}_encoder_', args.id, '.h5'))
    dummy_input = {tm.input_name(): np.zeros((1,) + full_model.get_layer(tm.input_name()).input_shape[0][1:]) for tm in args.tensor_maps_in}
    dummy_out = embed_model.predict(dummy_input)
    latent_dimensions = int(np.prod(dummy_out.shape[1:]))
    logging.info(f'Dummy output shape is: {dummy_out.shape} latent dimensions: {latent_dimensions} Will write inferences to: {inference_tsv}')
    with open(inference_tsv, mode='w') as inference_file:
        inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['FID', 'IID'] if tsv_style_is_genetics else ['sample_id']
        header += [f'latent_{i}' for i in range(latent_dimensions)]
        inference_writer.writerow(header)

        while True:
            batch = next(generate_test)
            input_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_PATHS_INDEX]
            if tensor_paths[0] in stats:
                next(generate_test)  # this prints end of epoch info
                logging.info(f"Latent space inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                break

            sample_id = os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')
            prediction = embed_model.predict(input_data, verbose=0)
            prediction = np.reshape(prediction, (latent_dimensions,))
            csv_row = [sample_id, sample_id] if tsv_style_is_genetics else [sample_id]
            csv_row += [f'{prediction[i]}' for i in range(latent_dimensions)]
            inference_writer.writerow(csv_row)
            stats[tensor_paths[0]] += 1
            stats['count'] += 1
            if stats['count'] % 500 == 0:
                logging.info(f"Wrote:{stats['count']} rows of latent space inference.  Last tensor:{tensor_paths[0]}")


def train_shallow_model(args):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = make_shallow_model(args.tensor_maps_in, args.tensor_maps_out, args.learning_rate, args.model_file, args.model_layers)
    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size,
        args.epochs, args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels,
    )

    p = os.path.join(args.output_folder, args.id + '/')
    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    return _predict_and_evaluate(
        model, test_data, test_labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, p, test_paths, args.embed_visualization, args.alpha,
        args.dpi, args.plot_width, args.plot_height,
    )


def train_char_model(args):
    args.num_workers = 0
    logging.info(f'Number of workers forced to 0 for character emitting LSTM model.')
    base_model = legacy_multimodal_multitask_model(**args.__dict__)
    model, char_model = make_character_model_plus(
        args.tensor_maps_in, args.tensor_maps_out, args.learning_rate, base_model, args.language_layer,
        args.language_prefix, args.model_layers,
    )
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)

    model = train_model_from_generators(
        model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size,
        args.epochs, args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels,
    )
    batch = next(generate_test)
    input_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_PATHS_INDEX]
    sample_from_char_embed_model(args.tensor_maps_in, char_model, input_data, tensor_paths)

    out_path = os.path.join(args.output_folder, args.id + '/')
    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    return _predict_and_evaluate(
        model, data, labels, args.tensor_maps_in, args.tensor_maps_out, args.tensor_maps_protected,
        args.batch_size, args.hidden_layer, out_path, paths, args.embed_visualization, args.alpha,
        args.dpi, args.plot_width, args.plot_height,
    )


def train_siamese_model(args):
    base_model = legacy_multimodal_multitask_model(**args.__dict__)
    siamese_model = make_siamese_model(base_model, **args.__dict__)
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__, siamese=True)
    siamese_model = train_model_from_generators(
        siamese_model, generate_train, generate_valid, args.training_steps, args.validation_steps, args.batch_size,
        args.epochs, args.patience, args.output_folder, args.id, args.inspect_model, args.inspect_show_labels,
    )

    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    prediction = siamese_model.predict(data)
    return subplot_roc_per_class(
        prediction, labels['output_siamese'], {'random_siamese_verification_task': 0},
        args.protected_maps, args.id, os.path.join(args.output_folder, args.id + '/'),
    )


def infer_encoders_block_multimodal_multitask(args):
    args.num_workers = 0
    tsv_style_is_genetics = 'genetics' in args.tsv_style
    sample_set = None
    if args.sample_csv is not None:
        with open(args.sample_csv, 'r') as csv_file:
            sample_ids = [row[0] for row in csv.reader(csv_file)]
            sample_set = set(sample_ids[1:])
    _, encoders, _, _ = make_multimodal_multitask_model(**args.__dict__)
    latent_dimensions = args.dense_layers[-1]
    for e in encoders:
        stats = Counter()
        inference_tsv = _hidden_file_name(args.output_folder, e.name, args.id, '.tsv')
        logging.info(f'Will write encodings from {e.name} to: {inference_tsv}')
        # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
        tensor_paths = [
            os.path.join(args.tensors, tp) for tp in sorted(os.listdir(args.tensors))
            if os.path.splitext(tp)[-1].lower() == TENSOR_EXT and (sample_set is None or os.path.splitext(tp)[0] in sample_set)
        ]
        generate_test = TensorGenerator(
            1, [e], [], tensor_paths, num_workers=0,
            cache_size=args.cache_size, keep_paths=True, mixup_alpha=args.mixup_alpha,
        )
        generate_test.set_worker_paths(tensor_paths)
        with open(inference_tsv, mode='w') as inference_file:
            inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = ['FID', 'IID'] if tsv_style_is_genetics else ['sample_id']
            header += [f'latent_{i}' for i in range(latent_dimensions)]
            inference_writer.writerow(header)

            while True:
                batch = next(generate_test)
                input_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_PATHS_INDEX]
                if tensor_paths[0] in stats:
                    next(generate_test)  # this prints end of epoch info
                    logging.info(f"Latent space inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                    del stats
                    break

                sample_id = os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')
                prediction = encoders[e].predict(input_data, verbose=0)
                prediction = np.reshape(prediction, (latent_dimensions,))
                csv_row = [sample_id, sample_id] if tsv_style_is_genetics else [sample_id]
                csv_row += [f'{prediction[i]}' for i in range(latent_dimensions)]
                inference_writer.writerow(csv_row)
                stats[tensor_paths[0]] += 1
                stats['count'] += 1
                if stats['count'] % 500 == 0:
                    logging.info(f"Wrote:{stats['count']} rows of latent space inference.  Last tensor:{tensor_paths[0]}")


def pca_on_hidden_inference(args):
    latent_cols = [f'latent_{i}' for i in range(args.dense_layers[0])]
    pca_on_tsv(args.app_csv, latent_cols, 'sample_id', args.dense_layers[1])


def plot_predictions(args):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model, _, _, _ = make_multimodal_multitask_model(**args.__dict__)
    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    predictions = model.predict(data, batch_size=args.batch_size)
    if len(args.tensor_maps_out) == 1:
        predictions = [predictions]
    folder = os.path.join(args.output_folder, args.id, 'prediction_pngs/')
    predictions_to_pngs(predictions, args.tensor_maps_in, args.tensor_maps_out, data, labels, paths, folder)


def plot_while_training(args):
    generate_train, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    model = legacy_multimodal_multitask_model(**args.__dict__)

    plot_folder = os.path.join(args.output_folder, args.id, 'training_frames/')
    plot_while_learning(
        model, args.tensor_maps_in, args.tensor_maps_out, generate_train, test_data, test_labels, test_paths, args.epochs,
        args.batch_size, args.training_steps, plot_folder, args.write_pngs,
    )


def saliency_maps(args):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = legacy_multimodal_multitask_model(**args.__dict__)
    data, labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
    in_tensor = data[args.tensor_maps_in[0].input_name()]
    for tm in args.tensor_maps_out:
        if len(tm.shape) > 1:
            continue
        for channel in tm.channel_map:
            gradients = saliency_map(in_tensor, model, tm.output_name(), tm.channel_map[channel])
            plot_saliency_maps(in_tensor, gradients, paths, os.path.join(args.output_folder, f'{args.id}/saliency_maps/{tm.name}_{channel}'))


def _predict_and_evaluate(
    model, test_data, test_labels, tensor_maps_in, tensor_maps_out, tensor_maps_protected,
    batch_size, hidden_layer, plot_path, test_paths, embed_visualization, alpha, dpi, width, height,
):
    layer_names = [layer.name for layer in model.layers]
    performance_metrics = {}
    scatters = []
    rocs = []

    y_predictions = model.predict(test_data, batch_size=batch_size, verbose=0)
    protected_data = {tm: test_labels[tm.output_name()] for tm in tensor_maps_protected}
    for y, tm in zip(y_predictions, tensor_maps_out):
        if tm.output_name() not in layer_names:
            continue
        if not isinstance(y_predictions, list):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
            y = y_predictions
        y_truth = np.array(test_labels[tm.output_name()])
        performance_metrics.update(
            evaluate_predictions(
                tm, y, y_truth, protected_data, tm.name, plot_path, test_paths,
                rocs=rocs, scatters=scatters, dpi=dpi, width=width, height=height,
            ),
        )
        if tm.is_language():
            sample_from_language_model(tensor_maps_in[0], tm, model, test_data, max_samples=16)

    if len(rocs) > 1:
        subplot_rocs(rocs, plot_path)
    if len(scatters) > 1:
        subplot_scatters(scatters, plot_path)

    test_labels_1d = {tm: np.array(test_labels[tm.output_name()]) for tm in tensor_maps_out if tm.output_name() in test_labels}
    test_labels_1d.update(protected_data)
    if embed_visualization == "tsne":
        _tsne_wrapper(
            model, hidden_layer, alpha, plot_path, test_paths, test_labels_1d, test_data=test_data,
            tensor_maps_in=tensor_maps_in, batch_size=batch_size, dpi=dpi, width=width, height=height,
        )

    return performance_metrics


def _predict_scalars_and_evaluate_from_generator(
    model, generate_test, tensor_maps_in, tensor_maps_out, tensor_maps_protected,
    steps, hidden_layer, plot_path, alpha, dpi, width, height,
):
    layer_names = [layer.name for layer in model.layers]
    model_predictions = [tm.output_name() for tm in tensor_maps_out if tm.output_name() in layer_names]
    scalar_predictions = {tm.output_name(): [] for tm in tensor_maps_out if len(tm.shape) == 1 and tm.output_name() in layer_names}
    test_labels = {tm.output_name(): [] for tm in tensor_maps_out if len(tm.shape) == 1}
    protected_data = {tm: [] for tm in tensor_maps_protected}

    logging.info(f"Scalar predictions {model_predictions} names: {scalar_predictions.keys()} test labels: {test_labels.keys()}")
    embeddings = []
    test_paths = []
    for i in range(steps):
        batch = next(generate_test)
        input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
        y_predictions = model.predict(input_data, verbose=0)
        test_paths.extend(tensor_paths)
        if hidden_layer in layer_names:
            x_embed = embed_model_predict(model, tensor_maps_in, hidden_layer, input_data, 2)
            embeddings.extend(np.copy(np.reshape(x_embed, (x_embed.shape[0], np.prod(x_embed.shape[1:])))))

        for tm_output_name in test_labels:
            test_labels[tm_output_name].extend(np.copy(output_data[tm_output_name]))
        for tm in tensor_maps_protected:
            protected_data[tm].extend(np.copy(output_data[tm.output_name()]))

        for y, tm_output_name in zip(y_predictions, model_predictions):
            if not isinstance(y_predictions, list):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
                y = y_predictions
            if tm_output_name in scalar_predictions:
                scalar_predictions[tm_output_name].extend(np.copy(y))

        if i % 100 == 0:
            logging.info(f'Processed {i} batches, {len(test_paths)} tensors.')

    performance_metrics = {}
    scatters = []
    rocs = []
    for tm in tensor_maps_protected:
        protected_data[tm] = np.array(protected_data[tm])

    for tm in tensor_maps_out:
        if tm.output_name() in scalar_predictions:
            y_predict = np.array(scalar_predictions[tm.output_name()])
            y_truth = np.array(test_labels[tm.output_name()])
            metrics = evaluate_predictions(
                tm, y_predict, y_truth, protected_data, tm.name, plot_path, test_paths,
                rocs=rocs, scatters=scatters, dpi=dpi, width=width, height=height,
            )
            performance_metrics.update(metrics)

    if len(rocs) > 1:
        subplot_rocs(rocs, plot_path)
    if len(scatters) > 1:
        subplot_scatters(scatters, plot_path)
    if len(embeddings) > 0:
        test_labels_1d = {tm: np.array(test_labels[tm.output_name()]) for tm in tensor_maps_out if tm.output_name() in test_labels}
        _tsne_wrapper(
            model, hidden_layer, alpha, plot_path, test_paths, test_labels_1d,
            embeddings=embeddings, dpi=dpi, width=width, height=height,
        )

    return performance_metrics


def _get_common_outputs(models_inputs_outputs, output_prefix):
    """Returns a set of outputs common to all the models so we can compare the models according to those outputs only"""
    all_outputs = []
    for (_, ios) in models_inputs_outputs.items():
        outputs = {k: v for (k, v) in ios.items() if k == output_prefix}
        for (_, output) in outputs.items():
            all_outputs.append(set(output))
    logging.info(f'Finding model output intersection of {all_outputs}')
    return reduce(set.intersection, all_outputs)


def _get_predictions(args, models_inputs_outputs, input_data, outputs, input_prefix, output_prefix):
    """Makes multi-modal predictions for a given number of models.

    Returns:
        dict: The nested dictionary of predicted values.

            {
                'tensor_map_1':
                    {
                        'model_1': [[value1, value2], [...]],
                        'model_2': [[value3, value4], [...]]
                    },
                'tensor_map_2':
                    {
                        'model_2': [[value5, value6], [...]],
                        'model_3': [[value7, value8], [...]]
                    }
            }
    """
    predictions = defaultdict(dict)
    for model_file in models_inputs_outputs.keys():
        args.tensor_maps_out = models_inputs_outputs[model_file][output_prefix]
        args.tensor_maps_in = models_inputs_outputs[model_file][input_prefix]
        args.model_file = model_file
        model = legacy_multimodal_multitask_model(**args.__dict__)
        model_name = os.path.basename(model_file).replace(MODEL_EXT, '_')

        # We can feed 'model.predict()' the entire input data because it knows what subset to use
        y_pred = model.predict(input_data, batch_size=args.batch_size)
        for i, tm in enumerate(args.tensor_maps_out):
            if tm in outputs:
                if len(args.tensor_maps_out) == 1:
                    predictions[tm][model_name] = y_pred
                else:
                    predictions[tm][model_name] = y_pred[i]

    return predictions


def _scalar_predictions_from_generator(args, models_inputs_outputs, generator, steps, outputs, input_prefix, output_prefix):
    """Makes multi-modal scalar predictions for a given number of models.

    Returns:
        dict: The nested dictionary of predicted values.

            {
                'tensor_map_1':
                    {
                        'model_1': [[value1, value2], [...]],
                        'model_2': [[value3, value4], [...]]
                    },
                'tensor_map_2':
                    {
                        'model_2': [[value5, value6], [...]],
                        'model_3': [[value7, value8], [...]]
                    }
            }
    """
    models = {}
    test_paths = []
    scalar_predictions = {}
    test_labels = {tm.output_name(): [] for tm in args.tensor_maps_out if len(tm.shape) == 1}

    for model_file in models_inputs_outputs:
        args.model_file = model_file
        args.tensor_maps_in = models_inputs_outputs[model_file][input_prefix]
        args.tensor_maps_out = models_inputs_outputs[model_file][output_prefix]
        model = legacy_multimodal_multitask_model(**args.__dict__)
        model_name = os.path.basename(model_file).replace(MODEL_EXT, '')
        models[model_name] = model
        scalar_predictions[model_name] = [tm for tm in models_inputs_outputs[model_file][output_prefix] if len(tm.shape) == 1]

    predictions = defaultdict(dict)
    for j in range(steps):
        batch = next(generator)
        input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
        test_paths.extend(tensor_paths)
        for tl in test_labels:
            test_labels[tl].extend(np.copy(output_data[tl]))

        for model_name, model_file in zip(models, models_inputs_outputs):
            # We can feed 'model.predict()' the entire input data because it knows what subset to use
            y_predictions = models[model_name].predict(input_data)

            for y, tm in zip(y_predictions, models_inputs_outputs[model_file][output_prefix]):
                if not isinstance(y_predictions, list):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
                    y = y_predictions
                if j == 0 and tm in scalar_predictions[model_name]:
                    predictions[tm][model_name] = []
                if tm in scalar_predictions[model_name]:
                    predictions[tm][model_name].extend(np.copy(y))

    for tm in predictions:
        logging.info(f"{tm.output_name()} labels: {len(test_labels[tm.output_name()])}")
        test_labels[tm.output_name()] = np.array(test_labels[tm.output_name()])
        for m in predictions[tm]:
            logging.info(f"{tm.output_name()} model: {m} prediction length:{len(predictions[tm][m])}")
            predictions[tm][m] = np.array(predictions[tm][m])

    return predictions, test_labels, test_paths


def _calculate_and_plot_prediction_stats(args, predictions, outputs, paths):
    rocs = []
    scatters = []
    for tm in args.tensor_maps_out:
        if tm not in predictions:
            continue
        plot_title = tm.name+'_'+args.id
        plot_folder = os.path.join(args.output_folder, args.id)
        if tm.is_categorical() and tm.axes() == 1:
            for m in predictions[tm]:
                logging.info(f"{tm.name} channel map {tm.channel_map}\nsum truth = {np.sum(outputs[tm.output_name()], axis=0)}\nsum preds = {np.sum(predictions[tm][m], axis=0)}")
            plot_rocs(
                predictions[tm], outputs[tm.output_name()], tm.channel_map, plot_title, plot_folder,
                dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            rocs.append((predictions[tm], outputs[tm.output_name()], tm.channel_map))
        elif tm.is_categorical() and tm.axes() == 4:
            for p in predictions[tm]:
                y = predictions[tm][p]
                melt_shape = (y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3], y.shape[4])
                predictions[tm][p] = y.reshape(melt_shape)

            y_truth = outputs[tm.output_name()].reshape(melt_shape)
            plot_rocs(
                predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder,
                dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            plot_precision_recalls(
                predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder,
                dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            roc_aucs = get_roc_aucs(predictions[tm], y_truth, tm.channel_map)
            precision_recall_aucs = get_precision_recall_aucs(predictions[tm], y_truth, tm.channel_map)
            aucs = {"ROC": roc_aucs, "Precision-Recall": precision_recall_aucs}
            log_aucs(**aucs)
        elif tm.is_categorical() and tm.axes() == 3:
            # have to plot dice before the reshape
            plot_dice(
                predictions[tm], outputs[tm.output_name()], tm.channel_map, paths, plot_title, plot_folder,
                dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            for p in predictions[tm]:
                y = predictions[tm][p]
                melt_shape = (y.shape[0]*y.shape[1]*y.shape[2], y.shape[3])
                predictions[tm][p] = y.reshape(melt_shape)
            y_truth = outputs[tm.output_name()].reshape(melt_shape)
            plot_rocs(
                predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder,
                dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            plot_precision_recalls(
                predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder,
                dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            roc_aucs = get_roc_aucs(predictions[tm], y_truth, tm.channel_map)
            precision_recall_aucs = get_precision_recall_aucs(predictions[tm], y_truth, tm.channel_map)
            aucs = {"ROC": roc_aucs, "Precision-Recall": precision_recall_aucs}
            log_aucs(**aucs)
        elif tm.is_continuous() and tm.axes() == 1:
            scaled_predictions = {k: tm.rescale(predictions[tm][k]) for k in predictions[tm]}
            plot_scatters(
                scaled_predictions, tm.rescale(outputs[tm.output_name()]), plot_title, plot_folder,
                dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            scatters.append((scaled_predictions, tm.rescale(outputs[tm.output_name()]), plot_title, None))
            coefs = get_pearson_coefficients(scaled_predictions, tm.rescale(outputs[tm.output_name()]))
            log_pearson_coefficients(coefs, tm.name)
        elif tm.is_time_to_event():
            new_predictions = {}
            for m in predictions[tm]:
                c_index = concordance_index_censored(outputs[tm.output_name()][:, 0] == 1.0, outputs[tm.output_name()][:, 1], predictions[tm][m][:, 0])
                concordance_return_values = ['C-Index', 'Concordant Pairs', 'Discordant Pairs', 'Tied Predicted Risk', 'Tied Event Time']
                logging.info(f"Model: {m} {[f'{label}: {value}' for label, value in zip(concordance_return_values, c_index)]}")
                new_predictions[f'{m}_C_Index_{c_index[0]:0.3f}'] = predictions[tm][m]
            plot_rocs(
                new_predictions, outputs[tm.output_name()][:, 0, np.newaxis], {f'_vs_ROC': 0}, plot_title,
                plot_folder, dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            rocs.append((new_predictions, outputs[tm.output_name()][:, 0, np.newaxis], {f'_vs_ROC': 0}))
            plot_prediction_calibrations(
                new_predictions, outputs[tm.output_name()][:, 0, np.newaxis], {f'_vs_ROC': 0},
                plot_title, plot_folder, dpi=args.dpi,
                width=args.plot_width, height=args.plot_height,
            )
        elif tm.is_survival_curve():
            for m in predictions[tm]:
                plot_survival(
                    predictions[tm][m], outputs[tm.output_name()], f'{m}_{plot_title}', tm.days_window,
                    prefix=plot_folder, dpi=args.dpi, width=args.plot_width, height=args.plot_height,
                )
        else:
            scaled_predictions = {k: tm.rescale(predictions[tm][k]) for k in predictions[tm]}
            plot_scatters(
                scaled_predictions, tm.rescale(outputs[tm.output_name()]), plot_title, plot_folder,
                dpi=args.dpi, width=args.plot_width, height=args.plot_height,
            )
            coefs = get_pearson_coefficients(scaled_predictions, tm.rescale(outputs[tm.output_name()]))
            log_pearson_coefficients(coefs, tm.name)

    if len(rocs) > 1:
        subplot_comparison_rocs(rocs, plot_folder, dpi=args.dpi, width=args.plot_width, height=args.plot_height)
    if len(scatters) > 1:
        subplot_comparison_scatters(scatters, plot_folder, dpi=args.dpi, width=args.plot_width, height=args.plot_height)


def _tsne_wrapper(
    model, hidden_layer_name, alpha, plot_path, test_paths, test_labels, test_data=None,
    tensor_maps_in=None, batch_size=16, embeddings=None, dpi=300, width=7, height=7,
):
    """Plot 2D t-SNE of a model's hidden layer colored by many different co-variates.

    Callers must provide either model's embeddings or test_data on which embeddings will be inferred

    :param model: Keras model
    :param hidden_layer_name: String name of the hidden layer whose embeddings will be visualized
    :param alpha: Transparency of each data point
    :param plot_path: Image file name and path for the t_SNE plot
    :param test_paths: Paths for hd5 file containing each sample
    :param test_labels: Dictionary mapping TensorMaps to numpy arrays of labels (co-variates) to color code the t-SNE plots with
    :param test_data: Input data for the model necessary if embeddings is None
    :param tensor_maps_in: Input TensorMaps of the model necessary if embeddings is None
    :param batch_size: Batch size necessary if embeddings is None
    :param embeddings: (optional) Model's embeddings
    :return: None
    """
    if hidden_layer_name not in [layer.name for layer in model.layers]:
        logging.warning(f"Can't compute t-SNE, layer:{hidden_layer_name} not in provided model.")
        return

    if embeddings is None:
        embeddings = embed_model_predict(model, tensor_maps_in, hidden_layer_name, test_data, batch_size)

    gene_labels = []
    label_dict, categorical_labels, continuous_labels = test_labels_to_label_map(test_labels, len(test_paths))
    if len(categorical_labels) > 0 or len(continuous_labels) > 0 or len(gene_labels) > 0:
        plot_tsne(
            embeddings, categorical_labels, continuous_labels, gene_labels,
            label_dict, plot_path, alpha, dpi, width, height,
        )


if __name__ == '__main__':
    arguments = parse_args()
    run(arguments)  # back to the top
