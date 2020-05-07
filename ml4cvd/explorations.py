# explorations.py

# Imports
import os
import csv
import math
import logging
import operator
import datetime
from operator import itemgetter
from functools import reduce
from itertools import combinations
from collections import defaultdict, Counter, OrderedDict
from typing import Dict, List, Tuple, Generator, Optional, DefaultDict

import h5py
import numpy as np
import pandas as pd
from multiprocess import Pool, Value
from tensorflow.keras.models import Model

import matplotlib
matplotlib.use('Agg')  # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt  # First import matplotlib, then use Agg, then import plt

from ml4cvd.models import make_multimodal_multitask_model
from ml4cvd.TensorMap import TensorMap, Interpretation, decompress_data
from ml4cvd.tensor_generators import TensorGenerator, test_train_valid_tensor_generators
from ml4cvd.tensor_generators import BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX
from ml4cvd.plots import evaluate_predictions, subplot_rocs, subplot_scatters
from ml4cvd.plots import plot_histograms_in_pdf, plot_heatmap, plot_cross_reference
from ml4cvd.defines import JOIN_CHAR, MRI_SEGMENTED_CHANNEL_MAP, CODING_VALUES_MISSING, CODING_VALUES_LESS_THAN_ONE
from ml4cvd.defines import TENSOR_EXT, IMAGE_EXT, ECG_CHAR_2_IDX, ECG_IDX_2_CHAR, PARTNERS_CHAR_2_IDX, PARTNERS_IDX_2_CHAR, PARTNERS_READ_TEXT


CSV_EXT = '.tsv'


def sort_csv(tensors, tensor_maps_in):
    stats = Counter()
    for folder in sorted(os.listdir(tensors)):

        for name in sorted(os.listdir(os.path.join(tensors, folder))):
            try:
                with h5py.File(os.path.join(tensors, folder, name), "r") as hd5:
                    for tm in tensor_maps_in:
                        tensor = tm.postprocess_tensor(tm.tensor_from_file(tm, hd5, {}), augment=False, hd5=hd5)

                        if tm.name == 'lead_v6_zeros' and tensor[0] > 1874:
                            stats[f'Total_{tm.name}_zero_padded'] += 1
                            stats[f'{folder}_{tm.name}_zero_padded'] += 1
                        elif tm.name == 'lead_i_zeros' and tensor[0] > 1249:
                            stats[f'Total_{tm.name}_zero_padded'] += 1
                            stats[f'{folder}_{tm.name}_zero_padded'] += 1
                        elif tm.name not in ['lead_i_zeros', 'lead_v6_zeros']:
                            stats[f'{folder}_{tm.name}_{tensor[0]}'] += 1
                            stats[f'Total_{tm.name}_{tensor[0]}'] += 1
            except (IndexError, KeyError, ValueError, OSError, RuntimeError) as e:
                pass
                #logging.info(f'Got error at {name} error:\n {e} {traceback.format_exc()}')

        logging.info(f'In folder {folder} with {len(os.listdir(os.path.join(tensors, folder)))} ECGs')
        if len(os.listdir(os.path.join(tensors, folder))) > 0:
            logging.info(f'I Zero padded has:{stats[f"{folder}_lead_i_zeros_zero_padded"]}, {100 * stats[f"{folder}_lead_i_zeros_zero_padded"] / len(os.listdir(os.path.join(tensors, folder))):.1f}%')
            logging.info(f'V6 Zero padded has:{stats[f"{folder}_lead_v6_zeros_zero_padded"]}, {100*stats[f"{folder}_lead_v6_zeros_zero_padded"]/len(os.listdir(os.path.join(tensors, folder))):.1f}%')
    for k, v in sorted(stats.items(), key=lambda x: x[0]):
        logging.info(f'{k} has {v}')


def predictions_to_pngs(
    predictions: np.ndarray, tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap], data: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray], paths: List[str], folder: str,
) -> None:
    # TODO Remove this command line order dependency
    input_map = tensor_maps_in[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    for y, tm in zip(predictions, tensor_maps_out):
        if not isinstance(predictions, list):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
            y = predictions
        for im in tensor_maps_in:
            if tm.is_categorical() and im.dependent_map == tm:
                input_map = im
            elif len(tm.shape) == len(im.shape):
                input_map = im
        logging.info(f"Write predictions as PNGs y:{y.shape} labels:{labels[tm.output_name()].shape} folder:{folder}")
        if tm.is_mesh():
            vmin = np.min(data[input_map.input_name()])
            vmax = np.max(data[input_map.input_name()])
            for i in range(y.shape[0]):
                sample_id = os.path.basename(paths[i]).replace(TENSOR_EXT, '')
                if input_map.axes() == 4 and input_map.shape[-1] == 1:
                    sample_data = data[input_map.input_name()][i, ..., 0]
                    cols = max(2, int(math.ceil(math.sqrt(sample_data.shape[-1]))))
                    rows = max(2, int(math.ceil(sample_data.shape[-1] / cols)))
                    path_prefix = f'{folder}{sample_id}_bbox_batch_{i:02d}{IMAGE_EXT}'
                    logging.info(f"sample_data shape: {sample_data.shape} cols {cols}, {rows} Predicted BBox: {y[i]}, True BBox: {labels[tm.output_name()][i]} Vmin {vmin} Vmax{vmax}")
                    _plot_3d_tensor_slices_as_gray(sample_data, path_prefix, cols, rows, bboxes=[labels[tm.output_name()][i], y[i]])
                else:
                    fig, ax = plt.subplots(1)
                    if input_map.axes() == 3 and input_map.shape[-1] == 1:
                        ax.imshow(data[input_map.input_name()][i, :, :, 0], cmap='gray', vmin=vmin, vmax=vmax)
                    elif input_map.axes() == 2:
                        ax.imshow(data[input_map.input_name()][i, :, :], cmap='gray', vmin=vmin, vmax=vmax)
                    corner, width, height = _2d_bbox_to_corner_and_size(labels[tm.output_name()][i])
                    ax.add_patch(matplotlib.patches.Rectangle(corner, width, height, linewidth=1, edgecolor='g', facecolor='none'))
                    y_corner, y_width, y_height = _2d_bbox_to_corner_and_size(y[i])
                    ax.add_patch(matplotlib.patches.Rectangle(y_corner, y_width, y_height, linewidth=1, edgecolor='y', facecolor='none'))
                    logging.info(f"True BBox: {corner}, {width}, {height} Predicted BBox: {y_corner}, {y_width}, {y_height} Vmin {vmin} Vmax{vmax}")
                plt.savefig(f"{folder}{sample_id}_bbox_batch_{i:02d}{IMAGE_EXT}")
        elif len(tm.shape) in [1, 2]:
            vmin = np.min(data[input_map.input_name()])
            vmax = np.max(data[input_map.input_name()])
            for i in range(y.shape[0]):
                sample_id = os.path.basename(paths[i]).replace(TENSOR_EXT, '')
                if input_map.axes() == 4 and input_map.shape[-1] == 1:
                    sample_data = data[input_map.input_name()][i, ..., 0]
                    cols = max(2, int(math.ceil(math.sqrt(sample_data.shape[-1]))))
                    rows = max(2, int(math.ceil(sample_data.shape[-1] / cols)))
                    path_prefix = f'{folder}{sample_id}_bbox_batch_{i:02d}{IMAGE_EXT}'
                    logging.info(f"sample_data shape: {sample_data.shape} cols {cols}, {rows} Predicted BBox: {y[i]}, True BBox: {labels[tm.output_name()][i]} Vmin {vmin} Vmax{vmax}")
                    _plot_3d_tensor_slices_as_gray(sample_data, path_prefix, cols, rows, bboxes=[labels[tm.output_name()][i], y[i]])
                else:
                    fig, ax = plt.subplots(1)
                    if input_map.axes() == 3 and input_map.shape[-1] == 1:
                        ax.imshow(data[input_map.input_name()][i, :, :, 0], cmap='gray', vmin=vmin, vmax=vmax)
                    elif input_map.axes() == 2:
                        ax.imshow(data[input_map.input_name()][i, :, :], cmap='gray', vmin=vmin, vmax=vmax)
                    corner, width, height = _2d_bbox_to_corner_and_size(labels[tm.output_name()][i])
                    ax.add_patch(matplotlib.patches.Rectangle(corner, width, height, linewidth=1, edgecolor='g', facecolor='none'))
                    y_corner, y_width, y_height = _2d_bbox_to_corner_and_size(y[i])
                    ax.add_patch(matplotlib.patches.Rectangle(y_corner, y_width, y_height, linewidth=1, edgecolor='y', facecolor='none'))
                    logging.info(f"True BBox: {corner}, {width}, {height} Predicted BBox: {y_corner}, {y_width}, {y_height} Vmin {vmin} Vmax{vmax}")
                plt.savefig(f"{folder}{sample_id}_bbox_batch_{i:02d}{IMAGE_EXT}")
        elif len(tm.shape) == 3:
            for i in range(y.shape[0]):
                sample_id = os.path.basename(paths[i]).replace(TENSOR_EXT, '')
                if tm.is_categorical():
                    plt.imsave(f"{folder}{sample_id}_truth_{i:02d}{IMAGE_EXT}", np.argmax(labels[tm.output_name()][i], axis=-1, cmap='gray'))
                    plt.imsave(f"{folder}{sample_id}_prediction_{i:02d}{IMAGE_EXT}", np.argmax(y[i], axis=-1), cmap='gray')
                    if input_map is not None:
                        plt.imsave(f"{folder}{sample_id}_mri_slice_{i:02d}{IMAGE_EXT}", data[input_map.input_name()][i, :, :, 0], cmap='gray')
                else:
                    for j in range(y.shape[3]):
                        plt.imsave(f"{folder}{sample_id}_truth_{i:02d}_{j:02d}{IMAGE_EXT}", labels[tm.output_name()][i, :, :, j], cmap='gray')
                        plt.imsave(f"{folder}{sample_id}_prediction_{i:02d}_{j:02d}{IMAGE_EXT}", y[i, :, :, j], cmap='gray')
                        plt.imsave(f"{folder}{sample_id}_mri_slice_{i:02d}_{j:02d}{IMAGE_EXT}", data[input_map.input_name()][i, :, :, j], cmap='gray')
        elif len(tm.shape) == 4:
            for im in tensor_maps_in:
                if im.dependent_map == tm:
                    break
            for i in range(y.shape[0]):
                sample_id = os.path.basename(paths[i]).replace(TENSOR_EXT, '')
                for j in range(y.shape[3]):
                    if tm.is_categorical():
                        truth = np.argmax(labels[tm.output_name()][i, :, :, j, :], axis=-1)
                        prediction = np.argmax(y[i, :, :, j, :], axis=-1)
                        true_donut = np.ma.masked_where(truth == 2, data[im.input_name()][i, :, :, j, 0])
                        predict_donut = np.ma.masked_where(prediction == 2, data[im.input_name()][i, :, :, j, 0])
                        plt.imsave(folder+sample_id+'_truth_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, truth, cmap='gray')
                        plt.imsave(folder+sample_id+'_prediction_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, prediction, cmap='gray')
                        plt.imsave(folder+sample_id+'_mri_slice_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, data[im.input_name()][i, :, :, j, 0], cmap='gray')
                        plt.imsave(folder+sample_id + '_true_donut_{0:03d}_{1:03d}'.format(i, j) + IMAGE_EXT, true_donut, cmap='gray')
                        plt.imsave(folder + sample_id + '_predict_donut_{0:03d}_{1:03d}'.format(i, j) + IMAGE_EXT, predict_donut, cmap='gray')
                    else:
                        plt.imsave(folder+sample_id+'_truth_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, labels[tm.output_name()][i, :, :, j, 0], cmap='gray')
                        plt.imsave(folder+sample_id+'_prediction_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, y[i, :, :, j, 0], cmap='gray')


def plot_while_learning(
    model, tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
    generate_train: Generator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[List[str]]], None, None],
    test_data: Dict[str, np.ndarray], test_labels: Dict[str, np.ndarray], test_paths: List[str], epochs: int, batch_size: int,
    training_steps: int, folder: str, write_pngs: bool,
):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(epochs):
        rocs = []
        scatters = []
        predictions = model.predict(test_data, batch_size=batch_size)
        if len(tensor_maps_out) == 1:
            predictions = [predictions]
        for y, tm in zip(predictions, tensor_maps_out):
            for im in tensor_maps_in:
                if im.dependent_map == tm:
                    break
            if not write_pngs:
                mri_in = test_data[im.input_name()]
                vmin = np.min(mri_in)
                vmax = np.max(mri_in)
                logging.info(f"epoch:{i} write segmented mris y shape:{y.shape} label shape:{test_labels[tm.output_name()].shape} to folder:{folder}")
                if tm.is_categorical() and len(tm.shape) == 3:
                    for yi in range(y.shape[0]):
                        plt.imsave(f"{folder}batch_{yi}_truth_epoch_{i:03d}{IMAGE_EXT}", np.argmax(test_labels[tm.output_name()][yi], axis=-1), cmap='gray')
                        plt.imsave(f"{folder}batch_{yi}_prediction_epoch_{i:03d}{IMAGE_EXT}", np.argmax(y[yi], axis=-1), cmap='gray')
                        plt.imsave(f"{folder}batch_{yi}_mri_epoch_{i:03d}{IMAGE_EXT}", mri_in[yi, :, :, 0], cmap='gray', vmin=vmin, vmax=vmax)
                elif tm.is_categorical() and len(tm.shape) == 4:
                    for yi in range(y.shape[0]):
                        for j in range(y.shape[3]):
                            truth = np.argmax(test_labels[tm.output_name()][yi, :, :, j, :], axis=-1)
                            prediction = np.argmax(y[yi, :, :, j, :], axis=-1)
                            true_donut = np.ma.masked_where(truth == 2, mri_in[yi, :, :, j, 0])
                            predict_donut = np.ma.masked_where(prediction == 2, mri_in[yi, :, :, j, 0])
                            plt.imsave(f"{folder}batch_{yi}_slice_{j:03d}_prediction_epoch_{i:03d}{IMAGE_EXT}", prediction, cmap='gray')
                            plt.imsave(f"{folder}batch_{yi}_slice_{j:03d}_p_donut_epoch_{i:03d}{IMAGE_EXT}", predict_donut, cmap='gray', vmin=vmin, vmax=vmax)
                            if i == 0:
                                plt.imsave(f"{folder}batch_{yi}_slice_{j:03d}_truth_epoch_{i:03d}{IMAGE_EXT}", truth, cmap='gray')
                                plt.imsave(f"{folder}batch_{yi}_slice_{j:03d}_t_donut_epoch_{i:03d}{IMAGE_EXT}", true_donut, cmap='gray', vmin=vmin, vmax=vmax)
                                plt.imsave(f"{folder}batch_{yi}_slice_{j:03d}_mri_epoch_{i:03d}{IMAGE_EXT}", mri_in[yi, :, :, j, 0], cmap='gray', vmin=vmin, vmax=vmax)
                else:
                    logging.warning(f'Not writing PNGs')
            elif write_pngs:
                if len(tensor_maps_out) == 1:
                    y = predictions[0]
                evaluate_predictions(tm, y, test_labels[tm.output_name()], f"{tm.name}_epoch_{i:03d}", folder, test_paths, rocs=rocs, scatters=scatters)
        if len(rocs) > 1:
            subplot_rocs(rocs, folder+f"epoch_{i:03d}_")
        if len(scatters) > 1:
            subplot_scatters(scatters, folder+f"epoch_{i:03d}_")

        model.fit_generator(generate_train, steps_per_epoch=training_steps, epochs=1, verbose=1)


def plot_histograms_of_tensors_in_pdf(
    run_id: str,
    tensor_folder: str,
    output_folder: str,
    max_samples: int = None,
) -> None:
    """
    :param id: name for the plotting run
    :param tensor_folder: directory with tensor files to plot histograms from
    :param output_folder: folder containing the output plot
    :param max_samples: specifies how many tensor files to down-sample from; by default all tensors are used
    """
    stats, num_tensor_files = _collect_continuous_stats_from_tensor_files(tensor_folder, max_samples)
    logging.info(f"Collected continuous stats for {len(stats)} fields. Now plotting histograms of them...")
    plot_histograms_in_pdf(stats, num_tensor_files, run_id, output_folder)


def plot_heatmap_of_tensors(
    id: str,
    tensor_folder: str,
    output_folder: str,
    min_samples: int,
    max_samples: int = None,
) -> None:
    """
    :param id: name for the plotting run
    :param tensor_folder: directory with tensor files to plot histograms from
    :param output_folder: folder containing the output plot
    :param min_samples: calculate correlation coefficient only if both fields have values from that many common samples
    :param max_samples: specifies how many tensor files to down-sample from; by default all tensors are used
    """
    stats, _ = _collect_continuous_stats_from_tensor_files(tensor_folder, max_samples, ['0'], 0)
    logging.info(f"Collected continuous stats for {len(stats)} fields. Now plotting a heatmap of their correlations...")
    plot_heatmap(stats, id, min_samples, output_folder)


def tabulate_correlations_of_tensors(
    run_id: str,
    tensor_folder: str,
    output_folder: str,
    min_samples: int,
    max_samples: int = None,
) -> None:
    """
    :param id: name for the plotting run
    :param tensor_folder: directory with tensor files to plot histograms from
    :param output_folder: folder containing the output plot
    :param min_samples: calculate correlation coefficient only if both fields have values from that many common samples
    :param max_samples: specifies how many tensor files to down-sample from; by default all tensors are used
    """
    stats, _ = _collect_continuous_stats_from_tensor_files(tensor_folder, max_samples)
    logging.info(f"Collected continuous stats for {len(stats)} fields. Now tabulating their cross-correlations...")
    _tabulate_correlations(stats, run_id, min_samples, output_folder)


def mri_dates(tensors: str, output_folder: str, run_id: str):
    incident_dates = []
    prevalent_dates = []
    disease = 'hypertension'
    disease_date_key = disease + '_date'
    data_date_key = 'assessment-date_0_0'
    tensor_paths = [tensors + tp for tp in os.listdir(tensors) if os.path.splitext(tp)[-1].lower() == TENSOR_EXT]
    for tp in tensor_paths:
        try:
            with h5py.File(tp, 'r') as hd5:
                if data_date_key in hd5 and disease_date_key in hd5:
                    if int(hd5[disease][0]) == 1:
                        data_date = str2date(str(hd5[data_date_key][0]))
                        disease_date = str2date(str(hd5[disease_date_key][0]))
                        if data_date < disease_date:
                            incident_dates.append(disease_date)
                        else:
                            prevalent_dates.append(disease_date)
        except:
            logging.exception(f"Broken tensor at:{tp}")

    plt.figure(figsize=(12, 12))
    plt.xlabel(data_date_key)
    plt.hist(incident_dates, bins=60)
    plt.savefig(os.path.join(output_folder, run_id, disease+'_'+data_date_key+'_incident'+IMAGE_EXT))
    plt.figure(figsize=(12,12))
    plt.xlabel(data_date_key)
    plt.hist(prevalent_dates, bins=60)
    plt.savefig(os.path.join(output_folder, run_id, disease+'_'+data_date_key+'_prevalent'+IMAGE_EXT))


def ecg_dates(tensors: str, output_folder: str, run_id: str):
    incident_dates = []
    prevalent_dates = []
    tensor_paths = [tensors + tp for tp in os.listdir(tensors) if os.path.splitext(tp)[-1].lower()==TENSOR_EXT]
    for tp in tensor_paths:
        try:
            with h5py.File(tp, 'r') as hd5:
                if 'ecg_bike_date_0' in hd5 and 'coronary_artery_disease_soft_date' in hd5:
                    ecg_date = str2date(str(hd5['ecg_bike_date_0'][0]))
                    cad_date = str2date(str(hd5['coronary_artery_disease_soft_date'][0]))
                    if ecg_date < cad_date:
                        incident_dates.append(ecg_date)
                    else:
                        prevalent_dates.append(ecg_date)
        except:
            logging.exception(f"Broken tensor at:{tp}")

    plt.figure(figsize=(12, 12))
    plt.xlabel('ECG Acquisition Date')
    plt.hist(incident_dates, bins=60)
    plt.savefig(os.path.join(output_folder, run_id, 'ecg_dates_incident'+IMAGE_EXT))
    plt.figure(figsize=(12, 12))
    plt.xlabel('ECG Acquisition Date')
    plt.hist(prevalent_dates, bins=60)
    plt.savefig(os.path.join(output_folder, run_id, 'ecg_dates_prevalent'+IMAGE_EXT))


def str2date(d):
    parts = d.split('-')
    if len(parts) < 2:
        return datetime.datetime.now().date()
    return datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))


def sample_from_char_model(tensor_maps_in: List[TensorMap], char_model: Model, test_batch: Dict[str, np.ndarray], test_paths: List[str]) -> None:
    for tm in tensor_maps_in:
        if tm.interpretation == Interpretation.LANGUAGE:
            language_map = tm
            if PARTNERS_READ_TEXT in tm.name:
                index_map = PARTNERS_IDX_2_CHAR
                char_map = PARTNERS_CHAR_2_IDX
            else:
                index_map = ECG_IDX_2_CHAR
                char_map = ECG_CHAR_2_IDX
        elif tm.interpretation == Interpretation.EMBEDDING:
            embed_map = tm

    try:
        embed_map
    except NameError:
        raise ValueError(f'Sampling from a character level model requires an embedding tmap.')

    window_size = test_batch[language_map.input_name()].shape[1]
    alphabet_size = test_batch[language_map.input_name()].shape[2]
    for i in range(test_batch[embed_map.input_name()].shape[0]):
        count = 0
        sentence = ''
        next_char = ''
        embed_in = test_batch[embed_map.input_name()][i:i+1, :]
        burn_in = np.zeros((1, window_size, alphabet_size), dtype=np.float32)
        window_size = burn_in.shape[1]
        with h5py.File(test_paths[i], 'r') as hd5:
            logging.info(f"\n")
            if 'read_' in language_map.name:
                caption = decompress_data(data_compressed=hd5[tm.name][()], dtype=hd5[tm.name].attrs['dtype'])
            else:
                caption = str(tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess())[()]).strip()
            logging.info(f"Real text: {caption}")
        while next_char != '!' and count < 400:
            cur_test = {embed_map.input_name(): embed_in, language_map.input_name(): burn_in}
            y_pred = char_model.predict(cur_test)
            next_char = index_map[_sample_with_heat(y_pred[0, :], 0.7)]
            sentence += next_char
            burn_in = np.zeros((1,) + test_batch[language_map.input_name()].shape[1:], dtype=np.float32)
            for j, c in enumerate(reversed(sentence)):
                if j == window_size:
                    break
                burn_in[0, window_size-j-1, char_map[c]] = 1.0
            count += 1
        logging.info(f"Model text:{sentence}")


def tensors_to_label_dictionary(
    categorical_labels: List,
    continuous_labels: List,
    gene_labels: List,
    samples2genes: Dict[str, str],
    test_paths: List,
) -> Dict[str, np.ndarray]:
    label_dict = {k: np.zeros((len(test_paths))) for k in categorical_labels + continuous_labels + gene_labels}
    for i, tp in enumerate(test_paths):
        hd5 = h5py.File(tp, 'r')
        for k in categorical_labels:
            if k in hd5['categorical']:
                label_dict[k][i] = 1
            elif k in hd5 and hd5[k][0] == 1:
                label_dict[k][i] = 1
        for mk in continuous_labels:
            for k in mk.split('|'):
                if k in hd5['continuous']:
                    label_dict[mk][i] = hd5['continuous'][k][0]
        for k in gene_labels:
            if tp in samples2genes and samples2genes[tp] == k:
                label_dict[k][i] = 1

    return label_dict


def test_labels_to_label_map(test_labels: Dict[TensorMap, np.ndarray], examples: int) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    label_dict = {tm: np.zeros((examples,)) for tm in test_labels}
    categorical_labels = []
    continuous_labels = []

    for tm in test_labels:
        for i in range(examples):
            if tm.is_continuous() and tm.axes() == 1:
                label_dict[tm][i] = tm.rescale(test_labels[tm][i])
                continuous_labels.append(tm)
            elif tm.is_categorical() and tm.axes() == 1:
                label_dict[tm][i] = np.argmax(test_labels[tm][i])
                categorical_labels.append(tm)

    return label_dict, categorical_labels, continuous_labels


def infer_with_pixels(args):
    stats = Counter()
    tensor_paths_inferred = {}
    args.num_workers = 0
    inference_tsv = os.path.join(args.output_folder, args.id, 'pixel_inference_' + args.id + '.tsv')
    tensor_paths = [args.tensors + tp for tp in sorted(os.listdir(args.tensors)) if os.path.splitext(tp)[-1].lower() == TENSOR_EXT]
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    model = make_multimodal_multitask_model(**args.__dict__)
    generate_test = TensorGenerator(
        1, args.tensor_maps_in, args.tensor_maps_out, tensor_paths, num_workers=args.num_workers,
        cache_size=args.cache_size, keep_paths=True, mixup=args.mixup_alpha,
    )
    with open(inference_tsv, mode='w') as inference_file:
        inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['sample_id']
        for ot, otm in zip(args.output_tensors, args.tensor_maps_out):
            if len(otm.shape) == 1 and otm.is_continuous():
                header.extend([ot+'_prediction', ot+'_actual'])
            elif len(otm.shape) == 1 and otm.is_categorical():
                channel_columns = []
                for k in otm.channel_map:
                    channel_columns.append(ot + '_' + k + '_prediction')
                    channel_columns.append(ot + '_' + k + '_actual')
                header.extend(channel_columns)
            elif otm.name in ['mri_systole_diastole_8_segmented', 'sax_all_diastole_segmented']:
                pix_tm = args.tensor_maps_in[1]
                header.extend(['pixel_size', 'background_pixel_prediction', 'background_pixel_actual', 'ventricle_pixel_prediction', 'ventricle_pixel_actual', 'myocardium_pixel_prediction', 'myocardium_pixel_actual'])
                if otm.name == 'sax_all_diastole_segmented':
                    header.append('total_b_slices')
        inference_writer.writerow(header)

        while True:
            batch = next(generate_test)
            input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
            if tensor_paths[0] in tensor_paths_inferred:
                logging.info(f"Inference on {stats['count']} tensors finished. Inference TSV file at: {inference_tsv}")
                break

            prediction = model.predict(input_data)
            if len(args.tensor_maps_out) == 1:
                prediction = [prediction]

            csv_row = [os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, '')]  # extract sample id
            for y, tm in zip(prediction, args.tensor_maps_out):
                if len(tm.shape) == 1 and tm.is_continuous():
                    csv_row.append(str(tm.rescale(y)[0][0]))  # first index into batch then index into the 1x1 structure
                    if tm.sentinel is not None and tm.sentinel == output_data[tm.output_name()][0][0]:
                        csv_row.append("NA")
                    else:
                        csv_row.append(str(tm.rescale(output_data[tm.output_name()])[0][0]))
                elif len(tm.shape) == 1 and tm.is_categorical():
                    for k in tm.channel_map:
                        csv_row.append(str(y[0][tm.channel_map[k]]))
                        csv_row.append(str(output_data[tm.output_name()][0][tm.channel_map[k]]))
                elif tm.name in ['mri_systole_diastole_8_segmented', 'sax_all_diastole_segmented']:
                    csv_row.append(f"{pix_tm.rescale(input_data['input_mri_pixel_width_cine_segmented_sax_inlinevf_continuous'][0][0]):0.3f}")
                    csv_row.append(f'{np.sum(np.argmax(y, axis=-1) == MRI_SEGMENTED_CHANNEL_MAP["background"]):0.2f}')
                    csv_row.append(f'{np.sum(output_data[tm.output_name()][..., MRI_SEGMENTED_CHANNEL_MAP["background"]]):0.1f}')
                    csv_row.append(f'{np.sum(np.argmax(y, axis=-1) == MRI_SEGMENTED_CHANNEL_MAP["ventricle"]):0.2f}')
                    csv_row.append(f'{np.sum(output_data[tm.output_name()][..., MRI_SEGMENTED_CHANNEL_MAP["ventricle"]]):0.1f}')
                    csv_row.append(f'{np.sum(np.argmax(y, axis=-1) == MRI_SEGMENTED_CHANNEL_MAP["myocardium"]):0.2f}')
                    csv_row.append(f'{np.sum(output_data[tm.output_name()][..., MRI_SEGMENTED_CHANNEL_MAP["myocardium"]]):0.1f}')
                    if tm.name == 'sax_all_diastole_segmented':
                        background_counts = np.count_nonzero(output_data[tm.output_name()][..., MRI_SEGMENTED_CHANNEL_MAP["background"]] == 0, axis=(0, 1, 2))
                        csv_row.append(f'{np.count_nonzero(background_counts):0.0f}')

            inference_writer.writerow(csv_row)
            tensor_paths_inferred[tensor_paths[0]] = True
            stats['count'] += 1
            if stats['count'] % 250 == 0:
                logging.info(f"Wrote:{stats['count']} rows of inference.  Last tensor:{tensor_paths[0]}")


def _sample_with_heat(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def _2d_bbox_to_corner_and_size(bbox):
    total_axes = bbox.shape[-1] // 2
    lower_left_corner = (bbox[1], bbox[0])
    height = bbox[total_axes] - bbox[0]
    width = bbox[total_axes+1] - bbox[1]
    return lower_left_corner, width, height


def _plot_3d_tensor_slices_as_gray(tensor, figure_path, cols=3, rows=10, bboxes=[]):
    colors = ['blue', 'red', 'green', 'yellow']
    _, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    vmin = np.min(tensor)
    vmax = np.max(tensor)
    for i in range(tensor.shape[-1]):
        axes[i // cols, i % cols].imshow(tensor[:, :, i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i // cols, i % cols].set_yticklabels([])
        axes[i // cols, i % cols].set_xticklabels([])
        for c, bbox in enumerate(bboxes):
            corner, width, height = _2d_bbox_to_corner_and_size(bbox)
            axes[i // cols, i % cols].add_patch(matplotlib.patches.Rectangle(corner, width, height, linewidth=1, edgecolor=colors[c], facecolor='none'))

    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def _tabulate_correlations(
    stats: Dict[str, Dict[str, List[float]]],
    output_file_name: str,
    min_samples: int,
    output_folder_path: str,
) -> None:

    """
    Tabulate in pdf correlations of field values given in 'stats'
    :param stats: field names extracted from hd5 dataset names to list of values, one per sample_instance_arrayidx
    :param output_file_name: name of output file in pdf
    :param output_folder_path: directory that output file will be written to
    :param min_samples: calculate correlation coefficient only if both fields have values from that many common samples
    :return: None
    """

    fields = stats.keys()
    num_fields = len(fields)
    field_pairs = combinations(fields, 2)
    table_rows: List[list] = []
    logging.info(f"There are {int(num_fields * (num_fields - 1) / 2)} field pairs.")
    processed_field_pair_count = 0
    nan_counter = Counter()  # keep track of if we've seen a field have NaNs
    for field1, field2 in field_pairs:
        if field1 not in nan_counter.keys() and field2 not in nan_counter.keys():
            common_samples = set(stats[field1].keys()).intersection(stats[field2].keys())
            num_common_samples = len(common_samples)
            processed_field_pair_count += 1
            if processed_field_pair_count % 50000 == 0:
                logging.debug(f"Processed {processed_field_pair_count} field pairs.")
            if num_common_samples >= min_samples:
                field1_values = reduce(operator.concat, [stats[field1][sample] for sample in common_samples])
                field2_values = reduce(operator.concat, [stats[field2][sample] for sample in common_samples])

                num_field1_nans = len(list(filter(math.isnan, field1_values)))
                num_field2_nans = len(list(filter(math.isnan, field2_values)))
                at_least_one_field_has_nans = False
                if num_field1_nans != 0:
                    nan_counter[field1] = True
                    at_least_one_field_has_nans = True
                if num_field2_nans != 0:
                    nan_counter[field2] = True
                    at_least_one_field_has_nans = True
                if at_least_one_field_has_nans:
                    continue

                if len(field1_values) == len(field2_values):
                    if len(set(field1_values)) == 1 or len(set(field2_values)) == 1:
                        logging.debug(
                            f"Not calculating correlation for fields {field1} and {field2} because at least one of "
                            f"the fields has all the same values for the {num_common_samples} common samples.",
                        )
                        continue
                    corr = np.corrcoef(field1_values, field2_values)[1, 0]
                    if not math.isnan(corr):
                        table_rows.append([field1, field2, corr, corr * corr, num_common_samples])
                    else:
                        logging.warning(f"Pearson correlation for fields {field1} and {field2} is NaN.")
                else:
                    logging.debug(
                        f"Not calculating correlation for fields '{field1}' and '{field2}' "
                        f"because they have different number of values ({len(field1_values)} vs. {len(field2_values)}).",
                    )
        else:
            continue

    # Note: NaNs mess up sorting unless they are handled specially by a custom sorting function
    sorted_table_rows = sorted(table_rows, key=operator.itemgetter(2), reverse=True)
    logging.info(f"Total number of correlations: {len(sorted_table_rows)}")

    fields_with_nans = nan_counter.keys()
    if len(fields_with_nans) != 0:
        logging.warning(f"The {len(fields_with_nans)} fields containing NaNs are: {', '.join(fields_with_nans)}.")

    table_path = os.path.join(output_folder_path, output_file_name + CSV_EXT)
    table_header = ["Field 1", "Field 2", "Pearson R", "Pearson R^2",  "Sample Size"]
    df = pd.DataFrame(sorted_table_rows, columns=table_header)
    df.to_csv(table_path, index=False)

    logging.info(f"Saved correlations table at: {table_path}")


def _collect_continuous_stats_from_tensor_files(
    tensor_folder: str,
    max_samples: int = None,
    instances: List[str] = ['0', '1', '2'],
    max_arr_idx: int = None,
) -> Tuple[DefaultDict[str, DefaultDict[str, List[float]]], int]:
    if not os.path.exists(tensor_folder):
        raise ValueError('Source directory does not exist: ', tensor_folder)
    all_tensor_files = list(filter(lambda file: file.endswith(TENSOR_EXT), os.listdir(tensor_folder)))
    if max_samples is not None:
        if len(all_tensor_files) < max_samples:
            logging.warning(
                f"{max_samples} was specified as number of samples to use but there are only "
                f"{len(all_tensor_files)} tensor files in directory '{tensor_folder}'. Proceeding with those...",
            )
            max_samples = len(all_tensor_files)
        tensor_files = np.random.choice(all_tensor_files, max_samples, replace=False)
    else:
        tensor_files = all_tensor_files

    num_tensor_files = len(tensor_files)
    logging.info(f"Collecting continuous stats from {num_tensor_files} of {len(all_tensor_files)} tensors at {tensor_folder}...")

    # Declare the container to hold {field_1: {sample_1: [values], sample_2: [values], field_2:...}}
    stats: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    file_count = 0
    for tensor_file in tensor_files:
        _collect_continuous_stats_from_tensor_file(tensor_folder, tensor_file, stats, instances, max_arr_idx)
        file_count += 1
        if file_count % 1000 == 0:
            logging.debug(f"Collected continuous stats from {file_count}.")

    return stats, num_tensor_files


def _collect_continuous_stats_from_tensor_file(
    tensor_folder: str,
    tensor_file: str,
    stats: DefaultDict[str, DefaultDict[str, List[float]]],
    instances: List[str],
    max_arr_idx,
) -> None:
    # Inlining the method below to be able to reference more from the scope than the arguments of the function
    # 'h5py.visititems()' expects. It expects a func(<name>, <object>) => <None or return value>).
    def _field_meaning_to_values_dict(_, obj):
        if _is_continuous_valid_scalar_hd5_dataset(obj):
            value_in_tensor_file = obj[0]
            if value_in_tensor_file in CODING_VALUES_LESS_THAN_ONE:
                field_value = 0.5
            else:
                field_value = value_in_tensor_file
            dataset_name_parts = os.path.basename(obj.name).split(JOIN_CHAR)
            if len(dataset_name_parts) == 4:  # e.g. /continuous/1488_Tea-intake_0_0
                field_id = dataset_name_parts[0]
                field_meaning = dataset_name_parts[1]
                instance = dataset_name_parts[2]
                array_idx = dataset_name_parts[3]
                if instance in instances:
                    if max_arr_idx is None or (max_arr_idx is not None and int(array_idx) <= max_arr_idx):
                        stats[f"{field_meaning}{JOIN_CHAR}{field_id}{JOIN_CHAR}{instance}"][sample_id].append(field_value)
            else:  # e.g. /continuous/VentricularRate
                field_meaning = dataset_name_parts[0]
                stats[field_meaning][sample_id].append(field_value)
    tensor_file_path = os.path.join(tensor_folder, tensor_file)
    sample_id = os.path.splitext(tensor_file)[0]
    with h5py.File(tensor_file_path, 'r') as hd5_handle:
        hd5_handle.visititems(_field_meaning_to_values_dict)


def _is_continuous_valid_scalar_hd5_dataset(obj) -> bool:
    return obj.name.startswith('/continuous') and \
           isinstance(obj, h5py.Dataset) and \
           obj[0] not in CODING_VALUES_MISSING and \
           len(obj.shape) == 1


def _init_dict_of_tensors(tmaps: list) -> dict:
    # Iterate through tmaps and initialize dict in which to store
    # tensors, error types, and fpath
    dict_of_tensors = defaultdict(dict)
    for tm in tmaps:
        if tm.channel_map:
            for cm in tm.channel_map:
                dict_of_tensors[tm.name].update({(tm.name, cm): list()})
        else:
            dict_of_tensors[tm.name].update({f'{tm.name}': list()})
        dict_of_tensors[tm.name].update({f'error_type_{tm.name}': list()})
        dict_of_tensors[tm.name].update({'fpath': list()})
    return dict_of_tensors


def _hd5_to_dict(tmaps, path, gen_name, tot):
    with count.get_lock():
        i = count.value
        if (i+1) % 500 == 0:
            logging.info(f"{gen_name} - Parsing {i}/{tot} ({i/tot*100:.1f}%) done")
        count.value += 1
    try:
        with h5py.File(path, "r") as hd5:
            dict_of_tensor_dicts = defaultdict(lambda: _init_dict_of_tensors(tmaps))
            # Iterate through each tmap
            for tm in tmaps:
                shape = tm.shape if tm.shape[0] is not None else tm.shape[1:]
                try:
                    tensors = tm.tensor_from_file(tm, hd5)
                    if tm.shape[0] is not None:
                        # If not a multi-tensor tensor, wrap in array to loop through
                        tensors = np.array([tensors])
                    for i, tensor in enumerate(tensors):
                        if tensor == None:
                            break

                        error_type = ''
                        try:
                            tensor = tm.postprocess_tensor(tensor, augment=False, hd5=hd5)
                            # Append tensor to dict
                            if tm.channel_map:
                                for cm in tm.channel_map:
                                    dict_of_tensor_dicts[i][tm.name][(tm.name, cm)] = tensor[tm.channel_map[cm]]
                            else:
                                # If tensor is a scalar, isolate the value in the array;
                                # otherwise, retain the value as array
                                if shape[0] == 1:
                                    if type(tensor) == np.ndarray:
                                        tensor = tensor.item()
                                dict_of_tensor_dicts[i][tm.name][tm.name] = tensor
                        except (IndexError, KeyError, ValueError, OSError, RuntimeError) as e:
                            if tm.channel_map:
                                for cm in tm.channel_map:
                                    dict_of_tensor_dicts[i][tm.name][(tm.name, cm)] = np.nan
                            else:
                                dict_of_tensor_dicts[i][tm.name][tm.name] = np.full(shape, np.nan)[0]
                            error_type = type(e).__name__

                        # Save error type, fpath, and generator name (set)
                        dict_of_tensor_dicts[i][tm.name][f'error_type_{tm.name}'] = error_type
                        dict_of_tensor_dicts[i][tm.name]['fpath'] = path
                        dict_of_tensor_dicts[i][tm.name]['generator'] = gen_name

                except (IndexError, KeyError, ValueError, OSError, RuntimeError) as e:
                    # Most likely error came from tensor_from_file and dict_of_tensor_dicts is empty
                    if tm.channel_map:
                        for cm in tm.channel_map:
                            dict_of_tensor_dicts[0][tm.name][(tm.name, cm)] = np.nan
                    else:
                        dict_of_tensor_dicts[0][tm.name][tm.name] = np.full(shape, np.nan)[0]
                    dict_of_tensor_dicts[0][tm.name][f'error_type_{tm.name}'] = type(e).__name__
                    dict_of_tensor_dicts[0][tm.name]['fpath'] = path
                    dict_of_tensor_dicts[0][tm.name]['generator'] = gen_name

            # Append list of dicts with tensor_dict
            return dict_of_tensor_dicts
    except OSError as e:
        logging.info(f"OSError {e}")
        return None


def _tensors_to_df(args):
    generators = test_train_valid_tensor_generators(**args.__dict__)
    tmaps = [tm for tm in args.tensor_maps_in]
    global count # TODO figure out how to not use global
    count = Value('l', 0)
    paths = [(path, gen.name) for gen in generators for path in gen.path_iters[0].paths]
    tot = len(paths)
    with Pool(processes=None) as pool:
        list_of_dicts_of_dicts = pool.starmap(
            _hd5_to_dict,
            [(tmaps, path, gen_name, tot) for path, gen_name in paths],
        )

    num_hd5 = len(list_of_dicts_of_dicts)
    list_of_tensor_dicts = [dotd[i] for dotd in list_of_dicts_of_dicts for i in dotd if dotd is not None]

    # Now we have a list of dicts where each dict has {tmaps:values} and
    # each HD5 -> one dict in the list
    # Next, convert list of dicts -> dataframe
    df = pd.DataFrame()

    for tm in tmaps:
        # Isolate all {tmap:values} from the list of dicts for this tmap
        list_of_tmap_dicts = list(map(itemgetter(tm.name), list_of_tensor_dicts))

        # Convert this tmap-specific list of dicts into dict of lists
        dict_of_tmap_lists = {
            k: [d[k] for d in list_of_tmap_dicts]
            for k in list_of_tmap_dicts[0]
        }

        # Convert list of dicts into dataframe and concatenate to big df
        df = pd.concat([df, pd.DataFrame(dict_of_tmap_lists)], axis=1)

    # Remove duplicate columns: error_types, fpath
    df = df.loc[:, ~df.columns.duplicated()]

    # Remove "_worker" from "generator" values
    df["generator"].replace("_worker", "", regex=True, inplace=True)

    # Rearrange df columns so fpath and generator are at the end
    cols = [col for col in df if col not in ["fpath", "generator"]] \
           + ["fpath", "generator"]
    df = df[cols]

    # Cast dtype of some columns to string. Note this is necessary; although a
    # df (or pd.series) of floats will have the type "float", a df of strings
    # assumes a dtype of "object". Casting to dtype "string" will confer performnace
    # improvements in future versions of Pandas
    df["fpath"] = df["fpath"].astype("string")
    df["generator"] = df["generator"].astype("string")

    # Iterate through tensor (and channel) maps and cast Pandas dtype to string
    if Interpretation.LANGUAGE in [tm.interpretation for tm in tmaps]:
        for tm in [tm for tm in args.tensor_maps_in if tm.interpretation is Interpretation.LANGUAGE]:
            if tm.channel_map:
                for cm in tm.channel_map:
                    key = (tm.name, cm)
                    df[key] = df[key].astype("string")
            else:
                key = tm.name
                df[key] = df[key].astype("string")
    logging.info(f"Extracted {len(tmaps)} tmaps from {df.shape[0]} tensors across {num_hd5} hd5 files into DataFrame")
    return df


def explore(args):
    args.num_workers = 0
    tmaps = args.tensor_maps_in
    fpath_prefix = "summary_stats"
    tsv_style_is_genetics = 'genetics' in args.tsv_style
    out_ext = 'tsv' if tsv_style_is_genetics else 'csv'
    out_sep = '\t' if tsv_style_is_genetics else ','

    if any([len(tm.shape) != 1 for tm in tmaps]) and any([(len(tm.shape) == 2) and (tm.shape[0] is not None) for tm in tmaps]):
        raise ValueError("Explore only works for 1D tensor maps, but len(tm.shape) returned a value other than 1.")

    # Iterate through tensors, get tmaps, and save to dataframe
    df = _tensors_to_df(args)
    if tsv_style_is_genetics:
        fid = df['fpath'].str.split('/').str[-1].str.split('.').str[0]
        df.insert(0, 'FID', fid)
        df.insert(1, 'IID', fid)

    # Save dataframe to CSV
    fpath = os.path.join(args.output_folder, args.id, f"tensors_all_union.{out_ext}")
    df.to_csv(fpath, index=False, sep=out_sep)
    fpath = os.path.join(args.output_folder, args.id, f"tensors_all_intersect.{out_ext}")
    df.dropna().to_csv(fpath, index=False, sep=out_sep)
    logging.info(f"Saved dataframe of tensors (union and intersect) to {fpath}")

    # Check if any tmaps are categorical
    if Interpretation.CATEGORICAL in [tm.interpretation for tm in tmaps]:

        # Iterate through 1) df, 2) df without NaN-containing rows (intersect)
        for df_cur, df_str in zip([df, df.dropna()], ["union", "intersect"]):
            for tm in [tm for tm in tmaps if tm.interpretation is Interpretation.CATEGORICAL]:
                counts = []
                counts_missing = []
                if tm.channel_map:
                    for cm in tm.channel_map:
                        key = (tm.name, cm)
                        counts.append(df_cur[key].sum())
                        counts_missing.append(df_cur[key].isna().sum())
                else:
                    key = tm.name
                    counts.append(df_cur[key].sum())
                    counts_missing.append(df_cur[key].isna().sum())

                # Append list with missing counts
                counts.append(counts_missing[0])

                # Append list with total counts
                counts.append(sum(counts))

                # Create list of row names
                cm_names = [cm for cm in tm.channel_map] + [f"missing", f"total"]

                # Transform list into dataframe indexed by channel maps
                df_stats = pd.DataFrame(counts, index=cm_names, columns=["counts"])

                # Add new column: percent of all counts
                df_stats["percent_of_total"] = df_stats["counts"] / df_stats.loc[f"total"]["counts"] * 100

                # Save parent dataframe to CSV on disk
                fpath = os.path.join(
                    args.output_folder, args.id,
                    f"{fpath_prefix}_{Interpretation.CATEGORICAL}_{tm.name}_{df_str}.csv",
                )
                df_stats = df_stats.round(2)
                df_stats.to_csv(fpath)
                logging.info(f"Saved summary stats of {Interpretation.CATEGORICAL} {tm.name} tmaps to {fpath}")

    # Check if any tmaps are continuous
    if Interpretation.CONTINUOUS in [tm.interpretation for tm in tmaps]:

        # Iterate through 1) df, 2) df without NaN-containing rows (intersect)
        for df_cur, df_str in zip([df, df.dropna()], ["union", "intersect"]):
            df_stats = pd.DataFrame()
            if df_cur.empty:
                logging.info(
                    f"{df_str} of tensors results in empty dataframe."
                    f" Skipping calculations of {Interpretation.CONTINUOUS} summary statistics",
                )
            else:
                for tm in [tm for tm in tmaps if tm.interpretation is Interpretation.CONTINUOUS]:
                    if tm.channel_map:
                        for cm in tm.channel_map:
                            stats = dict()
                            key = (tm.name, cm)
                            stats["min"] = df_cur[key].min()
                            stats["max"] = df_cur[key].max()
                            stats["mean"] = df_cur[key].mean()
                            stats["median"] = df_cur[key].median()
                            stats["mode"] = df_cur[key].mode()[0]
                            stats["variance"] = df_cur[key].var()
                            stats["count"] = df_cur[key].count()
                            stats["missing"] = df_cur[key].isna().sum()
                            stats["total"] = len(df_cur[key])
                            stats["missing_percent"] = stats["missing"] / stats["total"] * 100
                            df_stats = pd.concat([df_stats, pd.DataFrame([stats], index=[cm])])
                    else:
                        stats = dict()
                        key = tm.name
                        stats["min"] = df_cur[key].min()
                        stats["max"] = df_cur[key].max()
                        stats["mean"] = df_cur[key].mean()
                        stats["median"] = df_cur[key].median()
                        stats["mode"] = df_cur[key].mode()[0]
                        stats["variance"] = df_cur[key].var()
                        stats["count"] = df_cur[key].count()
                        stats["missing"] = df_cur[key].isna().sum()
                        stats["total"] = len(df_cur[key])
                        stats["missing_percent"] = stats["missing"] / stats["total"] * 100
                        df_stats = pd.concat([df_stats, pd.DataFrame([stats], index=[key])])

                # Save parent dataframe to CSV on disk
                fpath = os.path.join(
                    args.output_folder, args.id,
                    f"{fpath_prefix}_{Interpretation.CONTINUOUS}_{df_str}.csv",
                )
                df_stats = df_stats.round(2)
                df_stats.to_csv(fpath)
                logging.info(f"Saved summary stats of {Interpretation.CONTINUOUS} tmaps to {fpath}")

    # Check if any tmaps are language (strings)
    if Interpretation.LANGUAGE in [tm.interpretation for tm in tmaps]:
        for df_cur, df_str in zip([df, df.dropna()], ["union", "intersect"]):
            df_stats = pd.DataFrame()
            if df_cur.empty:
                logging.info(
                    f"{df_str} of tensors results in empty dataframe."
                    f" Skipping calculations of {Interpretation.LANGUAGE} summary statistics",
                )
            else:
                for tm in [tm for tm in tmaps if tm.interpretation is Interpretation.LANGUAGE]:
                    if tm.channel_map:
                        for cm in tm.channel_map:
                            stats = dict()
                            key = (tm.name, cm)
                            stats["count"] = df_cur[key].count()
                            stats["count_unique"] = len(df_cur[key].value_counts())
                            stats["missing"] = df_cur[key].isna().sum()
                            stats["total"] = len(df_cur[key])
                            stats["missing_percent"] = stats["missing"] / stats["total"] * 100
                            df_stats = pd.concat([df_stats, pd.DataFrame([stats], index=[cm])])
                    else:
                        stats = dict()
                        key = tm.name
                        stats["count"] = df_cur[key].count()
                        stats["count_unique"] = len(df_cur[key].value_counts())
                        stats["missing"] = df_cur[key].isna().sum()
                        stats["total"] = len(df_cur[key])
                        stats["missing_percent"] = stats["missing"] / stats["total"] * 100
                        df_stats = pd.concat([df_stats, pd.DataFrame([stats], index=[tm.name])])

                # Save parent dataframe to CSV on disk
                fpath = os.path.join(
                    args.output_folder, args.id,
                    f"{fpath_prefix}_{Interpretation.LANGUAGE}_{df_str}.csv",
                )
                df_stats = df_stats.round(2)
                df_stats.to_csv(fpath)
                logging.info(f"Saved summary stats of {Interpretation.LANGUAGE} tmaps to {fpath}")


def _report_cross_reference(args, cross_reference_df, title):
    title = title.replace(' ', '_')
    if args.reference_label in cross_reference_df:
        labels, counts = np.unique(cross_reference_df[args.reference_label], return_counts=True)
        labels = np.append(labels, ['Total'])
        counts = np.append(counts, [sum(counts)])

        # save outcome distribution to csv
        df_out = pd.DataFrame({ 'counts': counts, args.reference_label: labels }).set_index(args.reference_label, drop=True)
        fpath = os.path.join(args.output_folder, args.id, f'distribution_{args.reference_label.replace(" ", "_")}_{title}.csv')
        df_out.to_csv(fpath)
        logging.info(f'Saved distribution of {args.reference_label} in cross reference to {fpath}')

    # save cross reference to csv
    fpath = os.path.join(args.output_folder, args.id, f'list_{title}.csv')
    cross_reference_df.set_index(args.join_tensors, drop=True).to_csv(fpath)
    logging.info(f'Saved cross reference to {fpath}')


def cross_reference(args):
    """Cross reference a source cohort with a reference cohort."""
    args.num_workers = 0
    cohort_counts = OrderedDict()

    src_path = args.tensors
    src_name = args.tensors_name
    src_join = args.join_tensors
    src_time = args.time_tensor
    ref_path = args.reference_tensors
    ref_name = args.reference_name
    ref_join = args.reference_join_tensors
    ref_start = args.reference_start_time_tensor
    ref_end = args.reference_end_time_tensor
    ref_label = args.reference_label

    # parse options
    src_cols = list(src_join)
    ref_cols = list(ref_join)
    if ref_label is not None:
        ref_cols.append(ref_label)

    use_time = src_time is not None and len(ref_start) != 0 and len(ref_end) != 0
    if use_time:
        src_cols.append(src_time)

        # ref start and end are lists where the first element is the name of the time tensor
        # and the second element is the offset to the value of the time tensor

        # if there is no second element in list, append 0 (if there is, still ok)
        [l.append(0) for l in [ref_start, ref_end]]

        # add unique column names to ref_cols
        ref_cols.extend({ref_start[0], ref_end[0]})

        # parse second element in list as int
        ref_start[1] = int(ref_start[1])
        ref_end[1] = int(ref_end[1])

    # load data into dataframes
    def _load_data(name, path, cols):
        if os.path.isdir(src_path):
            logging.debug(f'Assuming {name} is directory of hd5 at {path}')
            from ml4cvd.arguments import _get_tmap
            args.tensor_maps_in = [_get_tmap(it, cols) for it in cols]
            df = _tensors_to_df(args)[cols]
        else:
            logging.debug(f'Assuming {name} is a csv at {path}')
            df = pd.read_csv(path, usecols=cols, low_memory=False)
        return df

    src_df = _load_data(src_name, src_path, src_cols)
    logging.info(f'Loaded {src_name} into dataframe')
    ref_df = _load_data(ref_name, ref_path, ref_cols)
    logging.info(f'Loaded {ref_name} into dataframe')

    # cleanup time col
    if use_time:
        src_df[src_time] = pd.to_datetime(src_df[src_time], errors='coerce', infer_datetime_format=True)
        src_df.dropna(inplace=True)

        for ref_time in {ref_start[0], ref_end[0]}:
            ref_df[ref_time] = pd.to_datetime(ref_df[ref_time], errors='coerce', infer_datetime_format=True)
        ref_df.dropna(inplace=True)

        def _add_offset_time(ref_time):
            offset = ref_time[1]
            if offset == 0:
                return ref_time[0]
            ref_time_col = f'{ref_time[1]}_days_relative_{ref_time[0]}'
            ref_df[ref_time_col] = ref_df[ref_start[0]].apply(lambda x: x + datetime.timedelta(days=offset))
            ref_cols.append(ref_time_col)
            return ref_time_col

        ref_start = _add_offset_time(ref_start)
        ref_end = _add_offset_time(ref_end)

        time_description = f'between {ref_start.replace("_", " ")} and {ref_end.replace("_", " ")}'
    logging.info('Cleaned data columns and removed rows that could not be parsed')

    # drop duplicates based on cols
    src_df.drop_duplicates(subset=src_cols, inplace=True)
    ref_df.drop_duplicates(subset=ref_cols, inplace=True)
    logging.info('Removed duplicates from dataframes, based on join, time, and label')

    cohort_counts[f'{src_name} (total)'] = len(src_df)
    cohort_counts[f'{src_name} (unique {" + ".join(src_join)})'] = len(src_df.drop_duplicates(subset=src_join))
    cohort_counts[f'{ref_name} (total)'] = len(ref_df)
    cohort_counts[f'{ref_name} (unique {" + ".join(ref_join)})'] = len(ref_df.drop_duplicates(subset=ref_join))

    # merge on join columns
    cross_reference_df = src_df.merge(ref_df, how='inner', left_on=src_join, right_on=ref_join)
    logging.info('Cross referenced based on join tensors')

    cohort_counts[f'{src_name} in {ref_name} (unique {" + ".join(src_cols)})'] = len(cross_reference_df.drop_duplicates(subset=src_cols))
    cohort_counts[f'{src_name} in {ref_name} (unique {" + ".join(src_join)})'] = len(cross_reference_df.drop_duplicates(subset=src_join))
    cohort_counts[f'{ref_name} in {src_name} (unique {" + ".join(ref_cols)})'] = len(cross_reference_df.drop_duplicates(subset=ref_cols))
    cohort_counts[f'{ref_name} in {src_name} (unique {" + ".join(ref_join)})'] = len(cross_reference_df.drop_duplicates(subset=ref_join))

    # report cross_reference no time filter
    title = f'all {src_name} in {ref_name}'
    _report_cross_reference(args, cross_reference_df, title)

    if use_time:
        cross_reference_df = cross_reference_df[(cross_reference_df[ref_start] <= cross_reference_df[src_time]) & (cross_reference_df[ref_end] >= cross_reference_df[src_time])]
        logging.info('Cross referenced based on time')

        # At this point, rows in source have probably been duplicated by the join
        # This is fine, each row in reference has all associated rows in source now

        cohort_counts[f'{src_name} {time_description} (total - {src_name} may be duplicated if valid for multiple {ref_name})'] = len(cross_reference_df)
        cohort_counts[f'{src_name} {time_description} (unique {" + ".join(src_cols)})'] = len(cross_reference_df.drop_duplicates(subset=src_cols))

        # report cross_reference, all, time filtered
        title = f'all {src_name} {time_description}'
        _report_cross_reference(args, cross_reference_df, title)
        plot_cross_reference(args, cross_reference_df, title, time_description, ref_start, ref_end)

        # get most recent row in source for each row in reference
        # sort in ascending order so last() returns most recent
        cross_reference_df = cross_reference_df.sort_values(by=ref_cols+[src_time], ascending=True)
        cross_reference_df = cross_reference_df.groupby(by=ref_cols, as_index=False).last()[list(src_df) + list(ref_df)]
        logging.info(f'Found most recent {src_name} per {ref_name}')

        cohort_counts[f'Most recent {src_name} in {ref_name} {time_description} (total - {src_name} may be duplicated if valid for multiple {ref_name})'] = len(cross_reference_df)
        cohort_counts[f'Most recent {src_name} in {ref_name} {time_description} (unique {" + ".join(src_cols)})'] = len(cross_reference_df.drop_duplicates(subset=src_cols))

        # report cross_reference, most recent, time filtered
        title = f'most recent {src_name} {time_description}'
        _report_cross_reference(args, cross_reference_df, title)
        plot_cross_reference(args, cross_reference_df, title, time_description, ref_start, ref_end)

    # report counts
    fpath = os.path.join(args.output_folder, args.id, 'summary_cohort_counts.csv')
    pd.DataFrame.from_dict(cohort_counts, orient='index', columns=['count']).rename_axis('description').to_csv(fpath)
    logging.info(f'Saved cohort counts to {fpath}')
