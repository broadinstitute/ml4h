# explorations.py

# Imports
import os
import csv
import math
import operator
import datetime
from functools import reduce
from itertools import combinations
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Generator, Optional, DefaultDict

import h5py
import logging
import numpy as np
import pandas as pd
from keras.models import Model

import matplotlib
matplotlib.use('Agg')  # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt  # First import matplotlib, then use Agg, then import plt

from ml4cvd.TensorMap import TensorMap
from ml4cvd.models import embed_model_predict
from ml4cvd.plots import plot_histograms_in_pdf, plot_heatmap, plot_tsne, evaluate_predictions, subplot_rocs, subplot_scatters
from ml4cvd.defines import TENSOR_EXT, IMAGE_EXT, ECG_CHAR_2_IDX, ECG_IDX_2_CHAR, CODING_VALUES_MISSING, CODING_VALUES_LESS_THAN_ONE, JOIN_CHAR

CSV_EXT = '.tsv'


def find_tensors(text_file, tensor_folder, tensor_maps_out):
    with open(text_file, 'w') as f:
        for tensor_file in sorted([tensor_folder + tp for tp in os.listdir(tensor_folder) if os.path.splitext(tp)[-1].lower() == TENSOR_EXT]):
            with h5py.File(tensor_file, 'r') as hd5:
                for tm in tensor_maps_out:
                    if tm.is_categorical_date():
                        index = int(hd5[tm.name][0])
                        if index != 0:
                            disease_date = str2date(str(hd5[tm.name + '_date'][0]))
                            assess_date = str2date(str(hd5['assessment-date_0_0'][0]))
                            if disease_date < assess_date:
                                f.write(f"{tensor_file}\tPrevalent {tm.name}\n")
                            else:
                                f.write(f"{tensor_file}\tIncident {tm.name}\n")


def sort_csv(input_csv_file, volume_csv):
    lvef = {}
    with open(volume_csv, 'r') as volumes:
        lol = list(csv.reader(volumes, delimiter='\t'))
        logging.info('CSV of MRI volumes header:{}'.format(list(enumerate(lol[0]))))
        for row in lol[1:]:
            sample_id = row[0]
            if row[5] != 'NA':
                lvef[sample_id] = float(row[5])

    print('try:', input_csv_file.replace(CSV_EXT, '_diff_sorted'+CSV_EXT))
    with open(input_csv_file, mode='r') as input_csv:
        with open(input_csv_file.replace(CSV_EXT, '_diff_sorted'+CSV_EXT), mode='w') as output_csv:
            csv_writer = csv.writer(output_csv, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_reader = csv.reader(input_csv, delimiter='\t')
            csv_writer.writerow(next(csv_reader)+['discrepancy'])
            csv_sorted = sorted(csv_reader, key=lambda row: abs(float(lvef[row[0]])-float(row[5])), reverse=True)
            [csv_writer.writerow(row + [float(lvef[row[0]])-float(row[5])]) for row in csv_sorted]


def predictions_to_pngs(predictions: np.ndarray, tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap], data: Dict[str, np.ndarray],
                        labels: Dict[str, np.ndarray], paths: List[str], folder: str) -> None:
    for y, tm in zip(predictions, tensor_maps_out):
        if not isinstance(predictions, list):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
            y = predictions
        logging.info(f"Write segmented MRI y:{y.shape} labels:{labels[tm.output_name()].shape} folder:{folder}")
        if len(tm.shape) == 3:
            input_map = None
            for im in tensor_maps_in:
                if tm.is_categorical_any() and im.dependent_map == tm:
                    input_map = im
                elif len(tm.shape) == len(im.shape):
                    input_map = im
            for i in range(y.shape[0]):
                sample_id = os.path.basename(paths[i]).replace(TENSOR_EXT, '')
                if tm.is_categorical_any():
                    plt.imsave(f"{folder}{sample_id}_truth_{i:02d}{IMAGE_EXT}", np.argmax(labels[tm.output_name()][i], axis=-1))
                    plt.imsave(f"{folder}{sample_id}_prediction_{i:02d}{IMAGE_EXT}", np.argmax(y[i], axis=-1))
                    if input_map is not None:
                        plt.imsave(f"{folder}{sample_id}_mri_slice_{i:02d}{IMAGE_EXT}", data[input_map.input_name()][i, :, :, 0])
                else:
                    for j in range(y.shape[3]):
                        plt.imsave(f"{folder}{sample_id}_truth_{i:02d}_{j:02d}{IMAGE_EXT}", labels[tm.output_name()][i, :, :, j])
                        plt.imsave(f"{folder}{sample_id}_prediction_{i:02d}_{j:02d}{IMAGE_EXT}", y[i, :, :, j])
                        plt.imsave(f"{folder}{sample_id}_mri_slice_{i:02d}_{j:02d}{IMAGE_EXT}", data[input_map.input_name()][i, :, :, j])
        elif len(tm.shape) == 4:
            for im in tensor_maps_in:
                if im.dependent_map == tm:
                    break
            for i in range(y.shape[0]):
                sample_id = os.path.basename(paths[i]).replace(TENSOR_EXT, '')
                for j in range(y.shape[3]):
                    if tm.is_categorical_any():
                        truth = np.argmax(labels[tm.output_name()][i, :, :, j, :], axis=-1)
                        prediction = np.argmax(y[i, :, :, j, :], axis=-1)
                        true_donut = np.ma.masked_where(truth == 2, data[im.input_name()][i, :, :, j, 0])
                        predict_donut = np.ma.masked_where(prediction == 2, data[im.input_name()][i, :, :, j, 0])
                        plt.imsave(folder+sample_id+'_truth_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, truth)
                        plt.imsave(folder+sample_id+'_prediction_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, prediction)
                        plt.imsave(folder+sample_id+'_mri_slice_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, data[im.input_name()][i, :, :, j, 0])
                        plt.imsave(folder+sample_id + '_true_donut_{0:03d}_{1:03d}'.format(i, j) + IMAGE_EXT, true_donut)
                        plt.imsave(folder + sample_id + '_predict_donut_{0:03d}_{1:03d}'.format(i, j) + IMAGE_EXT, predict_donut)
                    else:
                        plt.imsave(folder+sample_id+'_truth_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, labels[tm.output_name()][i, :, :, j, 0])
                        plt.imsave(folder+sample_id+'_prediction_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, y[i, :, :, j, 0])


def plot_while_learning(model, tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
                        generate_train: Generator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[List[str]]], None, None],
                        test_data: Dict[str, np.ndarray], test_labels: Dict[str, np.ndarray], test_paths: List[str], epochs: int, batch_size: int,
                        training_steps: int, folder: str, write_pngs: bool):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(epochs):
        rocs = []
        scatters = []
        predictions = model.predict(test_data, batch_size=batch_size)
        for y, tm in zip(predictions, tensor_maps_out):
            if len(tensor_maps_out) == 1:
                predictions = [predictions]
            if not write_pngs:
                if tm.is_categorical_any() and len(tm.shape) == 3:
                    for im in tensor_maps_in:
                        if im.dependent_map == tm:
                            break
                    logging.info(f"epoch:{i} write segmented mris y shape:{y.shape} label shape:{test_labels[tm.output_name()].shape} to folder:{folder}")
                    for yi in range(y.shape[0]):
                        plt.imsave(f"{folder}batch_index_{yi}_truth_epoch_{i:03d}{IMAGE_EXT}", np.argmax(test_labels[tm.output_name()][yi], axis=-1))
                        plt.imsave(f"{folder}batch_index_{yi}_prediction_epoch_{i:03d}{IMAGE_EXT}", np.argmax(y[yi], axis=-1))
                        plt.imsave(f"{folder}batch_index_{yi}_mri_epoch_{i:03d}{IMAGE_EXT}", test_data[im.input_name()][yi, :, :, 0])
                elif tm.is_categorical_any() and len(tm.shape) == 4:
                    for im in tensor_maps_in:
                        if im.dependent_map == tm:
                            break
                    logging.info(f"epoch:{i} write segmented mris y shape:{y.shape} label shape:{test_labels[tm.output_name()].shape} to folder:{folder}")
                    for yi in range(y.shape[0]):
                        for j in range(y.shape[3]):
                            truth = np.argmax(test_labels[tm.output_name()][yi, :, :, j, :], axis=-1)
                            prediction = np.argmax(y[yi, :, :, j, :], axis=-1)
                            true_donut = np.ma.masked_where(truth == 2, test_data[im.input_name()][yi, :, :, j, 0])
                            predict_donut = np.ma.masked_where(prediction == 2, test_data[im.input_name()][yi, :, :, j, 0])
                            plt.imsave(f"{folder}batch_index_{yi}_slice_{j:03d}_prediction_epoch_{i:03d}{IMAGE_EXT}", prediction)
                            plt.imsave(f"{folder}batch_index_{yi}_slice_{j:03d}_predict_donut_epoch_{i:03d}{IMAGE_EXT}", predict_donut)
                            if i == 0:
                                plt.imsave(f"{folder}batch_index_{yi}_slice_{j:03d}_truth_epoch_{i:03d}{IMAGE_EXT}", truth)
                                plt.imsave(f"{folder}batch_index_{yi}_slice_{j:03d}_true_donut_epoch_{i:03d}{IMAGE_EXT}", true_donut)
                                plt.imsave(f"{folder}batch_index_{yi}_slice_{j:03d}_mri_epoch_{i:03d}{IMAGE_EXT}", test_data[im.input_name()][yi, :, :, j, 0])

            elif write_pngs:
                if len(tensor_maps_out) == 1:
                    y = predictions[0]
                evaluate_predictions(tm, y, test_labels[tm.output_name()], f"{tm.name}_epoch_{i:03d}", folder, test_paths, rocs=rocs, scatters=scatters)
        if len(rocs) > 1:
            subplot_rocs(rocs, folder+f"epoch_{i:03d}_")
        if len(scatters) > 1:
            subplot_scatters(scatters, folder+f"epoch_{i:03d}_")

        model.fit_generator(generate_train, steps_per_epoch=training_steps, epochs=1, verbose=1)


def plot_histograms_from_tensor_files_in_pdf(run_id: str,
                                             tensor_folder: str,
                                             output_folder: str,
                                             max_samples: int = None) -> None:
    """
    :param id: name for the plotting run
    :param tensor_folder: directory with tensor files to plot histograms from
    :param output_folder: folder containing the output plot
    :param max_samples: specifies how many tensor files to down-sample from; by default all tensors are used
    """
    stats, num_tensor_files = _collect_continuous_stats_from_tensor_files(tensor_folder, max_samples)
    logging.info(f"Collected continuous stats for {len(stats)} fields. Now plotting histograms of them...")
    plot_histograms_in_pdf(stats, num_tensor_files, run_id, output_folder)


def plot_heatmap_from_tensor_files(id: str,
                                   tensor_folder: str,
                                   output_folder: str,
                                   min_samples: int,
                                   max_samples: int = None) -> None:
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


def tabulate_correlations_from_tensor_files(run_id: str,
                                            tensor_folder: str,
                                            output_folder: str,
                                            min_samples: int,
                                            max_samples: int = None) -> None:
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


def sample_from_char_model(char_model: Model, test_batch: Dict[str, np.ndarray], test_paths: List[str]) -> None:
    window_size = test_batch['input_ecg_rest_text_ecg_text'].shape[1]
    alphabet_size = test_batch['input_ecg_rest_text_ecg_text'].shape[2]
    for i in range(test_batch['input_embed_hidden_layer'].shape[0]):
        count = 0
        sentence = ''
        next_char = ''
        embed_in = test_batch['input_embed_hidden_layer'][i:i+1, :]
        burn_in = np.zeros((1, window_size, alphabet_size), dtype=np.float32)
        window_size = burn_in.shape[1]
        with h5py.File(test_paths[i], 'r') as hd5:
            logging.info(f"\n")
            logging.info(f"Real text: {str(hd5['ecg_rest_text'][0]).strip()}")
        while next_char != '!' and count < 400:
            cur_test = {'input_embed_hidden_layer': embed_in, 'input_ecg_rest_text_ecg_text': burn_in}
            y_pred = char_model.predict(cur_test)
            next_char = ECG_IDX_2_CHAR[_sample_with_heat(y_pred[0, :], 0.7)]
            sentence += next_char
            burn_in = np.zeros((1,) + test_batch['input_ecg_rest_text_ecg_text'].shape[1:], dtype=np.float32)
            for j, c in enumerate(reversed(sentence)):
                if j == window_size:
                    break
                burn_in[0, window_size-j-1, ECG_CHAR_2_IDX[c]] = 1.0
            count += 1
        logging.info(f"Model text:{sentence}")


def tensors_to_label_dictionary(categorical_labels: List,
                                continuous_labels: List,
                                gene_labels: List,
                                samples2genes: Dict[str, str],
                                test_paths: List) -> Dict[str, np.ndarray]:
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


def test_labels_to_label_dictionary(test_labels: Dict[TensorMap, np.ndarray], examples: int) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    label_dict = {tm: np.zeros((examples,)) for tm in test_labels}
    categorical_labels = []
    continuous_labels = []

    for tm in test_labels:
        for i in range(examples):
            if tm.is_continuous():
                label_dict[tm][i] = tm.rescale(test_labels[tm][i])
                continuous_labels.append(tm)
            else:
                label_dict[tm][i] = np.argmax(test_labels[tm][i])
                categorical_labels.append(tm)

    return label_dict, categorical_labels, continuous_labels


def _sample_with_heat(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def _tabulate_correlations(stats: Dict[str, Dict[str, List[float]]],
                           output_file_name: str,
                           min_samples: int,
                           output_folder_path: str) -> None:

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
                        logging.debug(f"Not calculating correlation for fields {field1} and {field2} because at least one of "
                                      f"the fields has all the same values for the {num_common_samples} common samples.")
                        continue
                    corr = np.corrcoef(field1_values, field2_values)[1, 0]
                    if not math.isnan(corr):
                        table_rows.append([field1, field2, corr, corr * corr, num_common_samples])
                    else:
                        logging.warning(f"Pearson correlation for fields {field1} and {field2} is NaN.")
                else:
                    logging.debug(f"Not calculating correlation for fields '{field1}' and '{field2}' "
                                  f"because they have different number of values ({len(field1_values)} vs. {len(field2_values)}).")
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


def _collect_continuous_stats_from_tensor_files(tensor_folder: str,
                                                max_samples: int = None,
                                                instances: List[str] = ['0', '1', '2'],
                                                max_arr_idx: int = None) -> Tuple[DefaultDict[str, DefaultDict[str, List[float]]], int]:
    if not os.path.exists(tensor_folder):
        raise ValueError('Source directory does not exist: ', tensor_folder)
    all_tensor_files = list(filter(lambda file: file.endswith(TENSOR_EXT), os.listdir(tensor_folder)))
    if max_samples is not None:
        if len(all_tensor_files) < max_samples:
            logging.warning(f"{max_samples} was specified as number of samples to use but there are only "
                            f"{len(all_tensor_files)} tensor files in directory '{tensor_folder}'. Proceeding with those...")
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


def _collect_continuous_stats_from_tensor_file(tensor_folder: str,
                                               tensor_file: str,
                                               stats: DefaultDict[str, DefaultDict[str, List[float]]],
                                               instances: List[str],
                                               max_arr_idx) -> None:
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
