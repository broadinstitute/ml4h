# explorations.py

# Imports
import os
import h5py
import logging
import datetime
import numpy as np
from typing import Dict, List, Tuple, Generator, Optional

import matplotlib
matplotlib.use('Agg')  # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt  # First import matplotlib, then use Agg, then import plt

from keras.models import Model

from ml4cvd.TensorMap import TensorMap
from ml4cvd.plots import evaluate_predictions
from ml4cvd.defines import TENSOR_EXT, IMAGE_EXT, ECG_CHAR_2_IDX, ECG_IDX_2_CHAR


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


def predictions_to_pngs(predictions: np.ndarray, tensor_maps_in: List[TensorMap],
                        tensor_maps_out: List[TensorMap],
                        data: Dict[str, np.ndarray],
                        labels: Dict[str, np.ndarray],
                        paths: List[str],
                        folder: str):
    for y, tm in zip(predictions, tensor_maps_out):
        logging.info(f"Write segmented MRI y:{y.shape} labels:{labels[tm.output_name()].shape} folder:{folder}")
        if tm.is_categorical_any() and len(tm.shape) == 3:
            for im in tensor_maps_in:
                if im.dependent_map == tm:
                    break
            for i in range(y.shape[0]):
                sample_id = os.path.basename(paths[i]).replace(TENSOR_EXT, '')
                plt.imsave(folder + sample_id + '_truth_{0:03d}'.format(i) + IMAGE_EXT,
                           np.argmax(labels[tm.output_name()][i], axis=-1))
                plt.imsave(folder + sample_id + '_prediction_{0:03d}'.format(i) + IMAGE_EXT, np.argmax(y[i], axis=-1))
                plt.imsave(folder + sample_id + '_mri_slice_{0:03d}'.format(i)+IMAGE_EXT, data[im.input_name()][i, :, :, 0])

        elif tm.is_categorical_any() and len(tm.shape) == 4:
            for im in tensor_maps_in:
                if im.dependent_map == tm:
                    break
            for i in range(y.shape[0]):
                sample_id = os.path.basename(paths[i]).replace(TENSOR_EXT, '')
                for j in range(y.shape[3]):
                    truth = np.argmax(labels[tm.output_name()][i, :, :, j, :], axis=-1)
                    prediction = np.argmax(y[i, :, :, j, :], axis=-1)
                    true_donut = np.ma.masked_where(truth == 2, data[im.input_name()][i, :, :, j, 0])
                    predict_donut = np.ma.masked_where(prediction == 2, data[im.input_name()][i, :, :, j, 0])
                    plt.imsave(folder+sample_id+'_truth_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, truth)
                    plt.imsave(folder+sample_id+'_prediction_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, prediction)
                    plt.imsave(folder+sample_id+'_mri_slice_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, data[im.input_name()][i, :, :, j, 0])
                    plt.imsave(folder+sample_id+'_true_donut_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, true_donut)
                    plt.imsave(folder+sample_id+'_predict_donut_{0:03d}_{1:03d}'.format(i, j)+IMAGE_EXT, predict_donut)


def plot_while_learning(model, tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
                        generate_train: Generator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[List[str]]], None, None],
                        data: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], test_paths: List[str], epochs: int, batch_size: int,
                        training_steps: int, folder: str, run_id: str, write_pngs: bool):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(epochs):
        predictions = model.predict(data, batch_size=batch_size)
        for y, tm in zip(predictions, tensor_maps_out):
            if len(tensor_maps_out) == 1:
                predictions = [predictions]
            if not write_pngs:
                if tm.is_categorical_any() and len(tm.shape) == 3:
                    for im in tensor_maps_in:
                        if im.dependent_map == tm:
                            break
                    logging.info(f"epoch:{i} write segmented mris y shape:{y.shape} label shape:{labels[tm.output_name()].shape} to folder:{folder}")
                    for yi in range(y.shape[0]):
                        plt.imsave(folder+str(yi)+'_truth_epoch_{0:03d}'.format(i)+IMAGE_EXT, np.argmax(labels[tm.output_name()][yi], axis=-1))
                        plt.imsave(folder+str(yi)+'_prediction_epoch_{0:03d}'.format(i)+IMAGE_EXT, np.argmax(y[yi], axis=-1))
                        plt.imsave(folder+str(yi)+'_mri_slice_epoch_{0:03d}'.format(i)+IMAGE_EXT, data[im.input_name()][yi,:,:,0])
                elif tm.is_categorical_any() and len(tm.shape) == 4:
                    for im in tensor_maps_in:
                        if im.dependent_map == tm:
                            break
                    logging.info(f"epoch:{i} write segmented mris y shape:{y.shape} label shape:{labels[tm.output_name()].shape} to folder:{folder}")
                    for yi in range(y.shape[0]):
                        for j in range(y.shape[3]):
                            truth = np.argmax(labels[tm.output_name()][yi,:,:,j,:], axis=-1)
                            prediction = np.argmax(y[yi,:,:,j,:], axis=-1)
                            true_donut = np.ma.masked_where(truth == 2, data[im.input_name()][yi,:,:,j,0])
                            predict_donut = np.ma.masked_where(prediction == 2, data[im.input_name()][yi,:,:,j,0])
                            plt.imsave(folder+str(yi)+'_slice_{0:03d}_prediction_epoch_{1:03d}'.format(j, i)+IMAGE_EXT, prediction)
                            plt.imsave(folder+str(yi)+'_slice_{0:03d}_predict_donut_epoch_{1:03d}'.format(j, i)+IMAGE_EXT, predict_donut)
                            if i == 0:
                                plt.imsave(folder+str(yi)+'_slice_{0:03d}_truth_epoch_{1:03d}'.format(j, i)+IMAGE_EXT, truth)
                                plt.imsave(folder+str(yi)+'_slice_{0:03d}_true_donut_epoch_{1:03d}'.format(j, i)+IMAGE_EXT, true_donut)
                                plt.imsave(folder+str(yi)+'_slice_{0:03d}_mri_epoch_{1:03d}'.format(j, i)+IMAGE_EXT, data[im.input_name()][yi,:,:,j,0])

            elif write_pngs:
                title = tm.name+'_epoch_{0:03d}'.format(i)
                metric_folder = os.path.join(folder, run_id, 'training_metrics/')
                if len(tensor_maps_out) == 1:
                    y = predictions[0]
                evaluate_predictions(tm, y, labels, data, title, metric_folder)

        model.fit_generator(generate_train, steps_per_epoch=training_steps, epochs=1, verbose=1)


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


def _sample_with_heat(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)