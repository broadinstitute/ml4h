import os
import csv
import time
import h5py
import copy
import biosppy
import seaborn as sns
import logging
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Dict, Any, Set
from itertools import combinations
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tensorflow.keras import Model
import datetime
import gc

from ml4cvd.defines import TENSOR_EXT, MODEL_EXT
from ml4cvd.logger import load_config
from ml4cvd.TensorMap import TensorMap, Interpretation, no_nans
from ml4cvd.tensor_writer_ukbb import tensor_path, first_dataset_at_path
from ml4cvd.normalizer import Standardize, Normalizer
from ml4cvd.tensor_from_file import _get_tensor_at_first_date
from ml4cvd.explorations import explore
from ml4cvd.tensor_generators import test_train_valid_tensor_generators, TensorGenerator
from ml4cvd.tensor_generators import BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX
from ml4cvd.models import train_model_from_generators, make_multimodal_multitask_model, BottleneckType
from ml4cvd.recipes import _make_tmap_nan_on_fail
from ml4cvd.metrics import coefficient_of_determination



PRETEST_DUR = 15  # DURs are measured in seconds
EXERCISE_DUR = 360
RECOVERY_DUR = 60
SAMPLING_RATE = 500
HRR_TIME = 50
HR_MEASUREMENT_TIMES = 0, HRR_TIME  # relative to recovery start
HR_SEGMENT_DUR = 10  # HR measurements in recovery coalesced across a segment of this length
TREND_TRACE_DUR_DIFF = 2  # Sum of phase durations from UKBB is 2s longer than the raw traces
LEAD_NAMES = 'lead_I', 'lead_2', 'lead_3'

TENSOR_FOLDER = '/mnt/disks/ecg-bike-tensors/2019-10-10/'
USER = 'ndiamant'
OUTPUT_FOLDER = f'/home/{USER}/ml/hrr_results'
PRETEST_EXPLORE_ID = 'pretest_explore'
EXPLORE_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, PRETEST_EXPLORE_ID)
EXPLORE_RESULT = os.path.join(EXPLORE_OUTPUT_FOLDER, 'tensors_all_union.csv')
TRAIN_CSV = os.path.join(OUTPUT_FOLDER, 'train_ids.csv')
VALID_CSV = os.path.join(OUTPUT_FOLDER, 'valid_ids.csv')
TEST_CSV = os.path.join(OUTPUT_FOLDER, 'test_ids.csv')

BIOSPPY_MEASUREMENTS_FILE = os.path.join(OUTPUT_FOLDER, 'biosppy_hr_recovery_measurements.csv')
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, 'figures')
BIOSPPY_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'biosppy')

PRETEST_LABEL_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'pretest_labels')
PRETEST_QUANTILE_CUTOFF = .99
PRETEST_LABEL_FILE = os.path.join(OUTPUT_FOLDER, f'hr_pretest_training_data.csv')
PRETEST_TRAINING_DUR = 10  # number of seconds of pretest ECG used for prediction

VALID_RATIO = .05
TEST_RATIO = .15
TRAIN_RATIO = 1 - VALID_RATIO - TEST_RATIO

PRETEST_MODEL_ID = 'pretest_model'
PRETEST_MODEL_PATH = os.path.join(OUTPUT_FOLDER, PRETEST_MODEL_ID, PRETEST_MODEL_ID + MODEL_EXT)
PRETEST_MODEL_LEADS = [0]
SEED = 217
PRETEST_INFERENCE_FILE = os.path.join(OUTPUT_FOLDER, 'pretest_model_inference.tsv')


# Tensor from file helpers
def _check_phase_full_len(hd5: h5py.File, phase: str):
    phase_len = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', f'{phase}_duration')
    valid = True
    if phase == 'pretest':
        valid &= phase_len == PRETEST_DUR
    elif phase == 'exercise':
        valid &= phase_len == EXERCISE_DUR
    elif phase == 'rest':
        valid &= phase_len == RECOVERY_DUR
    else:
        raise ValueError(f'Phase {phase} is not a valid phase.')
    if not valid:
        raise ValueError(f'{phase} phase is not full length.')


def _get_bike_ecg(hd5: h5py.File, start: int, stop: int, leads: Union[List[int], slice]):
    path_prefix, name = 'ecg_bike/float_array', 'full'
    ecg_dataset = first_dataset_at_path(hd5, tensor_path(path_prefix, name))
    tensor = np.array(ecg_dataset[start: stop, leads], dtype=np.float32)
    return tensor


def _get_downsampled_bike_ecg(length: float, hd5: h5py.File, start: int, rate: float, leads: Union[List[int], slice]):
    length = int(length * rate)
    ecg = _get_bike_ecg(hd5, start, start + length, leads)
    ecg = _downsample_ecg(ecg, rate)
    return ecg


def _make_pretest_ecg_tff(downsample_rate: float, leads: Union[List[int], slice], random_start=True):
    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        _check_phase_full_len(hd5, 'pretest')
        start = np.random.randint(0, SAMPLING_RATE * PRETEST_DUR - tm.shape[0] * downsample_rate) if random_start else 0
        return _get_downsampled_bike_ecg(tm.shape[0], hd5, start, downsample_rate, leads)
    return tff


def _make_downsampled_rest_tff(downsample_rate: float):
    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for k, idx in tm.channel_map.items():
            data = np.array(tm.hd5_first_dataset_in_group(hd5, f'{tm.path_prefix}/{k}/'))[:, np.newaxis]
            tensor[:, tm.channel_map[k]] = _downsample_ecg(data, downsample_rate)[0]
        return tensor
    return tff


def _get_trace_recovery_start(hd5: h5py.File) -> int:
    _check_phase_full_len(hd5, 'rest')
    pretest_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'pretest_duration')
    exercise_dur = _get_tensor_at_first_date(hd5, 'ecg_bike/continuous', 'exercise_duration')
    return int(SAMPLING_RATE * (pretest_dur + exercise_dur - HR_SEGMENT_DUR / 2 - TREND_TRACE_DUR_DIFF))


# ECG transformations
def _warp_ecg(ecg):
    warp_strength = .02
    i = np.linspace(0, 1, len(ecg))
    envelope = warp_strength * (.5 - np.abs(.5 - i))
    warped = i + envelope * (
        np.sin(np.random.rand() * 5 + np.random.randn() * 5)
        + np.cos(np.random.rand() * 5 + np.random.randn() * 5)
    )
    warped_ecg = np.zeros_like(ecg)
    for j in range(ecg.shape[1]):
        warped_ecg[:, j] = np.interp(i, warped, ecg[:, j])
    return warped_ecg


def _random_crop_ecg(ecg):
    cropped_ecg = ecg.copy()
    for j in range(ecg.shape[1]):
        crop_len = np.random.randint(len(ecg)) // 3
        crop_start = max(0, np.random.randint(-crop_len, len(ecg)))
        cropped_ecg[:, j][crop_start: crop_start + crop_len] = np.random.randn()
    return cropped_ecg


def _downsample_ecg(ecg, rate: float):
    """
    rate=2 halves the sampling rate. Uses linear interpolation. Requires ECG to be divisible by rate.
    """
    new_len = ecg.shape[0] // rate
    i = np.linspace(0, 1, new_len)
    x = np.linspace(0, 1, ecg.shape[0])
    downsampled = np.zeros((ecg.shape[0] // rate, ecg.shape[1]))
    for j in range(ecg.shape[1]):
        downsampled[:, j] = np.interp(i, x, ecg[:, j])
    return downsampled


def _rand_add_noise(ecg):
    noise_frac = np.random.rand() * .1  # max of 10% noise
    return ecg + noise_frac * ecg.mean(axis=0) * np.random.randn(*ecg.shape)


def make_pretest_tmap(downsample_rate: float, leads) -> TensorMap:
    return TensorMap(
        'pretest_ecg', shape=(int(PRETEST_TRAINING_DUR * SAMPLING_RATE // downsample_rate), len(leads)),
        interpretation=Interpretation.CONTINUOUS,
        validator=no_nans, normalization=Standardize(0, 100),
        tensor_from_file=_make_pretest_ecg_tff(downsample_rate, leads),
        cacheable=False, augmentations=[_warp_ecg, _rand_add_noise, _random_crop_ecg],
    )


# HR measurements from biosppy
BIOSPPY_DOWNSAMPLE_RATE = 4


def _get_segment_for_biosppy(ecg, mid_time: int):
    center = mid_time * SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE
    offset = (SAMPLING_RATE * HR_SEGMENT_DUR // BIOSPPY_DOWNSAMPLE_RATE) // 2
    return ecg[center - offset: center + offset]


def _get_biosppy_hr(segment: np.ndarray) -> float:
    return float(
        np.median(
            biosppy.signals.ecg.ecg(segment, sampling_rate=SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE, show=False)[-1],
        ),
    )


def _get_segments_for_biosppy(hd5: h5py.File):
    recovery_start_idx = _get_trace_recovery_start(hd5)
    length = (HR_MEASUREMENT_TIMES[-1] - HR_MEASUREMENT_TIMES[0] + HR_SEGMENT_DUR) * SAMPLING_RATE // BIOSPPY_DOWNSAMPLE_RATE
    ecg = _get_downsampled_bike_ecg(length, hd5, recovery_start_idx, BIOSPPY_DOWNSAMPLE_RATE, [0, 1, 2])
    for mid_time in HR_MEASUREMENT_TIMES:
        yield _get_segment_for_biosppy(ecg, mid_time + HR_SEGMENT_DUR // 2)


def _hr_and_diffs_from_segment(segment: np.ndarray) -> Tuple[float, float]:
    hr_per_lead = [_get_biosppy_hr(segment[:, i]) for i in range(segment.shape[-1])]
    max_diff = max(map(lambda pair: abs(pair[0] - pair[1]), combinations(hr_per_lead, 2)))
    return float(np.median(hr_per_lead)), max_diff


def _plot_segment(segment: np.ndarray):
    hr, max_diff = _hr_and_diffs_from_segment(segment)
    t = np.linspace(0, HR_SEGMENT_DUR, len(segment))
    for i, lead_name in enumerate(LEAD_NAMES):
        plt.plot(t, segment[:, i], label=lead_name)
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title(f'hr: {hr:.2f}, max hr difference between leads: {max_diff:.2f}')


def plot_segment_prediction(sample_id: str, t: int, pred: float, actual: float, diff: float):
    t_idx = HR_MEASUREMENT_TIMES.index(t)
    with h5py.File(_path_from_sample_id(sample_id), 'r') as hd5:
        segment = list(_get_segments_for_biosppy(hd5))[t_idx]
        x = np.linspace(0, HR_SEGMENT_DUR, len(segment))
        for i, lead_name in enumerate(LEAD_NAMES):
            plt.title(
                '\n'.join([
                    f'{sample_id} at time {t} after recovery',
                    f'biosppy hr {actual:.2f}',
                    f'model hr {pred:.2f}',
                    f'biosppy lead difference {diff:.2f}',
                ]),
            )
            plt.plot(x, segment[:, i], label=lead_name)


def _recovery_hrs_biosppy(hd5: h5py.File) -> List[Tuple[float, float]]:
    return list(map(_hr_and_diffs_from_segment, _get_segments_for_biosppy(hd5)))


def _path_from_sample_id(sample_id: str) -> str:
    return os.path.join(TENSOR_FOLDER, sample_id + TENSOR_EXT)


def _sample_id_from_hd5(hd5: h5py.File) -> int:
    return int(os.path.basename(hd5.filename).replace(TENSOR_EXT, ''))


def _sample_id_from_path(path: str) -> int:
    return int(os.path.basename(path).replace(TENSOR_EXT, ''))


def _plot_recovery_hrs(path: str):
    num_plots = len(HR_MEASUREMENT_TIMES)
    plt.figure(figsize=(10, 3 * num_plots))
    try:
        with h5py.File(path, 'r') as hd5:
            for i, segment in enumerate(_get_segments_for_biosppy(hd5)):
                plt.subplot(num_plots, 1, i + 1)
                _plot_segment(segment)
            plt.tight_layout()
            plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, f'biosppy_hr_recovery_measurements_{_sample_id_from_hd5(hd5)}.png'))
    except (ValueError, KeyError, OSError) as e:
        logging.debug(f'Plotting failed for {path} with error {e}.')


def df_hr_col(t):
    return f'{t}_hr'


def df_hrr_col(t):
    return f'{t}_hrr'


def df_diff_col(t):
    return f'{t}_diff'


DF_HR_COLS = [df_hr_col(t) for t in HR_MEASUREMENT_TIMES]
DF_DIFF_COLS = [df_diff_col(t) for t in HR_MEASUREMENT_TIMES]


def _recovery_hrs_from_path(path: str):
    sample_id = os.path.basename(path).replace(TENSOR_EXT, '')
    if sample_id.endswith('000'):
        logging.info(f'Processing sample_id {sample_id}.')
    hr_diff = np.full((len(HR_MEASUREMENT_TIMES), 2), np.nan)
    error = None
    try:
        with h5py.File(path, 'r') as hd5:
            hr_diff = np.array(_recovery_hrs_biosppy(hd5))
    except (ValueError, KeyError, OSError) as e:
        error = e
    measures = {'sample_id': sample_id, 'error': error}
    for i, (hr_col, diff_col) in enumerate(zip(DF_HR_COLS, DF_DIFF_COLS)):
        measures[hr_col] = hr_diff[i, 0]
        measures[diff_col] = hr_diff[i, 1]
    return measures


def plot_hr_from_biosppy_summary_stats():
    df = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE)

    # HR summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_HR_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna()
        sns.distplot(x, label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, 'biosppy_hr_recovery_measurements_summary_stats.png'))

    # HR lead diff summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_DIFF_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna().copy()
        sns.distplot(x[x < 5], label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, 'biosppy_hr_diff_recovery_measurements_summary_stats.png'))

    # Random sample of hr trends
    plt.figure(figsize=(15, 7))
    trend_samples = df[DF_HR_COLS].sample(1000).values
    plt.plot(HR_MEASUREMENT_TIMES, (trend_samples - trend_samples[:, :1]).T, alpha=.2, linewidth=1, c='k')
    plt.axhline(0, c='k', linestyle='--')
    plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, 'biosppy_hr_trend_samples.png'))

    # correlation heat map
    plt.figure(figsize=(7, 7))
    sns.heatmap(df[DF_HR_COLS + DF_DIFF_COLS].corr(), annot=True, cbar=False)
    plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, 'biosppy_correlations.png'))
    plt.close()


def plot_pretest_label_summary_stats():
    df = pd.read_csv(PRETEST_LABEL_FILE)

    # HR summary stats
    plt.figure(figsize=(15, 7))
    for col, t in zip(DF_HR_COLS, HR_MEASUREMENT_TIMES):
        x = df[col].dropna()
        sns.distplot(x, label=f' Time = {t}\n mean = {x.mean():.2f}\n std = {x.std():.2f}\n top 5% = {np.quantile(x, .95):.2f}')
    plt.legend()
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'pretest_training_labels_summary_stats.png'))

    # Random sample of hr trends
    plt.figure(figsize=(15, 7))
    trend_samples = df[DF_HR_COLS].sample(1000).values
    plt.plot(HR_MEASUREMENT_TIMES, (trend_samples - trend_samples[:, :1]).T, alpha=.2, linewidth=1, c='k')
    plt.axhline(0, c='k', linestyle='--')
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'pretest_training_labels_hr_trend_samples.png'))

    # correlation heat map
    plt.figure(figsize=(7, 7))
    sns.heatmap(df[DF_HR_COLS].corr(), annot=True, cbar=False)
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'biosppy_correlations.png'))
    plt.close()


def build_hr_biosppy_measurements_csv():
    paths = [os.path.join(TENSOR_FOLDER, p) for p in sorted(os.listdir(TENSOR_FOLDER)) if p.endswith(TENSOR_EXT)]
    logging.info('Plotting 10 random hr measurements from biosppy.')
    for path in np.random.choice(paths, 10):
        _plot_recovery_hrs(path)
    pool = Pool()
    logging.info('Beginning to get hr measurements from biosppy.')
    now = time.time()
    measures = pool.map(_recovery_hrs_from_path, paths)
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting hr measurements from biosppy took {delta_t // 60} minutes at {delta_t / len(paths):.2f}s per path.')
    df.to_csv(BIOSPPY_MEASUREMENTS_FILE, index=False)


def make_pretest_labels():
    biosppy_labels = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE)
    new_df = pd.DataFrame()
    hr_0 = biosppy_labels[df_hr_col(HR_MEASUREMENT_TIMES[0])]
    drop_idx = {'no ecg': biosppy_labels['error'].notnull()}
    new_df['sample_id'] = biosppy_labels['sample_id']
    double_sided_quantile = (1 - PRETEST_QUANTILE_CUTOFF) / 2
    for t in HR_MEASUREMENT_TIMES:
        hr_name = df_hr_col(t)
        hr = biosppy_labels[hr_name]
        new_df[hr_name] = hr
        diff = biosppy_labels[df_diff_col(t)]
        drop_idx[f'diff {t} too high'] = diff > diff.quantile(PRETEST_QUANTILE_CUTOFF)
        drop_idx[f'hr {t} higher than {PRETEST_QUANTILE_CUTOFF:.2%}'] = (hr > hr.quantile(1 - double_sided_quantile)) | (hr < hr.quantile(double_sided_quantile))
        new_df[hr_name] = hr
        if t != 0:
            hrr = hr_0 - hr
            hrr_name = df_hrr_col(t)
            new_df[hrr_name] = hrr
            drop_idx[f'hrr {t} outside center {PRETEST_QUANTILE_CUTOFF:.2%}'] = (hrr > hrr.quantile(1 - double_sided_quantile)) | (hrr < hrr.quantile(double_sided_quantile))
            new_df[hrr_name] = hrr

    logging.info(f'Pretest labels starting at length {len(new_df)}.')
    all_drop = False
    for name, idx in drop_idx.items():
        logging.info(f'Due to filter {name}, dropping {(idx & ~all_drop).sum()} values')
        all_drop |= idx
    new_df = new_df[~all_drop]
    unknown_errors = new_df.isna().any(axis=1)
    logging.info(f'Dropping {unknown_errors.sum()} due to unknown biosppy errors.')
    new_df = new_df[~unknown_errors]  # TODO: why needed?
    assert new_df.notna().all().all()
    logging.info(f'There are {len(new_df)} pretest labels after filtering.')
    new_df.to_csv(PRETEST_LABEL_FILE, index=False)


# hr tmaps
def _hr_file(file_name: str, t: int, hrr=False):
    error = None
    try:
        df = pd.read_csv(file_name, dtype={'sample_id': int})
        df = df.set_index('sample_id')
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        sample_id = _sample_id_from_hd5(hd5)
        try:
            row = df.loc[sample_id]
            hr = row[df_hr_col(t)]
            if hrr:
                peak = row[df_hr_col(0)]
                out = peak - hr
            else:
                out = hr
            return np.array([out])
        except KeyError:
            raise KeyError(f'Sample id not in {file_name} for TensorMap {tm.name}.')
    return tensor_from_file


def _make_hrr_tmap(
        file_name: str,
        normalizer: Normalizer = None,
) -> TensorMap:
    return TensorMap(
            df_hrr_col(HRR_TIME), shape=(1,), metrics=[],
            interpretation=Interpretation.CONTINUOUS,
            tensor_from_file=_hr_file(file_name, HRR_TIME, hrr=True),
            normalization=normalizer,
        )


# build cohort
def tmap_error_detect(tmap: TensorMap) -> TensorMap:
    """Modifies tm so it returns 1 unless previous tensor from file fails"""
    new_tm = copy.deepcopy(tmap)
    new_tm.shape = (1,)
    new_tm.interpretation = Interpretation.CONTINUOUS

    def tff(_: TensorMap, hd5: h5py.File, dependents=None):
        tmap.tensor_from_file(tmap, hd5, dependents)
        return np.array([1.])
    new_tm.tensor_from_file = tff
    return new_tm


def explore_pretest_tmaps():
    hrr_tmap = _make_hrr_tmap(PRETEST_LABEL_FILE)
    tmaps_in = [tmap_error_detect(make_pretest_tmap(1, PRETEST_MODEL_LEADS)), hrr_tmap]
    args = SimpleNamespace(**{
        'explore_export_errors': True,
        'output_folder': OUTPUT_FOLDER,
        'id': PRETEST_EXPLORE_ID,
        'tensor_maps_in': tmaps_in,
        'tensor_maps_out': [],
        'tensors': TENSOR_FOLDER,
        'batch_size': 1,
        'num_workers': 4,
        'cache_size': 0,
        'tsv_style': '',
        'balance_csvs': [],
        'test_ratio': .1,
        'valid_ratio': .05,
        'plot_hist': 'True',
        'training_steps': 1,
        'validation_steps': 1,
    })
    explore(args)


def build_csvs():
    df = pd.read_csv(EXPLORE_RESULT)
    df['sample_id'] = [_sample_id_from_path(p) for p in df['fpath']]
    df_error_cols = [col for col in df.columns if 'error' in col]
    df = df[df[df_error_cols].isnull().all(axis=1)]
    train, valid, test = np.split(
        df.sample(frac=1),
        [int(TRAIN_RATIO * len(df)), int((TRAIN_RATIO + VALID_RATIO) * len(df))]
    )
    train[['sample_id']].to_csv(TRAIN_CSV, index=False)
    valid[['sample_id']].to_csv(VALID_CSV, index=False)
    test[['sample_id']].to_csv(TEST_CSV, index=False)
    return train, test, valid


def _get_hrr_summary_stats(id_csv: str) -> Tuple[float, float]:
    df = pd.read_csv(PRETEST_LABEL_FILE)
    ids = pd.read_csv(id_csv)
    hrr = df[df_hrr_col(HRR_TIME)][df['sample_id'].isin(ids['sample_id'])]
    return hrr.mean(), hrr.std()


# Model training
def make_pretest_model(load_model: bool):
    pretest_tmap = make_pretest_tmap(BIOSPPY_DOWNSAMPLE_RATE, PRETEST_MODEL_LEADS)
    hrr_tmap = _make_hrr_tmap(PRETEST_LABEL_FILE, Standardize(*_get_hrr_summary_stats(TRAIN_CSV)))
    return make_multimodal_multitask_model(
        tensor_maps_in=[pretest_tmap],
        tensor_maps_out=[hrr_tmap],
        activation='swish',
        learning_rate=1e-3,
        bottleneck_type=BottleneckType.FlattenRestructure,
        optimizer='radam',
        dense_layers=[16, 64],
        dropout=0,
        conv_layers=[32],
        dense_blocks=[32, 24, 16],
        conv_type='conv',
        conv_normalize='layer_norm',
        conv_x=[16],
        conv_y=[1],
        conv_z=[1],
        pool_type='max',
        pool_x=2,
        block_size=3,
        model_file=PRETEST_MODEL_PATH if load_model else None,
    )  # TODO: reincorporate hyperopt?


def _train_pretest_model(
        model_id: str,
) -> Tuple[Any, Dict]:
    workers = cpu_count() * 2
    patience = 8
    epochs = 200
    batch_size = 128

    pretest_tmap = make_pretest_tmap(BIOSPPY_DOWNSAMPLE_RATE, PRETEST_MODEL_LEADS)
    hrr_tmap = _make_hrr_tmap(PRETEST_LABEL_FILE, Standardize(*_get_hrr_summary_stats(TRAIN_CSV)))

    # TODO: this could be 5 fold cross val
    train_len = len(pd.read_csv(TRAIN_CSV))
    valid_len = len(pd.read_csv(VALID_CSV))
    training_steps = train_len // batch_size
    validation_steps = valid_len // batch_size

    generate_train, generate_valid, _ = test_train_valid_tensor_generators(
        tensor_maps_in=[pretest_tmap],
        tensor_maps_out=[hrr_tmap],
        tensors=TENSOR_FOLDER,
        batch_size=batch_size,
        num_workers=workers,
        cache_size=1e7,
        balance_csvs=[],
        train_csv=TRAIN_CSV,
        valid_csv=VALID_CSV,
        test_csv=TEST_CSV,
        training_steps=training_steps,
        validation_steps=validation_steps,
    )
    model = make_pretest_model(False)
    try:
        model, history = train_model_from_generators(
            model, generate_train, generate_valid, training_steps, validation_steps, batch_size,
            epochs, patience, OUTPUT_FOLDER, model_id, True, True, return_history=True,
        )
    finally:
        generate_train.kill_workers()
        generate_valid.kill_workers()
        gc.collect()
    return model, history


# Inference
ACTUAL_POSTFIX = '_actual'
PRED_POSTFIX = '_predicted'


def tmap_to_actual_col(tmap: TensorMap):
    return f'{tmap.name}{ACTUAL_POSTFIX}'


def tmap_to_pred_col(tmap: TensorMap, model_id: str):
    return f'{model_id}_{tmap.name}{PRED_POSTFIX}'


def time_to_pred_hr_col(t: int, model_id: str):
    return f'{model_id}_{df_hr_col(t)}{PRED_POSTFIX}'


def time_to_pred_hrr_col(t: int, model_id: str):
    return f'{model_id}_{df_hrr_col(t)}{PRED_POSTFIX}'


def time_to_actual_hr_col(t: int):
    return f'{df_hr_col(t)}{ACTUAL_POSTFIX}'


def time_to_actual_hrr_col(t: int):
    return f'{df_hrr_col(t)}{ACTUAL_POSTFIX}'


def _handle_inference_batch(
        output_name_to_tmap: Dict[str, TensorMap], model: Model, model_id: str, batch,
        visited_paths: Set[str], rows: List[Dict[str, str]],
):
    input_data, output_data, tensor_paths = batch[BATCH_INPUT_INDEX], batch[BATCH_OUTPUT_INDEX], batch[BATCH_PATHS_INDEX]
    # TODO: compare with infer_multi branch
    preds = model.predict(input_data)
    if len(output_name_to_tmap) == 1:
        preds = [preds]
    for pred, out_name in zip(preds, model.output_names):
        tm = output_name_to_tmap[out_name]
        scaled = tm.rescale(pred)
        actual = output_data[tm.output_name()]
        for i, row in enumerate(rows):
            if tensor_paths[i] in visited_paths:
                continue
            visited_paths.add(tensor_paths[i])
            row[tmap_to_pred_col(tm, model_id)] = f'{float(scaled[i]):.3f}'
            row['sample_id'] = _sample_id_from_path(tensor_paths[i])
            if ((tm.sentinel is not None and tm.sentinel == actual[i][0])
                    or np.isnan(actual[i][0])):
                row[tmap_to_actual_col(tm)] = 'NA'
            else:
                row[tmap_to_actual_col(tm)] = f'{float(tm.rescale(actual[i])):.3f}'


def _infer_models(
        models: List[Model], model_ids: List[str], inference_tsv: str,
        input_tmaps: List[TensorMap], output_tmaps: List[TensorMap], transfer: bool = False,
):
    count = 0
    visited_paths = set()
    tensor_paths = [
        os.path.join(TENSOR_FOLDER, tp) for tp in sorted(os.listdir(TENSOR_FOLDER))
        if os.path.splitext(tp)[-1].lower() == TENSOR_EXT
    ]
    no_fail_tmaps_out = [_make_tmap_nan_on_fail(tmap) for tmap in output_tmaps]
    generate_test = None
    try:
        generate_test = TensorGenerator(
            128, input_tmaps, no_fail_tmaps_out, tensor_paths, num_workers=8,
            cache_size=0, keep_paths=True, mixup=0,
        )

        output_name_to_tmap = {tm.output_name(): tm for tm in output_tmaps}
        actual_cols = list(map(tmap_to_actual_col, no_fail_tmaps_out))
        prediction_cols = sum(
            [
                [tmap_to_pred_col(output_name_to_tmap[out_name], m_id) for out_name in m.output_names]
                for m, m_id in zip(models, model_ids)
            ],
            [],
        )
        with open(inference_tsv, mode='w') as inference_file:
            inference_writer = csv.DictWriter(
                inference_file, fieldnames=['sample_id'] + actual_cols + prediction_cols,
                delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL,
            )
            inference_writer.writeheader()
            while True:
                batch = next(generate_test)
                rows = [{} for _ in range(len(batch[BATCH_PATHS_INDEX]))]
                for model, model_id in zip(models, model_ids):
                    _handle_inference_batch(output_name_to_tmap, model, model_id, batch, visited_paths, rows)
                inference_writer.writerows([row for row in rows if row])
                if generate_test.stats_q.qsize() == generate_test.num_workers:
                    generate_test.aggregate_and_print_stats()
                    logging.info(f"Inference on {len(visited_paths)} tensors finished. Inference TSV file at: {inference_tsv}")
                    break
                count += 1
                logging.info(f"Wrote:{count} batches of inference.")
    finally:
        if generate_test:
            generate_test.kill_workers()


def _scatter_plot(ax, truth, prediction, title):
    ax.plot([np.min(truth), np.max(truth)], [np.min(truth), np.max(truth)], linewidth=2)
    ax.plot([np.min(prediction), np.max(prediction)], [np.min(prediction), np.max(prediction)], linewidth=4)
    pearson = np.corrcoef(prediction, truth)[1, 0]  # corrcoef returns full covariance matrix
    big_r_squared = coefficient_of_determination(truth, prediction)
    logging.info(f'{title} - pearson:{pearson:0.3f} r^2:{pearson*pearson:0.3f} R^2:{big_r_squared:0.3f}')
    ax.scatter(prediction, truth, label=f'Pearson:{pearson:0.3f} r^2:{pearson * pearson:0.3f} R^2:{big_r_squared:0.3f}', marker='.', s=1)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Actual')
    ax.set_title(title + '\n')
    ax.legend(loc="lower right")


def _dist_plot(ax, truth, prediction, title):
    ax.set_title(title)
    ax.legend(loc="lower right")
    sns.distplot(prediction, label='Predicted', color='r', ax=ax)
    sns.distplot(truth, label='Truth', color='b', ax=ax)
    ax.legend(loc="upper left")


def _evaluate_model(m_id: str, inference_file: str):
    logging.info(f'Plotting {m_id} model results.')
    inference_results = pd.read_csv(inference_file, sep='\t', dtype={'sample_id': str})
    test_ids = pd.read_csv(TEST_CSV, names=['sample_id'], dtype={'sample_id': str})
    test_results = inference_results.merge(test_ids, on='sample_id')
    figure_folder = os.path.join(FIGURE_FOLDER, f'{m_id}_results')
    os.makedirs(figure_folder, exist_ok=True)
    # negative HRR measurements
    name = time_to_pred_hrr_col(HRR_TIME, m_id)
    col = test_results[name].dropna()
    logging.info(f'HRR_{HRR_TIME} had {(col < 0).mean() * 100:.2f}% negative predictions in hold out data.')
    logging.info(f'HRR_{HRR_TIME} had {(col < -5).mean() * 100:.2f}% predictions < -5 in hold out data.')

    # correlations with actual measurements
    ax_size = 10
    _, ax = plt.subplots(figsize=(ax_size, ax_size))
    pred = test_results[time_to_pred_hrr_col(HRR_TIME, m_id)]
    actual = test_results[time_to_actual_hrr_col(HRR_TIME)]
    not_na = ~np.isnan(pred) & ~np.isnan(actual)
    _scatter_plot(ax, actual[not_na], pred[not_na], f'HRR at recovery time {HRR_TIME}')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, 'model_correlations.png'))
    plt.clf()

    # distributions of predicted and actual measurements
    _, ax = plt.subplots(figsize=(ax_size, ax_size))
    pred = test_results[time_to_pred_hrr_col(HRR_TIME, m_id)]
    actual = test_results[time_to_actual_hrr_col(HRR_TIME)]
    not_na = ~np.isnan(pred) & ~np.isnan(actual)
    _dist_plot(ax, actual[not_na], pred[not_na], f'HRR at recovery time {HRR_TIME}')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, 'distributions.png'))
    plt.clf()


if __name__ == '__main__':
    """Always remakes figures"""
    np.random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    os.makedirs(BIOSPPY_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(PRETEST_LABEL_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(EXPLORE_OUTPUT_FOLDER, exist_ok=True)
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    load_config('INFO', OUTPUT_FOLDER, 'log_' + now_string, USER)

    MAKE_LABELS = False or not os.path.exists(BIOSPPY_MEASUREMENTS_FILE)
    EXPLORE_PRETEST_TMAPS = False or not os.path.exists(EXPLORE_RESULT)
    MAKE_CSVS = False or not all((
        os.path.exists(TRAIN_CSV), os.path.exists(VALID_CSV), os.path.exists(TEST_CSV)
    ))
    TRAIN_PRETEST_MODEL = False or not os.path.exists(PRETEST_MODEL_PATH)
    INFER_PRETEST_MODEL = (
            False
            or not os.path.exists(PRETEST_INFERENCE_FILE) or TRAIN_PRETEST_MODEL
    )

    if MAKE_LABELS:
        logging.info('Making biosppy labels.')
        build_hr_biosppy_measurements_csv()
    plot_hr_from_biosppy_summary_stats()
    plt.close('all')
    make_pretest_labels()
    if EXPLORE_PRETEST_TMAPS:
        explore_pretest_tmaps()
    plot_pretest_label_summary_stats()
    if MAKE_CSVS:
        build_csvs()
    if TRAIN_PRETEST_MODEL:
        _train_pretest_model(PRETEST_MODEL_ID)
    if INFER_PRETEST_MODEL:
        logging.info('Running inference on pretest models.')
        _infer_models(
            models=[make_pretest_model(True)],
            model_ids=[PRETEST_MODEL_ID],
            input_tmaps=[make_pretest_tmap(BIOSPPY_DOWNSAMPLE_RATE, PRETEST_MODEL_LEADS)],
            output_tmaps=[_make_hrr_tmap(PRETEST_LABEL_FILE, Standardize(*_get_hrr_summary_stats(TRAIN_CSV)))],
            inference_tsv=PRETEST_INFERENCE_FILE,
        )
    _evaluate_model(PRETEST_MODEL_ID, PRETEST_INFERENCE_FILE)
# TODO: bonus: bootstrapping for test set full size?
# TODO: augmentations demonstrations
# TODO: does explore result align with PRETEST_LABEL_FILE?
# TODO: table explaining filtering
# TODO: explore necessary?
# TODO: inference only works for one model
# TODO: does inference infer all results??
