import os
import ray
import time
import h5py
import blosc
import biosppy
import seaborn as sns
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Union, Tuple, Dict, Any, Callable
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split
from multiprocessing import Pool, cpu_count
from matplotlib import cm
import matplotlib.pyplot as plt
from collections import namedtuple
import datetime
import gc


def set_no_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu,
            True,
        )  # do not allocate all memory right away


set_no_gpu_growth()


from ml4cvd.defines import TENSOR_EXT, MODEL_EXT
from ml4cvd.logger import load_config
from ml4cvd.TensorMap import TensorMap, Interpretation, no_nans
from ml4cvd.normalizer import Standardize
from ml4cvd.tensor_generators import test_train_valid_tensor_generators
from ml4cvd.models import train_model_from_generators, make_multimodal_multitask_model, BottleneckType
from ml4cvd.recipes import _infer_models
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
PHYSIOLOGICAL_HR_RANGE = 40, 220

TENSOR_FOLDER = '/mnt/disks/ecg-bike-tensors-2/2021-04-01/'
USER = 'ndiamant'
LEFT_UKB = f'/home/{USER}/w7089_20210201.csv'
COVARIATES = f'/home/{USER}/biosppy_hr_recovery_measurements_with_covariates.csv'
OUTPUT_FOLDER = f'/home/{USER}/ml/hrr_results_04-24'
TRAIN_CSV_NAME = 'train_ids.csv'
VALID_CSV_NAME = 'valid_ids.csv'
TEST_CSV_NAME = 'test_ids.csv'

BIOSPPY_MEASUREMENTS_FILE = os.path.join(OUTPUT_FOLDER, 'biosppy_hr_recovery_measurements.csv')
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, 'figures')
BIOSPPY_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'biosppy')
AUGMENTATION_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'augmentations')

PRETEST_ECG_SUMMARY_STATS_CSV = os.path.join(OUTPUT_FOLDER, 'pretest_ecg_summary_stats.csv')

PRETEST_LABEL_FIGURE_FOLDER = os.path.join(FIGURE_FOLDER, 'pretest_labels')
PRETEST_LABEL_FILE = os.path.join(OUTPUT_FOLDER, f'hr_pretest_training_data.csv')
PRETEST_TRAINING_DUR = 10  # number of seconds of pretest ECG used for prediction
VALIDATION_SPLIT = .1

DROPOUT = True
BATCH_NORM = True
AUG_RATE = .5
OVERWRITE_MODELS = False

PRETEST_MODEL_LEADS = [0]
SEED = 217
PRETEST_INFERENCE_NAME = 'pretest_model_inference.tsv'
K_SPLIT = 5


def _get_tensor_at_first_instance(hd5: h5py.File, path: str):
    key = min(hd5[path])
    return hd5["/".join([path, key])][()]


# Tensor from file helpers
def _check_phase_full_len(hd5: h5py.File, phase: str):
    phase_len = _get_tensor_at_first_instance(hd5, f'{phase}_duration')
    valid = True
    if phase == 'Pretest':
        valid &= phase_len == PRETEST_DUR
    elif phase == 'Exercise':
        valid &= phase_len == EXERCISE_DUR
    elif phase == 'Rest':
        valid &= phase_len == RECOVERY_DUR
    else:
        raise ValueError(f'Phase {phase} is not a valid phase.')
    if not valid:
        raise ValueError(f'{phase} phase is not full length.')


def read_compressed(data_set: h5py.Dataset):
    shape = data_set.attrs['shape']
    return np.frombuffer(blosc.decompress(data_set[()]), dtype=np.int16).reshape(shape)


def _get_bike_ecg(hd5: h5py.File, start: int, stop: int, leads: Union[List[int], slice]):
    path_prefix = "full_disclosure"
    key = min(hd5[path_prefix])  # first instance
    ecg_dataset = hd5["/".join([path_prefix, key])]
    tensor = read_compressed(ecg_dataset)[start: stop, leads]
    return tensor


def _get_downsampled_bike_ecg(length: float, hd5: h5py.File, start: int, rate: float, leads: Union[List[int], slice]):
    length = int(length * rate)
    ecg = _get_bike_ecg(hd5, start, start + length, leads)
    ecg = _downsample_ecg(ecg, rate)
    return ecg


def _make_pretest_ecg_tff(downsample_rate: float, leads: Union[List[int], slice], random_start=True):
    def tff(tm: TensorMap, hd5: h5py.File, dependents=None):
        _check_phase_full_len(hd5, 'Pretest')
        start = np.random.randint(0, SAMPLING_RATE * PRETEST_DUR - tm.shape[0] * downsample_rate) if random_start else 0
        return _get_downsampled_bike_ecg(tm.shape[0], hd5, start, downsample_rate, leads)
    return tff


def _get_trace_recovery_start(hd5: h5py.File) -> int:
    _check_phase_full_len(hd5, 'Rest')
    _check_phase_full_len(hd5, 'Pretest')
    pretest_dur = _get_tensor_at_first_instance(hd5, 'Pretest_duration')
    exercise_dur = _get_tensor_at_first_instance(hd5, 'Exercise_duration')
    return int(SAMPLING_RATE * (pretest_dur + exercise_dur - HR_SEGMENT_DUR / 2 - TREND_TRACE_DUR_DIFF))


PRETEST_MEAN_COL = 'pretest_mean'
PRETEST_STD_COL = 'pretest_std'


def _pretest_mean_std(sample_id: int) -> Dict[str, float]:
    if str(sample_id).endswith('000'):
        logging.info(f'Processing sample_id {sample_id}.')
    with h5py.File(_path_from_sample_id(str(sample_id)), 'r') as hd5:
        pretest = _get_bike_ecg(hd5, 0, PRETEST_DUR * SAMPLING_RATE, PRETEST_MODEL_LEADS)
        return {'sample_id': sample_id, PRETEST_MEAN_COL: pretest.mean(), PRETEST_STD_COL: pretest.std()}


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
    noise_frac = np.random.rand() * .2
    return ecg + noise_frac * ecg.std(axis=0) * np.random.randn(*ecg.shape)


def _apply_aug_rate(augmentation: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    return lambda a: augmentation(a) if np.random.rand() < AUG_RATE else a


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
    """
    Gets pretest, peak, and recovery segments
    """
    ecg = _get_bike_ecg(hd5, 0, -1, leads=[0, 1, 2])
    _check_phase_full_len(hd5, "Pretest")
    _check_phase_full_len(hd5, "Rest")
    recovery_start_idx = -60 * SAMPLING_RATE
    peak_start = recovery_start_idx - HR_SEGMENT_DUR * SAMPLING_RATE // 2
    peak_end = peak_start + HR_SEGMENT_DUR * SAMPLING_RATE
    recovery_start = HRR_TIME * SAMPLING_RATE + peak_start
    recovery_end = HRR_TIME * SAMPLING_RATE + peak_end
    return (
        ecg[:PRETEST_DUR * SAMPLING_RATE][::BIOSPPY_DOWNSAMPLE_RATE],
        ecg[peak_start: peak_end][::BIOSPPY_DOWNSAMPLE_RATE],
        ecg[recovery_start: recovery_end][::BIOSPPY_DOWNSAMPLE_RATE],
    )


def _hr_and_diffs_from_segment(segment: np.ndarray) -> Tuple[float, float]:
    hr_per_lead = [_get_biosppy_hr(segment[:, i]) for i in range(segment.shape[-1])]
    if np.any(np.isnan(hr_per_lead)):
        raise ValueError("Biosppy returned no HR measurements")
    max_diff = max(map(lambda pair: abs(pair[0] - pair[1]), combinations(hr_per_lead, 2)))
    return float(np.median(hr_per_lead)), max_diff


def _plot_segment(segment: np.ndarray, title=None):
    hr, max_diff = _hr_and_diffs_from_segment(segment)
    t = np.linspace(0, HR_SEGMENT_DUR, len(segment))
    for i, lead_name in enumerate(LEAD_NAMES):
        plt.plot(t, segment[:, i], label=lead_name)
    plt.xlabel('Time (s)')
    plt.legend()
    extra_info = f'hr: {hr:.2f}, max hr difference between leads: {max_diff:.2f}'
    plt.title(title + "\n" + extra_info if title else extra_info)


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


def _hrs_biosppy(hd5: h5py.File) -> List[Tuple[float, float]]:
    """
    returns HR and max HR diff for pretest, peak, and recovery
    """
    return list(map(_hr_and_diffs_from_segment, _get_segments_for_biosppy(hd5)))


def _path_from_sample_id(sample_id: str) -> str:
    return os.path.join(TENSOR_FOLDER, sample_id + TENSOR_EXT)


def _sample_id_from_hd5(hd5: h5py.File) -> int:
    return int(os.path.basename(hd5.filename).replace(TENSOR_EXT, ''))


def _sample_id_from_path(path: str) -> int:
    return int(os.path.basename(path).replace(TENSOR_EXT, ''))


def _plot_recovery_hrs(path: str, title: str = "biosppy_measurements"):
    num_plots = len(HR_MEASUREMENT_TIMES) + 1
    plt.figure(figsize=(10, 3 * num_plots))
    plt.suptitle(title)
    with h5py.File(path, 'r') as hd5:
        for i, (name, segment) in enumerate(zip(
            ["pretest", "peak", "recovery"],
            _get_segments_for_biosppy(hd5)
        )):
            plt.subplot(num_plots, 1, i + 1).set_title(name)
            _plot_segment(segment, title=name)
        plt.tight_layout()
        plt.savefig(os.path.join(BIOSPPY_FIGURE_FOLDER, f'{title}_{_sample_id_from_hd5(hd5)}.png'))


def df_hr_col(t):
    return f'{t}_hr'


def df_hrr_col(t):
    return f'{t}_hrr'


def df_diff_col(t):
    return f'{t}_hr_diff'


DF_HR_COLS = [df_hr_col(t) for t in HR_MEASUREMENT_TIMES]
DF_DIFF_COLS = [df_diff_col(t) for t in HR_MEASUREMENT_TIMES]
PRETEST_HR_COL = "pretest_hr"
PRETEST_DIFF_COL = "pretest_hr_diff"


def _recovery_hrs_from_path(path: str):
    sample_id = os.path.basename(path).replace(TENSOR_EXT, '')
    hr_diff = np.full((1 + len(HR_MEASUREMENT_TIMES), 2), np.nan)
    error = None
    instance = None
    protocol = None
    try:
        with h5py.File(path, 'r') as hd5:
            hr_diff = np.array(_hrs_biosppy(hd5))
            instance = min(hd5["full_disclosure"])
            protocol = hd5[f"protocol/{instance}"][()]
    except (ValueError, KeyError, OSError) as e:
        error = e
    measures = {'sample_id': sample_id, 'error': error, 'instance': instance, 'protocol': protocol}
    for i, (hr_col, diff_col) in enumerate(
        zip([PRETEST_HR_COL] + DF_HR_COLS, [PRETEST_DIFF_COL] + DF_DIFF_COLS)
    ):
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
    trend_samples = df[DF_HR_COLS].sample(1000, replace=True).values
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

    # HRR
    plt.figure(figsize=(15, 7))
    hrr_col = df_hrr_col(HRR_TIME)
    hrr = df[hrr_col]
    plt.title(f"HRR \n mean = {hrr.mean():.2f}\n std = {hrr.std():.2f}\n top 5% = {np.quantile(hrr, .95):.2f}")
    sns.distplot(hrr)
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'pretest_training_HRR_summary_stats.png'))

    # Random sample of hr trends
    plt.figure(figsize=(15, 7))
    trend_samples = df[DF_HR_COLS].sample(1000, replace=True).values
    plt.plot(HR_MEASUREMENT_TIMES, (trend_samples - trend_samples[:, :1]).T, alpha=.2, linewidth=1, c='k')
    plt.axhline(0, c='k', linestyle='--')
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'pretest_training_labels_hr_trend_samples.png'))

    # correlation heat map
    plt.figure(figsize=(7, 7))
    sns.heatmap(df[DF_HR_COLS + [PRETEST_HR_COL, hrr_col]].corr(), annot=True, cbar=False)
    plt.savefig(os.path.join(PRETEST_LABEL_FIGURE_FOLDER, 'biosppy_correlations.png'))
    plt.close()


def build_hr_biosppy_measurements_csv():
    paths = [os.path.join(TENSOR_FOLDER, p) for p in sorted(os.listdir(TENSOR_FOLDER)) if p.endswith(TENSOR_EXT)]
    logging.info('Plotting 10 random hr measurements from biosppy.')
    for path in np.random.choice(paths, 10):
        _plot_recovery_hrs(path)
    logging.info('Beginning to get hr measurements from biosppy.')
    now = time.time()
    measures = []
    with Pool() as pool:
        for i, measure in enumerate(pool.imap_unordered(_recovery_hrs_from_path, paths)):
            measures.append(measure)
            if i % 100 == 0:
                logging.info(f"Biosppy HR measures {(i + 1) / len(paths):.1%} done.")
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting hr measurements from biosppy took {delta_t // 60} minutes at {delta_t / len(paths):.2f}s per path.')
    df.to_csv(BIOSPPY_MEASUREMENTS_FILE, index=False)


def build_pretest_summary_stats_csv(sample_ids: List[int]) -> pd.DataFrame:
    pool = Pool()
    logging.info('Beginning to get pretest ecg means and stds.')
    now = time.time()
    measures = pool.map(_pretest_mean_std, sample_ids)
    df = pd.DataFrame(measures)
    delta_t = time.time() - now
    logging.info(f'Getting pretest ecg means and stds took {delta_t // 60} minutes at {delta_t / len(sample_ids):.2f}s per path.')
    return df


def _filter_biosppy(name: str, hr_col: pd.Series, diff_col: pd.Series) -> Dict[str, pd.Series]:
    """
    Picks rows to remove according to non-physiological HRs and diffs too high
    :param hr_col: series of measured HRs
    :param diff_col: series of max hr difference measured across the 3 leads
    :param seconds_in_segment: length of ECG segment used to get HRs
    :return: filter name -> rows to remove
    """
    T = 10  # segment length to allow at most 1 beat difference
    out = {
        f"{name} HR < {PHYSIOLOGICAL_HR_RANGE[0]}": hr_col < PHYSIOLOGICAL_HR_RANGE[0],
        f"{name} HR > {PHYSIOLOGICAL_HR_RANGE[1]}": hr_col > PHYSIOLOGICAL_HR_RANGE[1],
        # diff_col beats / min * 1 min / 60 s = beats / s
        # beats * seconds_in_segment / Ts = beat diff per Ts
        f"{name} more than 1 beat diff per {T}s across leads": (diff_col / 60) * T > 1,
    }
    return out


def make_pretest_labels(make_ecg_summary_stats: bool):
    biosppy_labels = pd.read_csv(BIOSPPY_MEASUREMENTS_FILE)
    new_df = pd.DataFrame()
    logging.info(f'Biosppy error counts:\n{biosppy_labels["error"].value_counts()}')
    left_ukb = set(pd.read_csv(LEFT_UKB)["sample_id"])
    drop_idx = {
        'Left UKB': biosppy_labels["sample_id"].isin(left_ukb),
        'Biosppy error': biosppy_labels['error'].notnull(),
        'protocol in F30, M40, or R': biosppy_labels['protocol'].isin({"F30", "M40", "R"}),
    }
    new_df['sample_id'] = biosppy_labels['sample_id']

    # fill in new_df columns and track errors
    # pretest
    pretest_hr = biosppy_labels[PRETEST_HR_COL]
    drop_idx = {**drop_idx, **_filter_biosppy("Pretest", pretest_hr, biosppy_labels[PRETEST_DIFF_COL])}
    new_df[PRETEST_HR_COL] = pretest_hr
    # peak hr
    hr_0 = biosppy_labels[df_hr_col(0)]
    new_df[df_hr_col(0)] = hr_0
    drop_idx = {**drop_idx, **_filter_biosppy("Peak", hr_0, biosppy_labels[df_diff_col(0)])}
    # recovery and hrr
    recovery_hr = biosppy_labels[df_hr_col(HRR_TIME)]
    new_df[df_hr_col(HRR_TIME)] = recovery_hr
    hrr_name = df_hrr_col(HRR_TIME)
    drop_idx = {**drop_idx, **_filter_biosppy("Recovery", recovery_hr, biosppy_labels[df_diff_col(HRR_TIME)])}
    new_df[hrr_name] = hr_0 - recovery_hr
    drop_idx['HRR negative'] = new_df[hrr_name] < 0

    logging.info(f'Pretest labels starting at length {len(new_df)}.')
    all_drop = False
    for name, idx in drop_idx.items():
        this_drop_idx = idx & ~all_drop
        logging.info(f'Due to filter {name}, dropping {(this_drop_idx).sum()} ({(idx & ~all_drop).mean():.2%}) values')
        if this_drop_idx.any():  # plot some example traces that get filtered out
            sample_ids = new_df[this_drop_idx]["sample_id"].sample(3)
            try:
                for i in range(3):
                    demo_path = _path_from_sample_id(str(sample_ids.iloc[i]))
                    _plot_recovery_hrs(demo_path, f"Dropped by filter {name}")
            except ValueError as e:
                pass
        all_drop |= idx
    new_df = new_df[~all_drop]
    covariates = pd.read_csv(COVARIATES).dropna(
        subset=['age', 'male', 'race', 'bmi', 'systolic_bp', 'current_smoker', 'instance_date'],
    )
    no_cov = ~new_df["sample_id"].isin(covariates["sample_id"])
    logging.info(f"Dropping {no_cov.sum()} due to missing covariates.")
    new_df = new_df[~no_cov]

    # plot some example traces that don't get filtered out
    hrr = new_df[hrr_name]
    for quantile in (.02, .51, 1):
        lo, hi = np.quantile(hrr.dropna(), [quantile - .02, quantile])
        sample_ids = (new_df["sample_id"][hrr.between(lo, hi)]).sample(3)
        for i in range(3):
            demo_path = _path_from_sample_id(str(sample_ids.iloc[i]))
            _plot_recovery_hrs(demo_path, f"HRR in quantile ({quantile - .02:.3f}, {quantile:.3f})")

    # make sure no errors missed
    assert new_df.notna().all().all()
    logging.info(f'There are {len(new_df)} pretest labels after filtering hr measures.')

    # make sure pretest ECG tmap works
    if make_ecg_summary_stats:
        pretest_df = build_pretest_summary_stats_csv(new_df['sample_id'])
        pretest_df.to_csv(PRETEST_ECG_SUMMARY_STATS_CSV, index=False)
    else:
        pretest_df = pd.read_csv(PRETEST_ECG_SUMMARY_STATS_CSV)
    new_df = new_df.merge(pretest_df, on='sample_id')
    logging.info(f'There are {len(new_df)} pretest labels after merging with pretest tmap.')

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


def split_folder_name(split_idx: int) -> str:
    return os.path.join(OUTPUT_FOLDER, f'split_{split_idx}')


def _split_train_name(split_idx: int) -> str:
    return os.path.join(split_folder_name(split_idx), TRAIN_CSV_NAME)


def _split_valid_name(split_idx: int) -> str:
    return os.path.join(split_folder_name(split_idx), VALID_CSV_NAME)


def _split_test_name(split_idx: int) -> str:
    return os.path.join(split_folder_name(split_idx), TEST_CSV_NAME)


# build cohort
def build_csvs():
    sample_ids = pd.read_csv(PRETEST_LABEL_FILE)['sample_id']
    split_ids = _split_ids(ids=sample_ids, n_split=K_SPLIT, validation_frac=.1)
    for i, (train_ids, valid_ids, test_ids) in enumerate(split_ids):
        pd.DataFrame({'sample_id': train_ids}).to_csv(_split_train_name(i), index=False)
        pd.DataFrame({'sample_id': valid_ids}).to_csv(_split_valid_name(i), index=False)
        pd.DataFrame({'sample_id': test_ids}).to_csv(_split_test_name(i), index=False)


def _get_hrr_summary_stats(id_csv: str) -> Tuple[float, float]:
    df = pd.read_csv(PRETEST_LABEL_FILE)
    ids = pd.read_csv(id_csv)
    hrr = df[df_hrr_col(HRR_TIME)][df['sample_id'].isin(ids['sample_id'])]
    return hrr.mean(), hrr.std()


ModelSetting = namedtuple('ModelSetting', ['model_id', 'downsample_rate', 'augmentations', 'shift', 'display_name'])


AUGMENTATIONS = [_warp_ecg, _random_crop_ecg, _rand_add_noise]
MODEL_SETTINGS = [
    ModelSetting(**{'model_id': 'baseline_model', 'downsample_rate': 1, 'augmentations': [], 'shift': False, 'display_name': 'Baseline CNN'}),
    ModelSetting(**{'model_id': 'shift', 'downsample_rate': 1, 'augmentations': [], 'shift': True, 'display_name': 'Random pretest selection'}),
    ModelSetting(**{'model_id': 'shift_augment', 'downsample_rate': 1, 'augmentations': AUGMENTATIONS, 'shift': True, 'display_name': 'Random pretest selection and augmentation'}),
]


# Augmentation demonstrations
Augmentation = Callable[[np.ndarray], np.ndarray]


def _demo_augmentations(hd5_path: str, setting: ModelSetting):
    tmap = _make_ecg_tmap(setting, 0)
    num_samples = 5
    ax_size = 10
    t = np.linspace(0, PRETEST_TRAINING_DUR, tmap.shape[0])
    with h5py.File(hd5_path, 'r') as hd5:
        ecg = tmap.tensor_from_file(tmap, hd5)
        fig, axes = plt.subplots(
            nrows=num_samples, ncols=1, figsize=(ax_size * 2, num_samples * ax_size), sharex='all',
        )
        orig = tmap.postprocess_tensor(ecg, augment=False, hd5=hd5)[:, 0]
        axes[0].set_title(f'Augmentation Samples for model {setting.model_id}')
        for ax in axes:
            ax.plot(t, orig, c='k', label='Original ECG')
            ax.plot(
                t, tmap.postprocess_tensor(ecg, augment=True, hd5=hd5)[:, 0],
                c='r', alpha=.5, label='Augmented ECG',
            )
            ax.legend()
    plt.savefig(os.path.join(AUGMENTATION_FIGURE_FOLDER, f'{setting.model_id}_{_sample_id_from_path(hd5_path)}.png'))


# Model training
def _make_ecg_tmap(setting: ModelSetting, split_idx: int) -> TensorMap:
    normalizer = Standardize(0, 200)  # this converts to units of mV
    augmentations = [_apply_aug_rate(aug) for aug in setting.augmentations]
    return TensorMap(
        f'pretest_ecg_downsample_{setting.downsample_rate}',
        shape=(int(PRETEST_TRAINING_DUR * SAMPLING_RATE // setting.downsample_rate), len(PRETEST_MODEL_LEADS)),
        interpretation=Interpretation.CONTINUOUS,
        validator=no_nans, normalization=normalizer,
        tensor_from_file=_make_pretest_ecg_tff(setting.downsample_rate, PRETEST_MODEL_LEADS, random_start=setting.shift),
        cacheable=False, augmentations=augmentations,
    )


def _make_hrr_tmap(split_idx: int) -> TensorMap:
    normalizer = Standardize(*_get_hrr_summary_stats(_split_train_name(split_idx)))
    return TensorMap(
        df_hrr_col(HRR_TIME), shape=(1,), metrics=[],
        interpretation=Interpretation.CONTINUOUS,
        tensor_from_file=_hr_file(PRETEST_LABEL_FILE, HRR_TIME, hrr=True),
        normalization=normalizer,
    )


def make_pretest_model(setting: ModelSetting, split_idx: int, load_model: bool):
    pretest_tmap = _make_ecg_tmap(setting, split_idx)
    hrr_tmap = _make_hrr_tmap(split_idx)
    model_path = pretest_model_file(split_idx, setting.model_id)
    return make_multimodal_multitask_model(
        tensor_maps_in=[pretest_tmap],
        tensor_maps_out=[hrr_tmap],
        activation='swish',
        learning_rate=1e-3,
        bottleneck_type=BottleneckType.GlobalAveragePoolStructured,
        optimizer='adam',
        dense_layers=[64],
        conv_layers=[32],
        dense_blocks=[16, 24, 32],
        conv_type='conv',
        conv_normalize='batch_norm' if BATCH_NORM else None,
        conv_x=[16],
        conv_y=[1],
        conv_z=[1],
        pool_type='max',
        pool_x=2,
        block_size=3,
        model_file=model_path if load_model else None,
        conv_regularize='spatial_dropout' if DROPOUT else None,
        conv_regularize_rate=.1 if DROPOUT else 0,
        dense_regularize='dropout',
        dense_regularize_rate=0.5,
    )


def _split_ids(ids: np.ndarray, n_split: int, validation_frac: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """output is [(train_ids, valid_ids, test_ids)]"""
    validation_frac = validation_frac * n_split / (n_split - 1)
    kf = KFold(n_splits=n_split, random_state=SEED, shuffle=True)
    out = []
    for train_idx, test_idx in kf.split(ids):
        train, valid = train_test_split(ids[train_idx], test_size=validation_frac)
        out.append((train, valid, ids[test_idx]))
    return out


def pretest_model_file(split_idx: int, model_id: str) -> str:
    return os.path.join(split_folder_name(split_idx), model_id, model_id + MODEL_EXT)


def history_tsv(split_idx: int, model_id: str) -> str:
    return os.path.join(split_folder_name(split_idx), model_id, 'history.tsv')


@ray.remote(num_cpus=8, num_gpus=1)
def _train_pretest_model(
        setting: ModelSetting, split_idx: int,
) -> Tuple[Any, Dict]:
    import tensorflow as tf  # necessary for ray
    set_no_gpu_growth()

    workers = 8
    patience = 20
    epochs = 100
    batch_size = 128

    train_csv = _split_train_name(split_idx)
    valid_csv = _split_valid_name(split_idx)
    test_csv = _split_test_name(split_idx)

    pretest_tmap = _make_ecg_tmap(setting, split_idx)
    hrr_tmap = _make_hrr_tmap(split_idx)

    train_len = len(pd.read_csv(train_csv))
    valid_len = len(pd.read_csv(valid_csv))
    training_steps = train_len // batch_size
    validation_steps = valid_len // batch_size * (2 if setting.shift else 1)

    generate_train, generate_valid, _ = test_train_valid_tensor_generators(
        tensor_maps_in=[pretest_tmap],
        tensor_maps_out=[hrr_tmap],
        tensors=TENSOR_FOLDER,
        batch_size=batch_size,
        num_workers=workers,
        cache_size=1e7,
        balance_csvs=[],
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
        training_steps=training_steps,
        validation_steps=validation_steps,
    )

    model = make_pretest_model(setting, split_idx, False)
    logging.info(f'Beginning training with {training_steps} training steps and {validation_steps} validation steps.')
    try:
        model, history = train_model_from_generators(
            model, generate_train, generate_valid, training_steps, validation_steps, batch_size,
            epochs, patience, split_folder_name(split_idx), setting.model_id, True, True, return_history=True,
        )
        history_df = pd.DataFrame(history.history)
        history_df['model_id'] = setting.model_id
        history_df['split_idx'] = split_idx
        history_df.to_csv(history_tsv(split_idx, setting.model_id), sep='\t', index=False)
    finally:
        generate_train.kill_workers()
        generate_valid.kill_workers()
        del model
        gc.collect()
    time.sleep(2)


# Inference
ACTUAL_POSTFIX = '_actual'
PRED_POSTFIX = '_prediction'


def _inference_file(split_idx: int) -> str:
    return os.path.join(split_folder_name(split_idx), PRETEST_INFERENCE_NAME)


def tmap_to_actual_col(tmap: TensorMap):
    return f'{tmap.name}{ACTUAL_POSTFIX}'


def tmap_to_pred_col(tmap: TensorMap, model_id: str):
    return f'{tmap.name}_{model_id}{PRED_POSTFIX}'


def time_to_pred_hr_col(t: int, model_id: str):
    return f'{df_hr_col(t)}_{model_id}{PRED_POSTFIX}'


def time_to_pred_hrr_col(t: int, model_id: str):
    return f'{df_hrr_col(t)}_{model_id}{PRED_POSTFIX}'


def time_to_actual_hr_col(t: int):
    return f'{df_hr_col(t)}{ACTUAL_POSTFIX}'


def time_to_actual_hrr_col(t: int):
    return f'{df_hrr_col(t)}{ACTUAL_POSTFIX}'


def _infer_models_split_idx(split_idx: int):
    tensor_paths = [
        _path_from_sample_id(str(sample_id)) for
        sample_id in pd.read_csv(_split_test_name(split_idx))['sample_id']
    ]
    models = [make_pretest_model(setting, split_idx, True) for setting in MODEL_SETTINGS]
    model_ids = [setting.model_id for setting in MODEL_SETTINGS]
    tmaps_in = [_make_ecg_tmap(setting, split_idx) for setting in MODEL_SETTINGS]
    tmaps_out = [_make_hrr_tmap(split_idx)]
    inference_tsv = _inference_file(split_idx)
    if os.path.exists(inference_tsv):
        logging.info(f"SKIPPING inference on split {split_idx} because the inference file exists.")
    _infer_models(
        models=models,
        model_ids=model_ids,
        tensor_maps_in=tmaps_in,
        tensor_maps_out=tmaps_out,
        inference_tsv=inference_tsv, num_workers=8, batch_size=128, tensor_paths=tensor_paths,
    )


# result plotting
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


def _evaluate_models():
    inference_dfs = []
    for i in range(K_SPLIT):
        inference_df = pd.read_csv(_inference_file(i), sep='\t')
        inference_df['split_idx'] = i
        inference_dfs.append(inference_df)
    inference_df = pd.concat(inference_dfs)
    inference_df.to_csv(os.path.join(OUTPUT_FOLDER, PRETEST_INFERENCE_NAME), sep='\t', index=False)

    R2_dfs = []
    ax_size = 10
    figure_folder = os.path.join(FIGURE_FOLDER, f'model_results')
    os.makedirs(figure_folder, exist_ok=True)
    for setting in MODEL_SETTINGS:
        m_id = setting.model_id
        _, ax = plt.subplots(figsize=(ax_size, ax_size))
        pred = inference_df[time_to_pred_hrr_col(HRR_TIME, m_id)]
        actual = inference_df[time_to_actual_hrr_col(HRR_TIME)]
        _scatter_plot(ax, actual, pred, f'HRR at recovery time {HRR_TIME}')
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, f'{m_id}_model_correlations.png'))
        plt.clf()

        # distributions of predicted and actual measurements
        _, ax = plt.subplots(figsize=(ax_size, ax_size))
        _dist_plot(ax, actual, pred, f'HRR at recovery time {HRR_TIME}')
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, f'{m_id}_distributions.png'))
        plt.clf()

        R2s = [
            coefficient_of_determination(
                actual[inference_df['split_idx'] == i], pred[inference_df['split_idx'] == i],
            ) for i in range(K_SPLIT)
        ]
        R2_df = pd.DataFrame({'R2': R2s})
        R2_df['model'] = m_id
        R2_dfs.append(R2_df)
        plt.close('all')

    R2_df = pd.concat(R2_dfs)
    plt.figure(figsize=(ax_size, ax_size))
    sns.boxplot(x='model', y='R2', data=R2_df)
    plt.savefig(os.path.join(figure_folder, f'model_performance_comparison.png'))
    plt.clf()

    logging.info('Beginning bootstrap performance evaluation.')
    num_bootstraps = 10000
    model_ids = list(R2_df['model'].unique())
    R2_df = bootstrap_compare_models(model_ids, inference_df, num_bootstraps=num_bootstraps, bootstrap_frac=1)
    cmap = cm.get_cmap('rainbow')
    final_model = MODEL_SETTINGS[-1].model_id
    final_R2 = R2_df['R2'][R2_df['model'] == final_model].values
    plt.figure(figsize=(2 * ax_size, ax_size))
    for i, setting in enumerate(MODEL_SETTINGS[:-1]):
        m_id = setting.model_id
        R2 = R2_df['R2'][R2_df['model'] == m_id].values
        diff = R2 - final_R2
        color = cmap(i / (K_SPLIT - 1))
        sns.distplot(diff, color=color)
        plt.axvline(diff.mean(), color=color, label=f'{setting.display_name} ({diff.mean():.3f}, {diff.std():.3f})', linestyle='--')
    plt.title(f'Bootstrapped Performance Comparison\nsamples = {num_bootstraps}')
    final_name = MODEL_SETTINGS[-1].display_name
    plt.axvline(0, c='k', linestyle='--', label=f'{final_name}')
    plt.xlabel(f'$R^2$ - {final_name} $R^2$')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(figure_folder, f'bootstrap_diff_distributions.png'))
    plt.close('all')


def plot_training_curves():
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), sharey=True)
    for setting in MODEL_SETTINGS:
        model_id = setting.model_id
        model_losses = []
        model_val_losses = []
        for split_idx in range(K_SPLIT):
            history = pd.read_csv(history_tsv(split_idx, model_id), sep='\t')
            model_losses.append(history['loss'])
            model_val_losses.append(history['val_loss'])
        max_len = max(map(len, model_losses))
        loss_array = np.full((K_SPLIT, max_len), np.nan)
        val_loss_array = np.full((K_SPLIT, max_len), np.nan)
        for loss, val_loss, split_idx in zip(model_losses, model_val_losses, range(K_SPLIT)):
            loss_array[split_idx, :len(loss)] = loss
            val_loss_array[split_idx, :len(loss)] = val_loss

        epoch = list(range(max_len))
        ax1.plot(epoch, loss_array.mean(axis=0), label=f'{setting.model_id} mean loss')
        ax1.fill_between(
            epoch, loss_array.min(axis=0), loss_array.max(axis=0),
            label=f'{setting.model_id} min and max loss', alpha=.2,
        )
        ax2.plot(epoch, val_loss_array.mean(axis=0), label=f'{setting.model_id} mean validation loss')
        ax2.fill_between(
            epoch, val_loss_array.min(axis=0), val_loss_array.max(axis=0),
            label=f'{setting.model_id} min and max validation loss', alpha=.2,
        )
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    plt.tight_layout()
    figure_folder = os.path.join(FIGURE_FOLDER, f'model_results')
    plt.savefig(os.path.join(figure_folder, f'training_curves.png'))


def bootstrap_compare_models(
        model_ids: List[str], inference_result: pd.DataFrame,
        num_bootstraps: int = 100, bootstrap_frac: float = 1.,
) -> pd.DataFrame:
    performance = {'model': [], 'R2': []}
    actual_col = time_to_actual_hrr_col(HRR_TIME)
    pred_cols = {m_id: time_to_pred_hrr_col(HRR_TIME, m_id) for m_id in model_ids}
    for i in range(num_bootstraps):
        df = inference_result.sample(frac=bootstrap_frac, replace=True)
        for m_id, pred_col in pred_cols.items():
            pred = df[pred_col]
            R2 = coefficient_of_determination(df[actual_col], pred)
            performance['model'].append(m_id)
            performance['R2'].append(R2)
        print(f'Bootstrap - {(i + 1) / num_bootstraps:.2%}', end='\r')
    print()
    return pd.DataFrame(performance)


if __name__ == '__main__':
    """Always remakes figures"""
    np.random.seed(SEED)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    os.makedirs(BIOSPPY_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(PRETEST_LABEL_FIGURE_FOLDER, exist_ok=True)
    os.makedirs(AUGMENTATION_FIGURE_FOLDER, exist_ok=True)
    now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    load_config('INFO', OUTPUT_FOLDER, 'log_' + now_string, USER)

    # pick tasks
    MAKE_LABELS = False or not os.path.exists(BIOSPPY_MEASUREMENTS_FILE)
    for i in range(K_SPLIT):
        os.makedirs(split_folder_name(i), exist_ok=True)
    MAKE_ECG_SUMMARY_STATS = False or not os.path.exists(PRETEST_ECG_SUMMARY_STATS_CSV)
    MAKE_SPLIT_CSVS = False or not all(
        os.path.exists(_split_train_name(i)) for i in range(K_SPLIT)
    )
    TRAIN_PRETEST_MODELS = False or not all(
        os.path.exists(pretest_model_file(i, MODEL_SETTINGS[0].model_id))
        for i in range(K_SPLIT)
    )
    INFER_PRETEST_MODELS = (
            False or TRAIN_PRETEST_MODELS
            or not all(os.path.exists(_inference_file(split_idx)) for split_idx in range(K_SPLIT))
    )

    # run tasks
    if MAKE_LABELS:
        logging.info('Making biosppy labels.')
        build_hr_biosppy_measurements_csv()
    plot_hr_from_biosppy_summary_stats()
    plt.close('all')
    make_pretest_labels(MAKE_ECG_SUMMARY_STATS)
    plt.close('all')
    plot_pretest_label_summary_stats()
    plt.close('all')
    if MAKE_SPLIT_CSVS:
        build_csvs()
    aug_demo_ids = pd.read_csv(PRETEST_LABEL_FILE)["sample_id"].astype(str).sample(3)
    for setting in MODEL_SETTINGS:
        for sample_id in aug_demo_ids:
            path = os.path.join(TENSOR_FOLDER, _path_from_sample_id(sample_id))
            _demo_augmentations(path, setting)
    if TRAIN_PRETEST_MODELS:
        ray.init(
            num_cpus=cpu_count(),
            num_gpus=len(tf.config.experimental.list_physical_devices("GPU")),
        )
        remotes = []
        for i in range(K_SPLIT):
            for setting in MODEL_SETTINGS:
                if os.path.exists(history_tsv(i, setting.model_id)) and not OVERWRITE_MODELS:
                    logging.info(f'Skipping {setting.model_id} in split {i} since it already exists.')
                    continue
                remotes.append(_train_pretest_model.remote(setting, i))
        try:
            ray.get(remotes)
        finally:
            ray.shutdown()
    if INFER_PRETEST_MODELS:
        for i in range(K_SPLIT):
            logging.info(f'Running inference on split {i}.')
            _infer_models_split_idx(i)
    _evaluate_models()
    plot_training_curves()
