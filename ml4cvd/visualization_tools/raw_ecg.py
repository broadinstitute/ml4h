"""Methods for reshaping raw ECG signal data for use in the pandas ecosystem."""
import os
import tempfile

from biosppy.signals.tools import filter_signal
import h5py
from ml4cvd.defines import ECG_REST_LEADS
from ml4cvd.defines import ECG_BIKE_LEADS
from ml4cvd.runtime_data_defines import get_exercise_ecg_hd5_folder
from ml4cvd.runtime_data_defines import get_resting_ecg_hd5_folder
from ml4cvd.tensor_maps_by_hand import TMAPS
from ml4cvd.TensorMap import TensorMap
import numpy as np
import pandas as pd
import tensorflow as tf


RAW_SCALE = 0.005  # Convert to mV.
SAMPLING_RATE = 500.0
RESTING_SIGNAL_LENGTH = 5000
RESTING_ECG_SIGNAL_TMAP = TMAPS['ecg_rest_raw']
EXERCISE_SIGNAL_LENGTH = 30000
EXERCISE_ECG_SIGNAL_TMAP = TMAPS['ecg-bike-raw-full']
EXERCISE_ECG_TREND_TMAPS = [
    TMAPS['ecg-bike-raw-trend-hr'],
    TMAPS['ecg-bike-raw-trend-load'],
    TMAPS['ecg-bike-raw-trend-grade'],
    TMAPS['ecg-bike-raw-trend-artifact'],
    TMAPS['ecg-bike-raw-trend-mets'],
    TMAPS['ecg-bike-raw-trend-pacecount'],
    TMAPS['ecg-bike-raw-trend-phasename'],
    TMAPS['ecg-bike-raw-trend-phasetime'],
    TMAPS['ecg-bike-raw-trend-time'],
    TMAPS['ecg-bike-raw-trend-vecount'],
]
EXERCISE_PHASES = {0.0: 'Pretest', 1.0: 'Exercise', 2.0: 'Recovery'}


def reshape_resting_ecg_to_tidy(sample_id, folder=None):
  """Wrangle raw resting ECG data to tidy.

  Args:
    sample_id: The id of the ECG sample to retrieve.
    folder: The local or Cloud Storage folder under which the files reside.

  Returns:
    A pandas dataframe in tidy format or a notebook-friendly error.
  """
  if folder is None:
    folder = get_resting_ecg_hd5_folder(sample_id)

  data = {'lead': [], 'raw': [], 'ts_reference': [], 'filtered': [], 'filtered_1': [], 'filtered_2': []}

  with tempfile.TemporaryDirectory() as tmpdirname:
    sample_hd5 = str(sample_id) + '.hd5'
    local_path = os.path.join(tmpdirname, sample_hd5)
    try:
      tf.io.gfile.copy(src=os.path.join(folder, sample_hd5), dst=local_path)
    except (tf.errors.NotFoundError, tf.errors.PermissionDeniedError) as e:
      print(f'Warning: Resting ECG raw signal not available for sample {sample_id}\n\n{e.message}')
      return pd.DataFrame(data)

    with h5py.File(local_path, mode='r') as hd5:
      try:
        signals = RESTING_ECG_SIGNAL_TMAP.tensor_from_file(RESTING_ECG_SIGNAL_TMAP, hd5)
      except ValueError as e:
        print(f'Warning: Resting ECG raw signal not available for sample {sample_id}\n\n{e.message}')
        return pd.DataFrame(data)
      for (lead, channel) in ECG_REST_LEADS.items():
        signal = signals[:, channel]
        if(RESTING_SIGNAL_LENGTH != len(signal)):
          print(f'Warning: Resting ECG raw signal is malformed for sample {sample_id}')
          return pd.DataFrame(data)
        data['raw'].extend(signal)
        data['lead'].extend([lead] * RESTING_SIGNAL_LENGTH)
        data['ts_reference'].extend(np.array([i*1./(SAMPLING_RATE+1.) for i in range(0, RESTING_SIGNAL_LENGTH)]))
        filtered, _, _ = filter_signal(
            signal=signal,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * SAMPLING_RATE),
            frequency=[.9, 50],
            sampling_rate=SAMPLING_RATE,
        )
        data['filtered'].extend(filtered)
        filtered_1, _, _ = filter_signal(
            signal=signal,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * SAMPLING_RATE),
            frequency=[.9, 20],
            sampling_rate=SAMPLING_RATE,
        )
        data['filtered_1'].extend(filtered_1)
        filtered_2, _, _ = filter_signal(
            signal=signal,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * SAMPLING_RATE),
            frequency=[.9, 30],
            sampling_rate=SAMPLING_RATE,
        )
        data['filtered_2'].extend(filtered_2)

  signal_df = pd.DataFrame(data)
  # Convert the raw signal to mV.
  signal_df['raw_mV'] = signal_df['raw'] * RAW_SCALE
  signal_df['filtered_mV'] = signal_df['filtered'] * RAW_SCALE
  signal_df['filtered_1_mV'] = signal_df['filtered_1'] * RAW_SCALE
  signal_df['filtered_2_mV'] = signal_df['filtered_2'] * RAW_SCALE
  # Reshape to tidy (long format).
  tidy_signal_df = signal_df.melt(
      id_vars=['lead', 'ts_reference'],
      value_vars=['raw_mV', 'filtered_mV', 'filtered_1_mV', 'filtered_2_mV'],
      var_name='filtering', value_name='signal_mV',
  )

  # The leads have a meaningful order, apply the order to this column.
  lead_factor_type = pd.api.types.CategoricalDtype(
      categories=[
          'strip_I', 'strip_aVR', 'strip_V1', 'strip_V4',
          'strip_II', 'strip_aVL', 'strip_V2', 'strip_V5',
          'strip_III', 'strip_aVF', 'strip_V3', 'strip_V6',
      ],
      ordered=True,
  )
  tidy_signal_df['lead'] = tidy_signal_df.lead.astype(lead_factor_type)

  return tidy_signal_df


def reshape_exercise_ecg_to_tidy(sample_id, folder=None):
  """Wrangle raw exercise ECG signal data to tidy format.

  Args:
    sample_id: The id of the ECG sample to retrieve.
    folder: The local or Cloud Storage folder under which the files reside.

  Returns:
    A tuple of pandas dataframesor a notebook-friendly error.
    * first tuple element is trend data in wide format
    * second tuple element is signal data in tidy format
  """
  if folder is None:
    folder = get_exercise_ecg_hd5_folder(sample_id)

  with tempfile.TemporaryDirectory() as tmpdirname:
    sample_hd5 = str(sample_id) + '.hd5'
    local_path = os.path.join(tmpdirname, sample_hd5)
    try:
      tf.io.gfile.copy(src=os.path.join(folder, sample_hd5), dst=local_path)
    except (tf.errors.NotFoundError, tf.errors.PermissionDeniedError) as e:
      print(f'Error: Exercise ECG raw signal not available for sample {sample_id}\n\n{e.message}')
      return (pd.DataFrame({}), pd.DataFrame({}))

    with h5py.File(local_path, mode='r') as hd5:
      trend_data = {}
      for tmap in EXERCISE_ECG_TREND_TMAPS:
        try:
          tensor = tmap.tensor_from_file(tmap, hd5)
          trend_data[tmap.name.replace('trend_', '')] = tensor
        except ValueError as e:
          print(f'Warning: Exercise ECG trend not available for sample {sample_id}\n\n{e.message}')
          return (pd.DataFrame({}), pd.DataFrame({}))
      try:
        full = EXERCISE_ECG_SIGNAL_TMAP.tensor_from_file(EXERCISE_ECG_SIGNAL_TMAP, hd5)
      except ValueError as e:
        print(f'Warning: Exercise ECG raw signal not available for sample {sample_id}\n\n{e.message}')
        return (pd.DataFrame({}), pd.DataFrame({}))

  signal_data = {}
  for (lead, channel) in ECG_BIKE_LEADS.items():
    signal_data['raw_mV_' + lead] = full[:, channel] * RAW_SCALE
  signal_data['time'] = np.arange(len(full)) / SAMPLING_RATE

  # Convert exercise ecg trend tensor dictionarys to a dataframe and
  # clean data as needed
  trend_df = pd.DataFrame(trend_data)
  # Clean data - convert to categorical string.
  trend_df['phasename'] = trend_df.phasename.map(EXERCISE_PHASES).astype('category')

  # Convert exercise ecg signal tensor dictionary to a dataframe, clean data
  # as needed, and then pivot to tidy.
  signal_df = pd.DataFrame(signal_data)
  tidy_signal_df = pd.wide_to_long(
      signal_df,
      stubnames=['raw_mV'],
      i='time',
      j='lead',
      sep='_',
      suffix='.*',
  )
  tidy_signal_df.reset_index(inplace=True)  # Turn pd multiindex into columns.
  # The leads have a meaningful order, apply the order to this column.
  lead_factor_type = pd.api.types.CategoricalDtype(categories=ECG_BIKE_LEADS.keys(), ordered=True)
  tidy_signal_df['lead'] = tidy_signal_df.lead.astype(lead_factor_type)

  return (trend_df, tidy_signal_df)


def reshape_exercise_ecg_and_trend_to_tidy(sample_id, folder=None):
  """Wrangle raw exercise ECG signal and trend data to tidy format.

  Args:
    sample_id: The id of the ECG sample to retrieve.
    folder: The local or Cloud Storage folder under which the files reside.

  Returns:
    A tuple of pandas dataframesor a notebook-friendly error.
    * first tuple element is trend data in tidy format
    * second tuple element is signal data in tidy format
  """

  # Get the trend data in wide format and pivot it to tidy.
  (trend_df, tidy_signal_df) = reshape_exercise_ecg_to_tidy(sample_id, folder)
  # Clean data - drop zero-valued columns.
  trend_df = trend_df.loc[:, ~trend_df.eq(0).all()]
  trend_id_vars = ['time', 'phasename', 'phasetime']
  trend_value_vars = trend_df.columns[~trend_df.columns.isin(trend_id_vars)].tolist()
  tidy_trend_df = trend_df.melt(
      id_vars=trend_id_vars,
      value_vars=trend_value_vars,
      var_name='measurement',
      value_name='value',
  )

  return (tidy_trend_df, tidy_signal_df)
