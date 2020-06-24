#%%
import os
import re
import math
import h5py
import glob
import logging
import hashlib
import operator
from textwrap import wrap
from functools import reduce
from datetime import datetime
from multiprocessing import Pool
from itertools import islice, product
from collections import Counter, OrderedDict, defaultdict
from typing import Iterable, DefaultDict, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import matplotlib
matplotlib.use('Agg')  # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt  # First import matplotlib, then use Agg, then import plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from ml4cvd.TensorMap import decompress_data
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sksurv.metrics import concordance_index_censored

import seaborn as sns
from biosppy.signals import ecg
from scipy.ndimage.filters import gaussian_filter

from ml4cvd.TensorMap import TensorMap
from ml4cvd.metrics import concordance_index, coefficient_of_determination
from ml4cvd.defines import IMAGE_EXT, JOIN_CHAR, PDF_EXT, TENSOR_EXT, ECG_REST_LEADS, ECG_REST_AMP_LEADS, ECG_REST_MEDIAN_LEADS, PARTNERS_DATETIME_FORMAT, PARTNERS_DATE_FORMAT
ECG_REST_PLOT_DEFAULT_YRANGE = 3.0
ECG_REST_PLOT_MAX_YRANGE = 10.0
ECG_REST_PLOT_LEADS = [
    ['I','aVR', 'V1', 'V4'],
    ['II','aVL', 'V2', 'V5'],
    ['III','aVF', 'V3', 'V6'],
]
ECG_REST_PLOT_MEDIAN_LEADS = [
    ['median_I','median_aVR', 'median_V1', 'median_V4'],
    ['median_II','median_aVL', 'median_V2', 'median_V5'],
    ['median_III','median_aVF', 'median_V3', 'median_V6'],
]
ECG_REST_PLOT_AMP_LEADS = [
    [0, 3, 6, 9],
    [1, 4, 7, 10],
    [2, 5, 8, 11],
]

def _ecg_rest_traces_and_text(hd5, date):
    """Extracts ECG resting traces from HD5 and returns a dictionary based on biosppy template"""
    path_prefix = 'partners_ecg_rest'
    ecg_text_group = 'read_md_clean'
    if path_prefix not in hd5:
        raise ValueError('Tensor does not contain resting ECGs')
    twelve_leads = defaultdict(dict)
    leads = list(ECG_REST_AMP_LEADS.keys())
    oldest_instance = date
    for lead in leads:
        twelve_leads[lead]['raw'] = decompress_data(hd5[path_prefix][oldest_instance][lead][()],
                                                    hd5[path_prefix][oldest_instance][lead].attrs['dtype'])       
        if len(twelve_leads[lead]['raw']) >= 2500:
            try:
                # Attempt analysis by biosppy, which may fail if not enough beats
                (
                    twelve_leads[lead]['ts_reference'], twelve_leads[lead]['filtered'], twelve_leads[lead]['rpeaks'],
                    twelve_leads[lead]['template_ts'], twelve_leads[lead]['templates'], twelve_leads[lead]['heart_rate_ts'],
                    twelve_leads[lead]['heart_rate'],
                ) = ecg.ecg(signal=twelve_leads[lead]['raw'], sampling_rate = 250., show=False)
            except:
                twelve_leads[lead]['ts_reference'] = np.linspace(0, len(twelve_leads[lead]['raw'])/500., len(twelve_leads[lead]['raw']))
    ecg_rest_text = ''
    if ecg_text_group in hd5[path_prefix][oldest_instance]:
        ecg_rest_text = decompress_data(hd5[path_prefix][oldest_instance][ecg_text_group][()],
                                        hd5[path_prefix][oldest_instance][ecg_text_group].attrs['dtype'])
    return twelve_leads, ecg_rest_text


def _ecg_rest_ylims(yrange, yplot):
    """Returns ECG plot y-axis limits based on the range covered by ECG traces"""
    deltas   = [-1.0, 1.0]
    extremes = np.array([np.min(yplot), np.max(yplot)])
    delta_ext = extremes[1]-extremes[0]
    yrange = np.max([yrange, delta_ext*1.10])
    ylim_min = -yrange/2.0
    ylim_max = yrange/2.0
    if ((extremes[0] - ylim_min) < yrange*0.2) or \
       ((ylim_max-extremes[1]) < yrange*0.2) :
        ylim_min = extremes[0] - (yrange-delta_ext)/2.0
        ylim_max = extremes[1] + (yrange-delta_ext)/2.0
    return ylim_min, ylim_max


def _ecg_rest_yrange(twelve_leads, default_yrange, raw_scale, time_interval):
    """Returns y-range necessary not to cut any of the plotted ECG waveforms"""
    yrange = default_yrange
    for is_median, offset in zip([False], [3]):
        for i in range(offset, offset+3):
            for j in range(0, 4):
                lead_name = ECG_REST_PLOT_LEADS[i-offset][j]
                lead = twelve_leads[lead_name]
                y_plot = np.array([elem_ * raw_scale for elem_ in lead['raw']])
                if not is_median:
                    y_plot = y_plot[
                        np.logical_and(
                            lead['ts_reference']>j*time_interval,
                            lead['ts_reference']<(j+1)*time_interval,
                        )
                    ]
                ylim_min, ylim_max = _ecg_rest_ylims(yrange, y_plot)
                yrange = ylim_max - ylim_min
    return min(yrange, ECG_REST_PLOT_MAX_YRANGE)


def _subplot_ecg_rest(twelve_leads, raw_scale, time_interval, hertz, lead_mapping, f, ax, yrange, offset, pat_df, is_median, is_blind):
    """Fills subplots with either median or raw resting ECG waveforms"""
    # plot will be in seconds vs mV, boxes are
    sec_per_box = 0.04
    mv_per_box = .1
    median_interval = 1.2  # 600 samples at 500Hz
    # if available, extract patient metadata and ECG interpretation
    if pat_df is not None:
        avl_yn = 'Y' if np.argmax(pat_df['aVL'])>0.5 else 'N'
        sl_yn  = 'Y' if np.argmax(pat_df['Sokolow_Lyon'])>0.5 else 'N'
        cor_yn = 'Y' if np.argmax(pat_df['Cornell'])>0.5 else 'N'
        sex_fm = 'F' if ((pat_df['sex'] == 'F') or (pat_df['sex'] == 'female')) else 'M'
        text   = f"ID: {pat_df['patient_id']}, sex: {sex_fm}\n"
        if not is_blind:
            text  += f"{pat_df['ecg_text']}\n"
            text  += f"LVH criteria - aVL: {avl_yn}, Sokolow-Lyon: {sl_yn}, Cornell: {cor_yn}"
        st=f.suptitle(text, x=0.0, y=1.05, ha='left', bbox=dict(facecolor='black', alpha=0.1))
    for i in range(offset, offset+3):
        for j in range(0, 4):
            lead_name = lead_mapping[i-offset][j]
            lead = twelve_leads[lead_name]
            # Convert units to mV
            if isinstance(lead, dict):
                yy = np.array([elem_ * raw_scale for elem_ in lead['raw']])
            else:
                yy = lead
            if not is_median:
                ax[i,j].set_xlim(j*time_interval,(j+1)*time_interval)
                # extract portion of waveform that is included in the actual plots
                yplot = yy[int(j*time_interval*hertz): int((j+1)*time_interval*hertz)]
            else:
                yplot = yy
            ylim_min, ylim_max = _ecg_rest_ylims(yrange, yplot)
            ax[i,j].set_ylim(ylim_min, ylim_max) # 3.0 mV range
            ax[i,j].xaxis.set_major_locator(MultipleLocator(0.2)) # major grids at every .2sec = 5 * 0.04 sec
            ax[i,j].yaxis.set_major_locator(MultipleLocator(0.5)) # major grids at every .5mV
            ax[i,j].xaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i,j].yaxis.set_minor_locator(AutoMinorLocator(5))
            ax[i,j].grid(which='major', color='#CCCCCC', linestyle='--')
            ax[i,j].grid(which='minor', color='#CCCCCC', linestyle=':')
            for label in ax[i,j].xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            if len(ax[i,j].yaxis.get_ticklabels()) > 10:
                for label in ax[i,j].yaxis.get_ticklabels()[::2]:
                    label.set_visible(False)
            #normalize data in muv
            if 'ts_reference' in lead:
                ax[i,j].plot(lead['ts_reference'], yy, label='raw')
            else:
                ax[i,j].plot(np.arange(0.0, median_interval, median_interval/len(lead['raw'])), yy, label='raw')
            ax[i,j].set_title(lead_name)
            if is_median and (pat_df is not None):
                # Find where to put the R and S amp text based on ECG baseline position
                dy_ecg = (yy[-1] - ylim_min) / yrange
                if dy_ecg > 0.3: # Put in bottom right
                    dy_amp = 0.2
                else: # Put in top right
                    dy_amp = 0.85
                ax[i,j].text(0.9, dy_amp*yrange+ylim_min, f"R: {pat_df['ramp'][ECG_REST_PLOT_AMP_LEADS[i-offset][j]]:.0f}")
                ax[i,j].text(0.9, (dy_amp-0.15)*yrange+ylim_min, f"S: {pat_df['samp'][ECG_REST_PLOT_AMP_LEADS[i-offset][j]]:.0f}")


def _str_to_list_float(str_list: str) -> List[int]:
    """'[ 3. 4. nan 3 ]' --> [ 3.0, 4.0, nan, 3.0 ]"""
    tmp_str = str_list[1:-1].split()
    return list(map(float, tmp_str))


def _ecg_rest_csv_to_df(csv):
    df = pd.read_csv(csv)
    df['ramp'] = df['ramp'].apply(_str_to_list_float)
    df['samp'] = df['samp'].apply(_str_to_list_float)
    df['patient_id'] = df['patient_id'].apply(str)
    df['Sokolow_Lyon'] = df['Sokolow_Lyon'].apply(float)
    df['Cornell'] = df['Cornell'].apply(float)
    df['aVL'] = df['aVL'].apply(float)
    return df


def _remove_duplicate_rows(df, out_folder):
    arr_list = []
    pdfs = glob.glob(out_folder+'/*.pdf')
    for i, row in df.iterrows():
        if os.path.join(out_folder, row['patient_id']+'.pdf') not in pdfs:
            arr_list.append(i)
    arr = np.array(arr_list, dtype=np.int)
    return arr


def plot_ecg_rest(tensor_paths: List[str], rows: List[int],
                  dates,
                  out_folder: str, is_blind: bool) -> None:
    """ Plots resting ECGs including annotations and LVH criteria
    
    :param tensor_paths: list of HDF5 file paths with ECG traces
    :param rows: indices of the subset of tensor_paths to be plotted (used by multiprocessing)
    :param out_folder: destination folder for the plots
    :param is_blind: if True, the plot gets blinded (helpful for review and annotation)
    """
    map_fields_to_tmaps = {
        'ramp': 'ecg_rest_ramplitude_raw', 
        'samp': 'ecg_rest_samplitude_raw',
        'aVL': 'ecg_rest_lvh_avl',
        'Sokolow_Lyon': 'ecg_rest_lvh_sokolow_lyon',
        'Cornell': 'ecg_rest_lvh_cornell'
        }    
    from ml4cvd.tensor_from_file import TMAPS
    raw_scale = 0.005 # Conversion from raw to mV
    # raw_scale = 1.0
    default_yrange = ECG_REST_PLOT_DEFAULT_YRANGE # mV
    time_interval = 2.5 # time-interval per plot in seconds. ts_Reference data is in s, voltage measurement is 5 uv per lsb
    hertz = 250 # number of samples per second
    for row in rows:
        tensor_path = tensor_paths[row]
        tensor_date = dates[row]
        patient_dic = {}
        patient_dic['patient_id'] = os.path.basename(tensor_path).replace(TENSOR_EXT, '')
        with h5py.File(tensor_path, 'r') as hd5:
            traces, text = _ecg_rest_traces_and_text(hd5, tensor_date)
            for field in map_fields_to_tmaps:
                tm = TMAPS[map_fields_to_tmaps[field]]
                patient_dic[field] = np.zeros(tm.shape)
                try:
                    patient_dic[field][:] = tm.tensor_from_file(tm, hd5)
                except ValueError as e:
                    logging.warning(e)
            is_female = False
            patient_dic['sex'] = 'F' if is_female else 'M'
            patient_dic['ecg_text'] = text
        matplotlib.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(24,18), tight_layout=True)
        yrange = _ecg_rest_yrange(traces, default_yrange, raw_scale, time_interval)
        _subplot_ecg_rest(
            traces, raw_scale, time_interval, hertz, ECG_REST_PLOT_LEADS, fig, ax, yrange,
            offset=3, pat_df=None, is_median=False, is_blind=is_blind,
        )
        # _subplot_ecg_rest(
        #     traces, raw_scale, time_interval, hertz, ECG_REST_PLOT_MEDIAN_LEADS, fig, ax, yrange,
        #     offset=0, pat_df=patient_dic, is_median=True, is_blind=is_blind,
        # )
        fig.savefig(os.path.join(out_folder, patient_dic['patient_id']+'.svg'), bbox_inches = "tight")


# # %%
# hd5 =  h5py.File('/data/partners_ecg/mgh/hd5/2402285.hd5', 'r')
# %matplotlib inline
# date = '2019-01-11T09:57:55'
# leads, text = _ecg_rest_traces_and_text(hd5, date)



# # %%
# f, ax = plt.subplots(12, 1)
# for i, lead in enumerate(leads):
#     ax[i].plot(leads[lead]['raw'])

# %%
