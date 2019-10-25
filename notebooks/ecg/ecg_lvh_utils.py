# Imports
import os
import h5py
import argparse
import matplotlib
import numpy as np
import pandas as pd
from ast import literal_eval
from biosppy.signals import ecg
import matplotlib.pyplot as plt
from IPython.display import Image
from collections import defaultdict
from ml4cvd.arguments import _get_tmap
import matplotlib.gridspec as gridspec
from ml4cvd.tensor_maps_by_hand import TMAPS
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from ml4cvd.tensor_generators import test_train_valid_tensor_generators, get_test_train_valid_paths, TensorGenerator

from matplotlib import rcParams
rcParams['font.size'] = 20

lead_mapping = np.array([['strip_I','strip_aVR', 'strip_V1', 'strip_V4'],
                  ['strip_II','strip_aVL', 'strip_V2', 'strip_V5'],
                  ['strip_III','strip_aVF', 'strip_V3', 'strip_V6'],
                 ])
median_mapping = np.array([['median_I','median_aVR', 'median_V1', 'median_V4'],
              ['median_II','median_aVL', 'median_V2', 'median_V5'],
              ['median_III','median_aVF', 'median_V3', 'median_V6'],
             ])
amp_mapping = np.array([[0, 3, 6, 9],
                        [1, 4, 7, 10],
                        [2, 5, 8, 11]])

def get_ecg_traces(hd5):
    leads = {}
    for field in hd5['ecg_rest']:
        leads[field] = list(hd5['ecg_rest'][field])

    twelve_leads = defaultdict(dict)
    for key, data in leads.items(): 
        twelve_leads[key]['raw'] = leads[key]
        if len(data) == 5000:
            (twelve_leads[key]['ts_reference'], twelve_leads[key]['filtered'], twelve_leads[key]['rpeaks'], 
             twelve_leads[key]['template_ts'], twelve_leads[key]['templates'], twelve_leads[key]['heart_rate_ts'], 
             twelve_leads[key]['heart_rate']) = ecg.ecg(signal=leads[key], sampling_rate = 500., show=False)    
    return twelve_leads

def get_ylims(yrange, y_plot):
    deltas   = [-1.0, 1.0]
    extremes = np.array([np.min(y_plot), np.max(y_plot)])
    delta_ext = extremes[1]-extremes[0]
    yrange = np.max([yrange, delta_ext*1.10])
    ylim_min = -yrange/2.0
    ylim_max = yrange/2.0
    if ((extremes[0] - ylim_min) < yrange*0.2) or \
       ((ylim_max-extremes[1]) < yrange*0.2) : 
        ylim_min = extremes[0] - (yrange-delta_ext)/2.0
        ylim_max = extremes[1] + (yrange-delta_ext)/2.0       
    return ylim_min, ylim_max


def get_yrange(twelve_leads, default_yrange=3.0, raw_scale=0.005, time_interval=2.5):
    yrange=default_yrange
    for is_median, offset in zip([False, True], [3, 0]):
        for i in range(offset,offset+3):
            for j in range(0,4):
                lead_name = lead_mapping[i-offset,j]
                lead = twelve_leads[lead_name]
                y_plot = np.array([elem_ * raw_scale for elem_ in lead['raw']])
                if not is_median:        
                    y_plot = y_plot[np.logical_and(lead['ts_reference']>j*int(time_interval),
                                    lead['ts_reference']<(j+1)*int(time_interval))]
                ylim_min, ylim_max = get_ylims(default_yrange, y_plot)
                yrange = ylim_max - ylim_min
    return yrange
    
def plot_traces(twelve_leads, lead_mapping, amp_mapping, f, ax, yrange, offset, pat_df=None, is_median=False, raw_scale = 0.005, is_blind=False):
    # plot will be in seconds vs mV, boxes are 
    sec_per_box = 0.04
    mv_per_box = .1
    time_interval = 2.5 # time-interval per plot in seconds. ts_Reference data is in s, voltage measurement is 5 uv per lsb
    median_interval = 1.2  # 600 samples at 500Hz
     # convert to mV
    if pat_df is not None:
        avl_yn = 'Y' if pat_df['aVL']>0.5 else 'N'
        sl_yn  = 'Y' if pat_df['Sokolow_Lyon']>0.5 else 'N'
        cor_yn = 'Y' if pat_df['Cornell']>0.5 else 'N'
        sex_fm = 'F' if pat_df['sex'] == 'female' else 'M'
        text   = f"ID: {pat_df['patient_id']}, sex: {sex_fm}\n"        
        if not is_blind:
            text  += f"{pat_df['ecg_text']}\n"
            text  += f"LVH criteria - aVL: {avl_yn}, Sokolow-Lyon: {sl_yn}, Cornell: {cor_yn}"            
        st=f.suptitle(text, x=0.0, y=1.05, ha='left', bbox=dict(facecolor='black', alpha=0.1))   
    for i in range(offset,offset+3):
        for j in range(0, 4):
            lead_name = lead_mapping[i-offset,j]
            lead = twelve_leads[lead_name]
            yy = np.array([elem_ * raw_scale for elem_ in lead['raw']])
            if not is_median:
                ax[i,j].set_xlim(j*time_interval,(j+1)*time_interval)  
                y_plot = yy[np.logical_and(lead['ts_reference']>j*time_interval,
                                lead['ts_reference']<(j+1)*time_interval)]
            else:
                y_plot = yy                       
            ylim_min, ylim_max = get_ylims(yrange, y_plot)            
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
                # Find where to put the R and S amps
                dy_ecg = (yy[-1] - ylim_min) / yrange
                if dy_ecg > 0.3: # Put in bottom right
                    dy_amp = 0.2
                else:
                    dy_amp = 0.85
                ax[i,j].text(0.9, dy_amp*yrange+ylim_min, f"R: {int(pat_df['ramp'][amp_mapping[i-offset, j]])}")
                ax[i,j].text(0.9, (dy_amp-0.15)*yrange+ylim_min, f"S: {int(pat_df['samp'][amp_mapping[i-offset, j]])}")

def ecg_csv_to_df(csv):
    df = pd.read_csv(csv)
    df['ramp'] = df['ramp'].apply(literal_eval)
    df['samp'] = df['samp'].apply(literal_eval)
    df['patient_id'] = df['patient_id'].apply(str)
    df['Sokolow_Lyon'] = df['Sokolow_Lyon'].apply(float)
    df['Cornell'] = df['Cornell'].apply(float)
    df['aVL'] = df['aVL'].apply(float)
    return df                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_url', help='URL to storage bucket',
                        default='https://console.cloud.google.com/storage/browser/_details/ml4cvd/ecg_views_10_23_2019_all/')
    parser.add_argument('--project_url', help='URL tail to select project', default='?project=broad-ml4cvd')
    parser.add_argument('--ecg_csv', help='CSV file with dataframe of ECGs', default='ecg_views_all_10_23_2019.csv')
    parser.add_argument('--row_num', help='Patient entry number to plot', type=int)
    parser.add_argument('--out_folder', help='Folder where to save PDFs', default='/home/pdiachil/ml/notebooks/ecg')
    parser.add_argument('--blind', help='Removes automatic ECG interpretation for blind test', action='store_true')
    args = parser.parse_args()
    
    df = ecg_csv_to_df(args.ecg_csv)
    pat_df = df.iloc[args.row_num]
    with h5py.File(pat_df['full_path'], 'r') as hd5:
        traces = get_ecg_traces(hd5)
    fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(24,18), tight_layout=True)
    yrange = get_yrange(traces)
    plot_traces(traces, lead_mapping, amp_mapping, fig, ax, yrange, offset=3, pat_df=None, is_median=False, is_blind=args.blind)
    plot_traces(traces, median_mapping, amp_mapping, fig, ax, yrange, offset=0, pat_df=pat_df, is_median=True, is_blind=args.blind)
    fig.savefig(os.path.join(args.out_folder, pat_df['patient_id']+'.pdf'), bbox_inches = "tight")
    
    


