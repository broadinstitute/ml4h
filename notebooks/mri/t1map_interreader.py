# %%
import pandas as pd
import glob
import imageio
from adjustText import adjust_text
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from skimage.morphology import skeletonize, medial_axis
from skimage.segmentation import find_boundaries, clear_border
from skimage.measure import label, regionprops
from skimage.draw import line, disk
from google.cloud import storage 
import numpy.ma as ma
import numpy as np
from PIL import ImageColor
import cv2
from collections import defaultdict
import os
import seaborn as sns

storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')


def lineEnds(P):
    """Central pixel and just one other must be set to be a line end"""
    return ((P[4]==255) and np.sum(P)==510)

def erode_until(binary, ntarget):
    kernel = np.ones((3,3),np.uint8)
    iterations = 1
    npixels = 1e6
    while npixels > ntarget:
        erosion = cv2.erode(binary.astype(float), kernel, iterations = iterations)
        npixels = erosion.sum()
        iterations += 1
    return erosion                

colors = {
    "Body" : { "id": 1, "color": "#4169e1" },
    "Thoracic Cavity": { "id": 2, "color": "#984800" },
    "Liver": { "id": 3, "color": "#ffff00" },
    "Stomach": { "id": 4, "color": "#fca283" },
    "Spleen": { "id": 5, "color": "#f82387" },
    "Kidney": { "id": 6, "color": "#ffa500" },
    "Interventricular Septum": { "id": 7, "color": "#bfdad4", "id2": 1 },
    "Interventricular Septum ROI": { "id": 8, "color": "#42f94b" },
    "LV Free Wall": { "id": 9, "color": "#f9424b", "id2": 2 },
    "LV Free Wall ROI": { "id": 10, "color": "#42f9f0"},
    "LV Pap": { "id": 11, "color": "#0000ff" },
    "LV Cavity": { "id": 12, "color": "#256676", "id2": 3 },
    "LV Cavity ROI": { "id": 13, "color": "#a290ff" },
    "RV Free Wall": { "id": 14, "color": "#f996f1" },
    "RV Cavity": { "id": 15, "color": "#0ba47e", "id2": 4 },
    "RV Cavity ROI": { "id": 16, "color": "#a40b31" }	  
}

colors_roi = {
    "LV Free Wall ROI": { "id": 10, "color": "#42f9f0"},
    "LV Cavity ROI": { "id": 13, "color": "#a290ff" },
    "RV Cavity ROI": { "id": 16, "color": "#a40b31" }	  
}

test_df = pd.read_csv('/home/pdiachil/projects/t1map/t1_list_inter_reader_comparison_combined.txt', sep='\t')
test_df['ukbid'] = test_df['dicom_file'].str.split('_').str[0].apply(int)

# %%
means = defaultdict(list)
for color in colors:
    means[color] = defaultdict(list)

npats = len(test_df)
for key, item in means.items():
    item['vnauffal'] = [np.nan]*len(test_df)
    item['jcunning'] = [np.nan]*len(test_df)

means['sample_id'] = [np.nan]*npats
means['instance'] = [np.nan]*npats


for i, row in test_df.iterrows():
    print(i, row['ukbid'])
    try:
        sample_id = int(row['ukbid'])
        instance = 2
        means['sample_id'][i] = sample_id
        means['instance'][i] = instance
        sample_idx = test_df[test_df['ukbid']==sample_id].index[0]
        png_vnauffal = f'/home/pdiachil/projects/t1map/interreader/vnauffal/{sample_id}_2_0.png.mask.png'
        png_jcunning = f'/home/pdiachil/projects/t1map/interreader/jcunning_interreader/{sample_id}_2_0.png.mask.png'
        hd5_data_path = f'pdiachil/segmented-sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122-t1map/{sample_id}.hd5'
        blob = bucket.blob(hd5_data_path)
        blob.download_to_filename(f'{sample_id}.hd5')
        hd5_data_fname = f'{sample_id}.hd5'
        with h5py.File(hd5_data_fname, 'r') as hd5_data:
            found_anything = False
            best_mean = 0
            for key in hd5_data['ukb_cardiac_mri']:
                if ('t1map' in key) and ('sax' in key) and (str(instance) in hd5_data[f'ukb_cardiac_mri/{key}']):
                    found_anything = True
                    cur_data = hd5_data[f'ukb_cardiac_mri/{key}/{instance}/instance_0'][()]
                    cur_mean = np.mean(cur_data)
                    if cur_mean > best_mean:
                        best_mean = cur_mean
                        arr_data = cur_data
            image_vnauffal = imageio.imread(png_vnauffal)[:, :, 0]
            image_jcunning = imageio.imread(png_jcunning)[:, :, 0]

                                
            skeletons = {}                    
            for color, color_dic in colors.items():
                roi_arr_vnauffal = image_vnauffal==color_dic["id"]
                roi_arr_jcunning = image_jcunning==color_dic["id"]
                if 'roi' in color.lower():
                    select_vnauffal = arr_data[:, :, 0][roi_arr_vnauffal>0.5]
                    select_jcunning = arr_data[:, :, 0][roi_arr_jcunning>0.5]
                    mean_vnauffal = np.median(select_vnauffal)
                    mean_jcunning = np.median(select_jcunning)

                    means[color]['vnauffal'][i] = mean_vnauffal
                    means[color]['jcunning'][i] = mean_jcunning
                elif color + ' ROI' in colors:
                    roi2_arr_vnauffal = image_vnauffal==colors[color+' ROI']["id"]
                    roi2_arr_jcunning = image_jcunning==colors[color+' ROI']["id"]
                    dice = np.sum(np.logical_and(roi_arr_vnauffal+roi2_arr_vnauffal, 
                                                 roi_arr_jcunning+roi2_arr_jcunning)) * 2.0
                    dice /= np.sum(roi_arr_vnauffal+roi2_arr_vnauffal) \
                            + np.sum(roi_arr_jcunning+roi2_arr_jcunning)
                    means[color]['vnauffal'][i] = dice
    except FutureWarning:
        pass
    os.remove(f'{sample_id}.hd5')
# %%
import scipy.stats

df_dic = {}
for region in means:
    if 'sample_id' in region:
        df_dic[region] = means['sample_id']
    elif 'instance' in region:
        df_dic[region] = means['instance']
    else:
        df_dic[f'{region}_vnauffal'] = means[region]['vnauffal']
        df_dic[f'{region}_jcunning'] = means[region]['jcunning']
df = pd.DataFrame(df_dic)

# %%
f, ax = plt.subplots()
ax.plot(df['LV Free Wall ROI_vnauffal'], df['LV Free Wall ROI_jcunning'], 'ko')
ax.set_xlabel('LV FW VN (ms)')
ax.set_ylabel('LV FW JC (ms)')
ax.set_aspect('equal')
ax.plot([400.0, 1200], [400, 1200], 'k--')
ax.set_xlim([400.0, 1200])
ax.set_ylim([400.0, 1200])
ax.set_title(f"r={np.corrcoef(df['LV Free Wall ROI_vnauffal'], df['LV Free Wall ROI_jcunning'])[0, 1]:.2f}")
f.savefig('interreader_fw.png', dpi=500)


# %%
cols = [col for col in df.columns if 'ROI' in col]
labels = ['LV FW', 'IVS', 'LV BP', 'RV BP']
xticklabels = []
for l in labels:
    xticklabels.append(l+' VN')
    xticklabels.append(l+' JC')
f, ax = plt.subplots()
sns.heatmap(df[cols].corr(), annot=True, cmap='gray')
ax.set_xticklabels(xticklabels, rotation=45, ha='right')
ax.set_yticklabels(xticklabels)
plt.tight_layout()
f.savefig('interreader.png', dpi=500)

# %%
# Plot dice scores
cols_dice = ['Interventricular Septum', 'LV Free Wall', 'RV Cavity', 'LV Cavity']
f, ax = plt.subplots()
xticks = []
xticklabels = []
for i, col in enumerate(cols_dice):
    avg_dice = np.mean(means[col]['vnauffal'][:])
    std_dice = np.std(means[col]['vnauffal'][:])
    ax.bar(i, avg_dice, yerr=std_dice, color=np.array([0.2, 0.2, 0.2])*i, label=col)
    xticks.append(i)
    xticklabels.append(col)
ax.set_ylabel('Dice Score')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=45, ha='right')
plt.tight_layout()
f.savefig('dice_scores.png', dpi=500)

# %%
# df = df[df['LV Free Wall_model']>30.0]
i = 0
j = 0
f, ax = plt.subplots(2, 2)
f.set_size_inches(6, 6)
for region in ['LV Free Wall ROI']:
    if 'sample_id' in region:
        continue
    ax[i, j].plot(df[f'{region}_vnauffal'], df[f'{region}_jcunning'], 'ko', alpha=0.1)
    ax[i, j].plot([300.0, 2000.], [300., 2000.], 'k-')
    if 'cavity' in region.lower():
        ax[i, j].set_xlim([1000.0, 2000.0])
        ax[i, j].set_ylim([1000.0, 2000.0])
    else:
        ax[i, j].set_xlim([500, 1200.0])
        ax[i, j].set_ylim([500, 1200.0])

    ax[i, j].set_xlabel('T1 segmentation')
    ax[i, j].set_ylabel('T1 model')

    # Outliers
    diff = np.abs(df[f'{region}_vnauffal'] - df[f'{region}_jcunning']) / df[f'{region}_vnauffal']
    outliers = []
    outliers.extend(diff.argsort()[-2:])
    diff = np.abs((df[f'{region}_vnauffal'] - np.mean(df[f'{region}_vnauffal'])) / np.std(df[f'{region}_vnauffal']))
    outliers.extend(diff.argsort()[-2:])
    diff = np.abs((df[f'{region}_jcunning'] - np.mean(df[f'{region}_jcunning'])) / np.std(df[f'{region}_jcunning']))
    outliers.extend(diff.argsort()[-2:])
    texts = []
    for outlier in list(set(outliers)):
        row = df.iloc[outlier]
        texts.append(ax[i, j].text(row[f'{region}_vnauffal'], row[f'{region}_jcunning'], str(int(row['sample_id'])), ha='center', va='center'))
    adjust_text(texts)
    r = scipy.stats.spearmanr(df[f'{region}_vnauffal'], df[f'{region}_jcunning']).correlation
    r = np.corrcoef(df[f'{region}_vnauffal'], df[f'{region}_jcunning'])[0, 1]
    ax[i, j].set_title(f'{region} r: {r:.2f}')    
    j += 1
    i = j // 2 + i
    j = j % 2
plt.tight_layout()

# %%
df.to_csv('performance_training_3px_circle.csv', index=False)
f.savefig('performance_training_3px_circle.png')
# %%
