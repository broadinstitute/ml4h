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

test_df = pd.read_csv('/mnt/disks/pdiachil-t1map/predictions3/evaluate_batch_holdout.tsv', sep='\t')

skip = [
2948608,
5240555,
2186364,
2178219,
1390350,
2997282,
1573604,
3164760,
3791035,
4418788,
4594283,
4867069,
4868739,
5007385,
5204020,
1778902]
# %%
means = defaultdict(list)
means['LV Free Wall'] = defaultdict(list)
means['Interventricular Septum'] = defaultdict(list)
means['LV Cavity'] = defaultdict(list)
means['RV Cavity'] = defaultdict(list)
for i, row in test_df.iterrows():
    print(i, row['ukbid'])
    try:
        sample_id = int(row['ukbid'])
        sample_idx = test_df[test_df['ukbid']==sample_id].index[0]
        hd5_model_fname = f'/mnt/disks/pdiachil-t1map/predictions3/holdout_results.h5'
        hd5_data_path = f'pdiachil/segmented-sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122-t1map/{sample_id}.hd5'
        hd5_segmentation_fname = glob.glob(f'/mnt/disks/pdiachil-t1map/prepared3/*__{sample_id}__*.h5')[0]
        blob = bucket.blob(hd5_data_path)
        blob.download_to_filename(f'{sample_id}.hd5')
        hd5_data_fname = f'{sample_id}.hd5'
        with h5py.File(hd5_model_fname, 'r') as hd5_model:
            with h5py.File(hd5_data_fname, 'r') as hd5_data:
                with h5py.File(hd5_segmentation_fname, 'r') as hd5_segmentation:
                    arr_model = hd5_model['predictions_argmax'][()][sample_idx]
                    for key in hd5_data['ukb_cardiac_mri']:
                        if ('t1map' in key) and ('sax' in key):
                            arr_data = hd5_data[f'ukb_cardiac_mri/{key}/2/instance_0'][()]
                    arr_data = arr_data[:, :-20, 0]
                    arr_data = cv2.resize(arr_data, (288, 384))
                    arr_segmentation = np.argmax(hd5_segmentation['segmentation_all'][()], axis=2)

                    arr_model_color = np.zeros((arr_model.shape[0], arr_model.shape[1], 3))
                    arr_segmentation_color = np.zeros((arr_segmentation.shape[0], arr_model.shape[1], 3))

                                      
                    skeletons = {}                    
                    for color, color_dic in colors.items():
                        if 'roi' in color.lower():
                            roi_arr_segmentation = arr_segmentation==color_dic["id"]
                            roi_arr_label =  label(roi_arr_segmentation)
                            label_regions = regionprops(roi_arr_label)
                            max_id = -1
                            max_area = 0
                            for iii, label_region in enumerate(label_regions):
                                if label_region.area > max_area:
                                    max_id = label_region.label
                                    max_area = label_region.area
                            roi_arr_segmentation_new = roi_arr_label==max_id
                            roi_arr_segmentation_diff = roi_arr_segmentation.astype(int) - roi_arr_segmentation_new.astype(int)
                            roi_arr_segmentation_diff = roi_arr_segmentation_diff > 0
                            # print(roi_arr_segmentation_diff.sum())
                            # f, ax = plt.subplots()
                            # ax.imshow(roi_arr_segmentation)
                            # ax.set_title(color.lower())                            
                            
                            # arr_segmentation_color[arr_segmentation==color_dic["id"], :] = ImageColor.getrgb(color_dic["color"])
                            arr_segmentation_color[roi_arr_segmentation] = ImageColor.getrgb(color_dic["color"])
                            # arr_segmentation[roi_arr_segmentation_diff] = 0
                        if "id2" in color_dic:
                            arr_model_color[arr_model==color_dic["id2"], :] = ImageColor.getrgb(color_dic["color"])                    
                            binary_model = arr_model==color_dic["id2"]
                            if 'cavity' in color.lower():
                                skeleton = erode_until(binary_model, 300)
                                regions = regionprops(skeleton.astype(int))
                                bubble = regions[0]
                                radius = int(np.sqrt(bubble.area / np.pi))
                                mycircle = disk(bubble.centroid, radius, shape=skeleton.shape)
                                template = np.zeros_like(skeleton)
                                template[mycircle] = 1
                                skeletons[color] = template
                                skeletons[color] = skeleton
                            else:
                                skeleton = skeletonize(binary_model, method='lee')
                                for i in range(5):
                                    result = generic_filter(skeleton, lineEnds, (3, 3))
                                    skeleton -= result*255
                                kernel = np.ones((4, 4), dtype=np.uint8)
                                skeletons[color] = np.logical_and(cv2.dilate(skeleton.astype(float), kernel, iterations=1), binary_model)
                                # skeletons[color] = erode_until(binary_model, 300)
                                # skeletons[color] = medial_axis(binary_model, return_distance=True)
                                # skeletons[color] = skeleton
                    f, ax = plt.subplots(1, 3)
                    f.set_size_inches(16, 9)
                    ax[0].imshow(arr_data, cmap='gray')
                    ax[1].imshow(arr_data, cmap='gray')
                    ax[2].imshow(arr_data, cmap='gray')
                    # ax[1].imshow(arr_model_color/255.0, alpha=0.5)
                    # ax[2].imshow(arr_model_color/255.0, alpha=0.5)
                    ax[2].imshow(ma.masked_array(skeletons['LV Cavity'], skeletons['LV Cavity']<0.5))
                    ax[2].imshow(ma.masked_array(skeletons['RV Cavity'], skeletons['RV Cavity']<0.5))
                    ax[2].imshow(ma.masked_array(skeletons['LV Free Wall'], skeletons['LV Free Wall']<0.5))
                    ax[2].imshow(ma.masked_array(skeletons['Interventricular Septum'], skeletons['Interventricular Septum']<0.5))
                    ax[0].imshow(arr_segmentation_color/255.0, alpha=0.5)
                    f.savefig(f'/home/pdiachil/projects/t1map/postprocessing/{sample_id}.png')
                    plt.close(f)
                    for region, binary_model in skeletons.items():
                        mean_model = np.median(arr_data[binary_model>0.5])
                        binary_segmentation = arr_segmentation==colors[region]['id']
                        mean_segmentation = np.median(arr_data[binary_segmentation>0.5])
                        means[region]['segmentation'].append(mean_segmentation)
                        means[region]['model'].append(mean_model)
                    means['sample_id'].append(sample_id)
                        # print(f'Region {region}: {mean_model} {mean_segmentation}')        
    except:
        pass          
    os.remove(f'{sample_id}.hd5')
# %%
import scipy.stats

df_dic = {}
for region in means:
    if 'sample_id' in region:
        df_dic[region] = means['sample_id']
    else:
        df_dic[f'{region}_model'] = means[region]['model']
        df_dic[f'{region}_segmentation'] = means[region]['segmentation']
df = pd.DataFrame(df_dic)
df = df[~df['sample_id'].isin(skip)]
df = df.dropna()
df
# %%
# df = df[df['LV Free Wall_model']>30.0]
i = 0
j = 0
f, ax = plt.subplots(2, 2)
f.set_size_inches(6, 6)
for region in means:
    if 'sample_id' in region:
        continue
    ax[i, j].plot(df[f'{region}_segmentation'], df[f'{region}_model'], 'ko', alpha=0.1)
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
    diff = np.abs(df[f'{region}_segmentation'] - df[f'{region}_model']) / df[f'{region}_segmentation']
    outliers = []
    outliers.extend(diff.argsort()[-2:])
    diff = np.abs((df[f'{region}_segmentation'] - np.mean(df[f'{region}_segmentation'])) / np.std(df[f'{region}_segmentation']))
    outliers.extend(diff.argsort()[-2:])
    diff = np.abs((df[f'{region}_model'] - np.mean(df[f'{region}_model'])) / np.std(df[f'{region}_model']))
    outliers.extend(diff.argsort()[-2:])
    texts = []
    for outlier in list(set(outliers)):
        row = df.iloc[outlier]
        texts.append(ax[i, j].text(row[f'{region}_segmentation'], row[f'{region}_model'], str(int(row['sample_id'])), ha='center', va='center'))
    adjust_text(texts)
    r = scipy.stats.spearmanr(df[f'{region}_segmentation'], df[f'{region}_model']).correlation
    r = np.corrcoef(df[f'{region}_segmentation'], df[f'{region}_model'])[0, 1]
    ax[i, j].set_title(f'{region} r: {r:.2f}')    
    j += 1
    i = j // 2 + i
    j = j % 2
plt.tight_layout()

# %%
df.to_csv('performance_holdout_4px_shape.csv', index=False)
f.savefig('performance_holdout_4px_shape.png')
# %%
