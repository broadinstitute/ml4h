# %%
import h5py
import pandas as pd
import glob
import imageio
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
import os, sys
import blosc
import logging

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


def read_compressed(data_set: h5py.Dataset):
    shape = data_set.attrs['shape']
    return np.frombuffer(blosc.decompress(data_set[()]), dtype=np.uint8).reshape(shape)

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

storage_client = storage.Client('broad-ml4cvd')
bucket = storage_client.get_bucket('ml4cvd')

df_remaining = pd.read_csv('/home/pdiachil/ml/notebooks/mri/remaining_rois2.csv')
hd5 = h5py.File('/mnt/disks/pdiachil-t1map/predictions3/ML4H_mdrk_ukb__cardiac_t1_weighted__predictions__5e806c4c75fa47d59f3270711fc35106.h5', 'r')
# %%
means = defaultdict(list)
means['LV Free Wall'] = defaultdict(list)
means['Interventricular Septum'] = defaultdict(list)
means['LV Cavity'] = defaultdict(list)
means['RV Cavity'] = defaultdict(list)
means['Wall'] = defaultdict(list)


# %%

start_id = int(sys.argv[1])
stop_id = int(sys.argv[2])

npats = stop_id - start_id
for key, item in means.items():
    item['model'] = [np.nan]*npats
    item['model_iqr'] = [np.nan]*npats

means['sample_id'] = [np.nan]*npats
means['instance'] = [np.nan]*npats


for pat_i, row_diff in df_remaining.iterrows():
    if pat_i < start_id:
        continue
    if pat_i == stop_id:
        break
    try:
        row = row_diff['diff']
        sample_id, instance = map(int, row.split('_'))        
        means['sample_id'][pat_i - start_id] = sample_id
        means['instance'][pat_i - start_id] = instance
        hd5_data_path = f'pdiachil/segmented-sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122-t1map/{sample_id}.hd5'
        
        blob = bucket.blob(hd5_data_path)
        blob.download_to_filename(f'{sample_id}.hd5')
        hd5_data_fname = f'{sample_id}.hd5'
        with h5py.File(hd5_data_fname, 'r') as hd5_data:
            arr_model = read_compressed(hd5[f'{row}/pred_class'])
            best_mean = 0
            found_anything = False
            for key in hd5_data['ukb_cardiac_mri']:
                if ('t1map' in key) and ('sax' in key) and (str(instance) in hd5_data[f'ukb_cardiac_mri/{key}']):
                    found_anything = True
                    cur_data = hd5_data[f'ukb_cardiac_mri/{key}/{instance}/instance_0'][()]
                    cur_mean = np.mean(cur_data)
                    if cur_mean > best_mean:
                        best_mean = cur_mean
                        arr_data = cur_data
            if not(found_anything):
                logging.warning(f'Did not find anything for {sample_id}/{instance}')
                continue
            arr_data = arr_data[:, :-20, 0]
            arr_data = cv2.resize(arr_data, (288, 384))
            arr_model_color = np.zeros((arr_model.shape[0], arr_model.shape[1], 3))          
                                
            skeletons = {}                    
            for color, color_dic in colors.items():
                if "id2" in color_dic:
                    arr_model_color[arr_model==color_dic["id2"], :] = ImageColor.getrgb(color_dic["color"])                    
                    binary_model = arr_model==color_dic["id2"]
                    nregions, label_model, stats, centroids = cv2.connectedComponentsWithStats(binary_model.astype(np.uint8), 4, cv2.CV_32S)
                    if nregions < 2:
                        skeletons[color] = binary_model
                    else:
                        order_by_area = np.argsort(stats[:, cv2.CC_STAT_AREA])
                        binary_model = label_model==order_by_area[-2]
                    if 'cavity' in color.lower():
                        skeleton = erode_until(binary_model, 300)
                        skeletons[color] = skeleton
                    else:
                        skeleton = skeletonize(binary_model, method='lee')
                        for i in range(5):
                            result = generic_filter(skeleton, lineEnds, (3, 3))
                            skeleton -= result*255
                        kernel = np.ones((3, 3), dtype=np.uint8)
                        skeletons[color] = np.logical_and(cv2.dilate(skeleton.astype(float), kernel, iterations=1), binary_model)
            skeletons['Wall'] = skeletons['LV Free Wall'] + skeletons['Interventricular Septum']
                        
            f, ax = plt.subplots(1, 3)
            f.set_size_inches(16, 9)
            ax[0].imshow(arr_data, cmap='gray')
            ax[1].imshow(arr_data, cmap='gray')
            ax[2].imshow(arr_data, cmap='gray')
            ax[1].imshow(arr_model_color/255.0, alpha=0.5)
            # ax[2].imshow(arr_model_color/255.0, alpha=0.5)
            ax[2].imshow(ma.masked_array(skeletons['LV Cavity'], skeletons['LV Cavity']<0.5))
            ax[2].imshow(ma.masked_array(skeletons['RV Cavity'], skeletons['RV Cavity']<0.5))
            ax[2].imshow(ma.masked_array(skeletons['LV Free Wall'], skeletons['LV Free Wall']<0.5))
            ax[2].imshow(ma.masked_array(skeletons['Interventricular Septum'], skeletons['Interventricular Septum']<0.5))
            
            f.savefig(f'/home/pdiachil/projects/t1map/inference/{sample_id}_{instance}.png')
            plt.close(f)
            for region, binary_model in skeletons.items():
                select_data = arr_data[binary_model>0.5]
                mean_model = np.median(select_data)
                means[region]['model'][pat_i - start_id] = mean_model
                iqr = np.percentile(select_data, 75) - np.percentile(select_data, 25)
                
                if not(np.isnan(mean_model)):
                    if 'Cavity' in region:
                        percentile_model = np.median(select_data[select_data > (np.percentile(select_data, 25) - 1.5 * iqr)])
                    else:
                        percentile_model = np.median(select_data[select_data < (np.percentile(select_data, 75) + 1.5 * iqr)])
                else:
                    percentile_model = np.nan
                means[region]['model_iqr'][pat_i - start_id] = percentile_model
            
    except:
        pass          
    os.remove(f'{sample_id}.hd5')

skip = []
df_dic = {}
for region in means:
    if ('sample_id' in region) or ('instance' in region):
        df_dic[region] = means[region]
    else:
        df_dic[f'{region}_model'] = means[region]['model']
        df_dic[f'{region}_model_iqr'] = means[region]['model_iqr']
df = pd.DataFrame(df_dic)
# df = df[~df['sample_id'].isin(skip)]
# df = df.dropna()

df.to_csv(f'/home/pdiachil/projects/t1map/inference/t1map_inference_remaining2_{start_id}_{stop_id}.csv', index=False)
