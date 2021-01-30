# %%
import matplotlib.pyplot as plt
import skimage
import glob
import numpy as np
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from skimage.draw import line   
import tarfile
import imageio
from google.cloud import storage
import time
import sys
import os
import pandas as pd

# start_id = int(sys.argv[1])
# stop_id = int(sys.argv[2])

start_id = 0
stop_id = 5

manifest = open('/home/pdiachil/projects/manifests/lvot_diameters.csv')
results_dic = {'sample_id': [], 'instance': [], 'nset': [], 'frame': [], 'fname': []}
region_points = {'lvot': 1, 'aortic_root': 1, 'ascending_aorta': 6}

for region, n_centers in region_points.items():
    for n_center in range(n_centers):
        for end in ['0', '1']:
            for coor in ['r', 'c']:
                results_dic[f'{region}_{n_center}_{end}_{coor}'] = []

for pat_i, sample_id_line in enumerate(manifest):
    if pat_i < start_id:
        continue
    if pat_i == stop_id:
        break
    
    sample_dir, sample_file = os.path.split(sample_id_line)
    lvot_path = sample_dir.replace('gs://ml4cvd/', '')
    lvot_file = sample_file.replace('.overlay.tar.gz\n', '')

    sample_id, _, instance, nset = map(int, lvot_file.split('_'))
    print(pat_i, sample_id)
    start_time = time.time()
    storage_client = storage.Client('broad-ml4cvd')
    bucket = storage_client.get_bucket('ml4cvd')
    segmented_lvot = f'{lvot_path}/{sample_id}_20212_{instance}_{nset}.overlay.tar.gz'
    blob = bucket.blob(segmented_lvot)
    blob.download_to_filename('lvot.tar.gz')
    with tarfile.open('lvot.tar.gz', 'r:gz') as tar:   
        imgs = tar.getmembers()
        names = tar.getnames()
        for img_n, (img_info, img_name) in enumerate(zip(imgs, names)):
            results_dic['sample_id'].append(sample_id)
            results_dic['instance'].append(instance)
            results_dic['nset'].append(nset)
            results_dic['frame'].append(img_n)
            results_dic['fname'].append(img_name)
            img_file = tar.extractfile(img_info)
            img = imageio.imread(img_file)                
            img = np.asarray(img, dtype=int)
            img_bin = np.logical_or(img==11, img==7)
            img_bin = np.logical_or(img_bin, img==8)
            img_bin = np.logical_or(img_bin, img==6)
            skeleton = skeletonize(img_bin)
            skeleton = skimage.transform.rescale(skeleton, 2, order=0, preserve_range=True)
            img = skimage.transform.rescale(img, 2, order=0, preserve_range=True)
            img_bin = skimage.transform.rescale(img_bin, 2, order=0, preserve_range=True)
            arg_skeleton = np.argwhere(skeleton)
            
            boundary_normals = []
            normal_imgs = []
            centroids = []
            for col_name, col in zip(region_points, [11, 7, 8]):
                for n_center in range(region_points[col_name]):
                    for end in ['0', '1']:
                        for coor in ['r', 'c']:
                            results_dic[f'{col_name}_{n_center}_{end}_{coor}'].append(-1.0)
            # for col_name, col in zip(['ascending_aorta'], [8]):
                try:
                    centroids_col = []
                    img_lvot = np.zeros_like(img, dtype=int)
                    boundaries = find_boundaries(img==col, mode='inner')
                    img_lvot[:, :] = np.asarray(img==col, dtype=int)
                    img_lvot[:, :] = label(img_lvot)                            
            
                    props = regionprops(img_lvot)
                    max_area = 0
                    max_idarea = -1
                    for iprop, prop in enumerate(props):
                        if prop.area > max_area:
                            max_area = prop.area
                            max_idarea = iprop
                    img_lvot[:, :] = np.asarray(img_lvot == max_idarea+1, dtype=int)
                
                    props = regionprops(img_lvot)
                    centroid = np.array(list(map(int, props[0].centroid)))
                    
                    idx = np.argmin(np.linalg.norm(arg_skeleton-centroid, axis=1))                    

                    if region_points[col_name] > 1:
                        skeleton_aorta = np.argwhere(img_lvot+skeleton > 1.5)
                        idxs = np.linspace(16, len(skeleton_aorta)-16, 6, dtype=np.int)
                        for idx in idxs:
                            centroids_col.append(skeleton_aorta[idx])
                    else:
                        centroids_col.append(arg_skeleton[idx])
                          
                    normal_img = np.zeros_like(skeleton) 
                
       
                    for icentroid, centroid in enumerate(centroids_col):
                        normal_centroid_img = np.zeros_like(skeleton)
                
                        dist_centroid = np.linalg.norm(arg_skeleton-centroid, axis=1)                
                        arg_dist_centroid = np.where(np.logical_and(dist_centroid<20.0, dist_centroid>0.001))[0]

                        normals = np.zeros((len(arg_dist_centroid), 2))
                        for i, close_pt in enumerate(arg_dist_centroid):
                            coor = arg_skeleton[close_pt]
                            delta = coor - centroid
                            normals[i] = np.array([delta[1], -delta[0]]) / np.linalg.norm(delta)
                            if normals[i, 0] < 0:
                                normals[i] = -normals[i]
                        normals_mask = np.min(np.abs(normals), axis=1) > 0.001
                        normal = np.median(normals[normals_mask], axis=0)
                        px0 = np.array(list(map(int, centroid + normal*40+1)))
                        px1 = np.array(list(map(int, centroid - normal*40+1)))

                        normal = line(px0[0], px0[1], px1[0], px1[1])                        
                        normal_img[normal[0], normal[1]] = 1
                        normal_centroid_img[normal[0], normal[1]] = 1
                        normal_centroid_img = skimage.morphology.binary_dilation(normal_centroid_img, selem=np.ones((2,2),dtype=np.int))
                        boundary_normal = np.argwhere((boundaries.astype(int)+normal_centroid_img) > 1.5)
                        boundary_normal = boundary_normal[[0, -1]]
                        results_dic[f'{col_name}_{icentroid}_0_r'][-1] = boundary_normal[0, 0] / 2.0
                        results_dic[f'{col_name}_{icentroid}_0_c'][-1] = boundary_normal[0, 1] / 2.0
                        results_dic[f'{col_name}_{icentroid}_1_r'][-1] = boundary_normal[1, 0] / 2.0
                        results_dic[f'{col_name}_{icentroid}_1_c'][-1] = boundary_normal[1, 1] / 2.0
                        boundary_normals.append(boundary_normal)                
                except IndexError:
                    boundary_normal = -1.0*np.ones((2, 2))
                    boundary_normals.append(boundary_normal)
                    centroid = -1.0*np.ones((2,))
                    normal_img = np.zeros_like(skeleton)
                normal_imgs.append(normal_img)
                centroids += centroids_col
            f, ax = plt.subplots()
            f.set_size_inches(9, 9)
            ax.imshow(normal_imgs[0]*5
                      +normal_imgs[0]*5
                      +normal_imgs[0]*5
                      +skeleton*4+img, cmap='gray')
            for boundary_normal in boundary_normals:
                ax.plot(boundary_normal[:, 1], boundary_normal[:, 0], 'rx')
            for centroid in centroids:
                ax.plot(centroid[1], centroid[0], 'rx')
            f.savefig(f'{img_n}.png')
            plt.close(f)
        
        with imageio.get_writer(f'/home/pdiachil/projects/chambers/{sample_id}_{instance}_{nset}.gif', mode='I') as writer:
            for i in range(img_n):
                image = imageio.imread(f'{i}.png')
                writer.append_data(image)
    end_time = time.time()
    print(end_time - start_time)
# %%
results = pd.DataFrame(results_dic)
results.to_csv(f'/home/pdiachil/projects/chambers/lvot_diameter_{start_id}_{stop_id}.csv', index=False)
# %%
