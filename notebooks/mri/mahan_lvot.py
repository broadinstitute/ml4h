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

# %%
sample_ids=[1000107, 1000169, 1000336, 1000387]
for sample_id in sample_ids:
    start_time = time.time()
    storage_client = storage.Client('broad-ml4cvd')
    bucket = storage_client.get_bucket('ml4cvd')
    segmented_lvot = f'jamesp/annotation/lvot/v20201210/apply/v1/output/overlay/{sample_id}_20212_2_0.overlay.tar.gz'
    blob = bucket.blob(segmented_lvot)
    blob.download_to_filename('lvot.tar.gz')
    with tarfile.open('lvot.tar.gz', 'r:gz') as tar:   

        imgs = tar.getmembers()
        for img_n, img_filename in enumerate(imgs):
            img_file = tar.extractfile(img_filename)
            img = imageio.imread(img_file)
                
            img = np.asarray(img, dtype=int)
            # img = skimage.transform.rescale(img, 2, order=0, preserve_range=True)
            # img = np.asarray(img, dtype=int)

            img_bin = np.logical_or(img==11, img==7)
            img_bin = np.logical_or(img_bin, img==8)
            img_bin = np.logical_or(img_bin, img==6)
            skeleton = skeletonize(img_bin)
            skeleton = skimage.transform.rescale(skeleton, 2, order=0, preserve_range=True)
            img = skimage.transform.rescale(img, 2, order=0, preserve_range=True)
            img_bin = skimage.transform.rescale(img_bin, 2, order=0, preserve_range=True)
            arg_skeleton = np.argwhere(skeleton)
            # arg_skeleton[:, 0] = np.convolve(arg_skeleton[:, 0], np.ones(10), 'same')/10
            # arg_skeleton[:, 1] = np.convolve(arg_skeleton[:, 1], np.ones(10), 'same')/10
            boundary_normals = []
            normal_imgs = []
            centroids = []
            for col in [11, 7, 8]:
                boundaries = find_boundaries(img==col, mode='inner')
                img_lvot = np.asarray(img==col, dtype=int)
                props = regionprops(img_lvot)
                centroid = np.array(list(map(int, props[0].centroid)))
                
                idx = np.argmin(np.linalg.norm(arg_skeleton-centroid, axis=1))
                centroid = arg_skeleton[idx]
                centroids.append(centroid)
                # f, ax = plt.subplots()
                # ax.imshow(skeleton)
                # ax.plot(centroid[1], centroid[0], 'rx')
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
                normal_img = np.zeros_like(skeleton)
                normal_img[normal[0], normal[1]] = 1
                normal_img = skimage.morphology.binary_dilation(normal_img, selem=np.ones((2,2),dtype=np.int))
                normal_imgs.append(normal_img)
                boundary_normal = np.argwhere((boundaries.astype(int)+normal_img) > 1.5)
                boundary_normals.append(boundary_normal[[0, -1]])
            f, ax = plt.subplots()
            f.set_size_inches(9, 9)
            ax.imshow(normal_imgs[0]*5
                      +normal_imgs[1]*5
                      +normal_imgs[2]*5
                      +skeleton*4+img, cmap='gray')
                        
            ax.plot(boundary_normals[0][:, 1], boundary_normals[0][:, 0], 'rx')            
            ax.plot(boundary_normals[1][:, 1], boundary_normals[1][:, 0], 'rx')
            ax.plot(boundary_normals[2][:, 1], boundary_normals[2][:, 0], 'rx')
            ax.plot(centroids[0][1], centroids[0][0], 'rx')
            ax.plot(centroids[1][1], centroids[1][0], 'rx')
            ax.plot(centroids[2][1], centroids[2][0], 'rx')
            f.savefig(f'{img_n}.png')
        
        with imageio.get_writer(f'/home/pdiachil/projects/chambers/{sample_id}.gif', mode='I') as writer:
            for i in range(img_n):
                image = imageio.imread(f'{i}.png')
                writer.append_data(image)
    end_time = time.time()
    print(end_time - start_time)
# %%

# %%
