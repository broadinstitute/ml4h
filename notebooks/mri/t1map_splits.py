# %%
import pandas as pd
import glob
import shutil

t1map_original = pd.read_csv('/home/pdiachil/segmentation_t1_id_list.csv')
# %%
t1map_validation = pd.read_csv('/home/pdiachil/projects/manifests/validation_set_ids.txt', sep='\t')
# %%
t1map_original[t1map_original['set']=='training'].to_csv('/home/pdiachil/segmentation_t1map_training.csv', index=False)
t1map_original[t1map_original['set']=='validation'].to_csv('/home/pdiachil/segmentation_t1map_holdout.csv', index=False)
# %%
validation_ids = []
for i, row in t1map_original[t1map_original['set']=='validation'].iterrows():
    validation_ids.append(row['ID'])

training_ids = []
for i, row in t1map_original[t1map_original['set']=='training'].iterrows():
    training_ids.append(row['ID'])

# %%
import shutil
for i in training_ids:
    if os.path.isfile(f'/home/pdiachil/projects/t1map/dataset/tmp/{i}_2_0.png.mask.png'):
        shutil.copyfile(f'/home/pdiachil/projects/t1map/dataset/tmp/{i}_2_0.png.mask.png',
                        f'/home/pdiachil/projects/t1map/dataset/training_labels/{i}_2_0.png.mask.png',
        )
    else:
        print(i)
# %%
print(*ids, sep=' ')
# %%
training_images = glob.glob('/home/pdiachil/projects/t1map/dataset/training/*.png')
training_images = set([int(t.split('/')[-1].replace('_2_0.png', '')) for t in training_images])
training_labels = glob.glob('/home/pdiachil/projects/t1map/dataset/training_labels/*.png')
training_labels = set([int(t.split('/')[-1].replace('_2_0.png.mask.png', '')) for t in training_labels])
training_ids = set(t1map_original[t1map_original['set']=='training']['ID'])
# %%
training_ids - training_images
# %%
training_labels - training_images
# %%
missing_annotations = sorted(list(training_ids - training_labels))
print(*missing_annotations, sep='\n')
# %%
inference_images = glob.glob('/home/pdiachil/projects/t1map/dataset/inference/*.png')
# %%
for training_id in training_ids:
    if os.path.isfile(f'/home/pdiachil/projects/t1map/dataset/inference/{training_id}_2_0.png'):
        shutil.move(f'/home/pdiachil/projects/t1map/dataset/inference/{training_id}_2_0.png',
                    f'/home/pdiachil/projects/t1map/dataset/inference_removed/{training_id}_2_0.png',
        )
    # if os.path.isfile(f'/home/pdiachil/projects/t1map/dataset/inference/{training_id}_3_0.png'):
    #     shutil.move(f'/home/pdiachil/projects/t1map/dataset/inference/{training_id}_3_0.png',
    #                 f'/home/pdiachil/projects/t1map/dataset/inference_removed/{training_id}_3_0.png',
    #     )
# %%
for validation_id in validation_ids:
    if os.path.isfile(f'/home/pdiachil/projects/t1map/dataset/inference/{validation_id}_2_0.png'):
        shutil.move(f'/home/pdiachil/projects/t1map/dataset/inference/{validation_id}_2_0.png',
                    f'/home/pdiachil/projects/t1map/dataset/inference_removed/{validation_id}_2_0.png',
        )
    # if os.path.isfile(f'/home/pdiachil/projects/t1map/dataset/inference/{validation_id}_3_0.png'):
    #     shutil.move(f'/home/pdiachil/projects/t1map/dataset/inference/{validation_id}_3_0.png',
    #                 f'/home/pdiachil/projects/t1map/dataset/inference_removed/{validation_id}_3_0.png',
    #     )
# %%
import imageio
import numpy as np
inference_images = glob.glob('/home/pdiachil/projects/t1map/dataset/inference/*.png')
mean_pixels = np.zeros(len(inference_images))
min_pixels = np.zeros(len(inference_images))
max_pixels = np.zeros(len(inference_images))

for i, inference_image in enumerate(inference_images):
    image = imageio.imread(inference_image)
    break

# %%
