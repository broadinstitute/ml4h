# %%
from statsmodels.multivariate.pca import PCA
import pandas as pd

# %%
# df_rv_sax_v20201102_lax_v20201006 = pd.read_csv('/home/pdiachil/projects/surface_reconstruction/all_SAX_RV_processed_v20201102.csv')
# df_rv_sax_v20201124_lax_v20201122 = pd.read_csv('/home/pdiachil/projects/surface_reconstruction/sax-v20201124-lax-v20201122/all_RV_processed_sax_v20201124_lax_v20201122.csv')
df_rv_sax_v20201124_lax_v20201122_separation = pd.read_csv('/home/pdiachil/projects/surface_reconstruction/sax-v20201124-lax-20201122-petersen-separation/all_RV_processed_separation_v20201124_v20201122.csv')
df_rv_sax_v20201203_lax_v20201122 = pd.read_csv('/home/pdiachil/projects/surface_reconstruction/sax-ml4h-v20201203-v20201122-petersen/all_RV_processed_ml4h_v20201203_v20201122.csv')
# for t in range(50):
#     df_rv_sax_v20201124_lax_v20201122_separation[f'RV_poisson_{t}_v20201124_v20201122_sep'] = df_rv_sax_v20201124_lax_v20201122_separation[f'RV_poisson_{t}']
#     df_rv_sax_v20201124_lax_v20201122_separation = df_rv_sax_v20201124_lax_v20201122_separation.drop(columns=f'RV_poisson_{t}')

labels = ['v20201124_v20201122', 'v20201203_v20201122']

surfaces = ['poisson', 'poisson']

dfs_rv = [df_rv_sax_v20201124_lax_v20201122_separation, df_rv_sax_v20201203_lax_v20201122]

for i, df_rv in enumerate(dfs_rv):
    dfs_rv[i] = df_rv[df_rv['sample_id']!=-1]

df_all = dfs_rv[0]
label_left = labels[0]
for df_rv, label_right in zip(dfs_rv[1:], labels[1:]):
    df_all = df_all.merge(df_rv, on='sample_id', suffixes=(f'_{label_left}', f'_{label_right}'))
    label_left = label_right



# %%
keys = []
for label, surface in zip(labels, surfaces):
    keys.append([f'RV_{surface}_{d}_{label}' for d in range(50)])


for label, surface, key in zip(labels, surfaces, keys):
    df_all[f'RVEDV_{surface}_{label}'] = df_all[key].max(axis=1)
    df_all[f'RVESV_{surface}_{label}'] = df_all[key].min(axis=1)
    df_all[f'RVEF_{surface}_{label}'] = (df_all[f'RVEDV_{surface}_{label}'] - df_all[f'RVESV_{surface}_{label}']) / df_all[f'RVEDV_{surface}_{label}'] * 100.
# %%
# df_all['RVEDV_poisson_mean'] = 0.5*(df_all[f'RVEDV_poisson_{labels[0]}'] + df_all[f'RVEDV_poisson_{labels[1]}'])
# df_all['RVESV_poisson_mean'] = 0.5*(df_all[f'RVESV_poisson_{labels[0]}'] + df_all[f'RVESV_poisson_{labels[1]}'])
# df_all['RVEF_poisson_mean'] = (df_all[f'RVEDV_poisson_mean'] - df_all['RVESV_poisson_mean']) / df_all[f'RVEDV_poisson_mean'] * 100.

df_petersen = pd.read_csv('returned_lv_mass.tsv', sep='\t')
df_all_petersen = df_petersen.merge(df_all, on='sample_id')


# %%
import numpy as np
for label, surface in zip(labels, surfaces):
    df_all_petersen[f'err_{label}'] = np.abs(df_all_petersen[f'RVEF_{surface}_{label}'] - df_all_petersen[f'RVEF']) / df_all_petersen[f'RVEF']

df_all_petersen_sorted = df_all_petersen.sort_values(by=[f'err_{label}' for label in labels], ascending='False').dropna()

# df_all_petersen_sorted[['sample_id', 'RVEF', 'RVEF_poisson_v20201102_v20201006']].corr()
# %%
import scipy.stats
import matplotlib.pyplot as plt
# labels += ['mean']
# surfaces += ['poisson']
subset = [f'RVEDV_poisson_{label}' for label in labels]
subset += [f'RVESV_poisson_{label}' for label in labels]
subset += [f'RVEF_poisson_{label}' for label in labels]
subset += ['RVEDV', 'RVESV', 'RVEF']

df_inuse = df_all_petersen.dropna(subset=subset)
df_inuse.to_csv('/home/pdiachil/projects/surface_reconstruction/sax-ml4h-v20201203-v20201122-petersen/all_RV_processed_ml4h_v20201203_v20201122_petersen.csv', index=False)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 20]
f, ax = plt.subplots(3, (len(subset)-3)//3)
f.set_size_inches((len(subset)-3)//3*2.5, 6)



for i, (meas, extent) in enumerate(zip(['RVEDV', 'RVESV', 'RVEF'], [400, 200, 100])):
    for j, (surface, label) in enumerate(zip(surfaces, labels)):
        ax[i, j].hexbin(df_inuse[f'{meas}_{surface}_{label}'], df_inuse[meas],
                        extent=(0, extent, 0, extent), mincnt=1, cmap='gray')
        ax[i, j].set_aspect('equal')
        ax[i, j].plot([0, 400], [0, 400], color='k')
        if i == 2:
            ax[i, j].set_xlabel(f'{label}')
        else:
            ax[i, j].set_xlabel('')
        if j == 0:
            ax[i, j].set_ylabel(meas)
        else:
            ax[i, j].set_ylabel('')

        ax[i, j].set_xlim([0, extent])
        ax[i, j].set_ylim([0, extent])
        pearson = scipy.stats.pearsonr(df_inuse[f'{meas}_{surface}_{label}'], df_inuse[meas])[0]
        spearman = scipy.stats.spearmanr(df_inuse[f'{meas}_{surface}_{label}'], df_inuse[meas])[0]
        ax[i, j].set_title(f'n={len(df_inuse)}, r={pearson:.2f}')

plt.tight_layout()
f.savefig('RV_petersen_all_20201204.png', dpi=500)

df_inuse.to_csv('/home/pdiachil/df_inuse_petersen.csv', index=False)
# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = [f'RVEDV_poisson_{label}' for label in labels]
subset += [f'RVESV_poisson_{label}' for label in labels]
subset += [f'RVEF_poisson_{label}' for label in labels]

df_inuse = df_all.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 20]

df_inuse.to_csv('/home/pdiachil/df_inuse.csv', index=False)
# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = [f'RVEDV_poisson_{label}' for label in labels]
subset += [f'RVESV_poisson_{label}' for label in labels]
subset += [f'RVEF_poisson_{label}' for label in labels]
subset += ['RVEDV', 'RVESV', 'RVEF']

df_inuse = df_all_petersen.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 400]
    df_inuse = df_inuse[df_inuse[feat] > 10]

# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['SAXLV_poisson_max', 'SAXLVM_poisson_max', 'LV_poisson_max', 'LVM_poisson_max', 'RV_poisson_max']
df_inuse = df_all.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 400]
    df_inuse = df_inuse[df_inuse[feat] > 10]
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_inuse['SAXLV_poisson_max'], df_inuse['LV_poisson_max'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'SAXLV_poisson_max (ml)')
ax.set_ylabel('LV_poisson_max (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([0, 400])
ax.set_ylim([0, 400])
pearson = scipy.stats.pearsonr(df_inuse['SAXLV_poisson_max'], df_inuse['LV_poisson_max'])[0]
ax.set_title(f'n={len(df_inuse)}, r={pearson:.2f}')
plt.tight_layout()

# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['SAXLV_poisson_max', 'SAXLVM_poisson_max', 'LV_poisson_max', 'LVM_poisson_max', 'RV_poisson_max']
df_inuse = df_all.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_inuse['SAXLVM_poisson_max'], df_inuse['LVM_poisson_max'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'SAXLVM_poisson_max (ml)')
ax.set_ylabel('LVM_poisson_max (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([0, 400])
ax.set_ylim([0, 400])
pearson = scipy.stats.pearsonr(df_inuse['SAXLV_poisson_max'], df_inuse['LV_poisson_max'])[0]
ax.set_title(f'n={len(df_inuse)}, r={pearson:.2f}')
plt.tight_layout()

# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['SAXLV_poisson_max', 'SAXLVM_poisson_max', 'LV_poisson_max', 'LVM_poisson_max', 'LVEDV']
df_inuse = df_all_petersen.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots(1, 3)
f.set_size_inches(9, 3)
for i, feat in enumerate(['LV_poisson_max', 'SAXLV_poisson_max', 'mean_LV_poisson_max']):
    ax[i].hexbin(df_inuse[feat], df_inuse['LVEDV'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
    ax[i].set_aspect('equal')
    ax[i].plot([0, 400], [0, 400], color='k')
    ax[i].set_xlabel(f'{feat} (ml)')
    ax[i].set_ylabel('LVEDV (ml)')
    ax[i].set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax[i].set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax[i].set_xlim([0, 400])
    ax[i].set_ylim([0, 400])
    pearson = scipy.stats.pearsonr(df_inuse[feat], df_inuse['LVEDV'])[0]
    ax[i].set_title(f'n={len(df_inuse)}, r={pearson:.2f}')
    plt.tight_layout()
f.savefig('petersen_LV_sax_lax.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
df_all_petersen['RV_poisson_ef'] = (df_all_petersen['RV_poisson_max']-df_all_petersen['RV_poisson_min'])/df_all_petersen['RV_poisson_max']
subset = ['RV_poisson_max', 'RV_poisson_min', 'RV_poisson_ef']
subset_y = ['RVEDV', 'RVESV', 'RVEF']
df_inuse = df_all_petersen.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots(1, 3)
f.set_size_inches(9, 3)
for i, (feat, feat_y) in enumerate(zip(subset, subset_y)):
    ax[i].hexbin(df_inuse[feat], df_inuse[feat_y], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
    ax[i].set_aspect('equal')
    ax[i].plot([0, 400], [0, 400], color='k')
    ax[i].set_xlabel(f'{feat} (ml)')
    ax[i].set_ylabel(feat_y)
    ax[i].set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax[i].set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax[i].set_xlim([0, 400])
    ax[i].set_ylim([0, 400])
    pearson = scipy.stats.pearsonr(df_inuse[feat], df_inuse['LVEDV'])[0]
    ax[i].set_title(f'n={len(df_inuse)}, r={pearson:.2f}')
    plt.tight_layout()
f.savefig('petersen_RV_sax.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['SAXLV_poisson_max', 'SAXLVM_poisson_max', 'LV_poisson_max', 'LVM_poisson_max', 'LVEDV', 'LVM']
df_inuse = df_all_petersen.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots(1, 3)
f.set_size_inches(9, 3)
for i, feat in enumerate(['LVM_poisson_max', 'SAXLVM_poisson_max', 'mean_LVM_poisson_max']):
    ax[i].hexbin(df_inuse[feat], df_inuse['LVM'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
    ax[i].set_aspect('equal')
    ax[i].plot([0, 400], [0, 400], color='k')
    ax[i].set_xlabel(f'{feat} (ml)')
    ax[i].set_ylabel('LVM (ml)')
    ax[i].set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax[i].set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax[i].set_xlim([0, 400])
    ax[i].set_ylim([0, 400])
    pearson = scipy.stats.pearsonr(df_inuse[feat], df_inuse['LVM'])[0]
    ax[i].set_title(f'n={len(df_inuse)}, r={pearson:.2f}')
    plt.tight_layout()

f.savefig('petersen_LVM_sax_lax.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['SAXLV_poisson_max', 'SAXLVM_poisson_max', 'LV_poisson_max', 'LVM_poisson_max', 'RV_poisson_max', 'LVEDV', 'RVEDV', 'LVM']
df_inuse = df_all_petersen.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots()
f.set_size_inches(9, 3)
for i, feat in enumerate(['RV_poisson_max']):
    ax.hexbin(df_inuse[feat], df_inuse['RVEDV'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
    ax.set_aspect('equal')
    ax.plot([0, 400], [0, 400], color='k')
    ax.set_xlabel(f'{feat} (ml)')
    ax.set_ylabel('RVEDV (ml)')
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    pearson = scipy.stats.pearsonr(df_inuse[feat], df_inuse['RVEDV'])[0]
    ax.set_title(f'n={len(df_inuse)}, r={pearson:.2f}')
    plt.tight_layout()

# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['SAXLV_poisson_max', 'SAXLVM_poisson_max', 'LV_poisson_max', 'LVM_poisson_max', 'RV_poisson_max', 'LVEDV', 'RVEDV', 'LVM', 'LA_Biplan_vol_max', 'LA_poisson_max']
df_inuse = df_all_petersen.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots()
f.set_size_inches(9, 3)
for i, feat in enumerate(['RV_poisson_max']):
    ax.hexbin(df_inuse[feat], df_inuse['RVEDV'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
    ax.set_aspect('equal')
    ax.plot([0, 400], [0, 400], color='k')
    ax.set_xlabel(f'{feat} (ml)')
    ax.set_ylabel('RVEDV (ml)')
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 400])
    pearson = scipy.stats.pearsonr(df_inuse[feat], df_inuse['RVEDV'])[0]
    ax.set_title(f'n={len(df_inuse)}, r={pearson:.2f}')
    plt.tight_layout()



# %%
# Let's use pixelcounts as well
df_pixelcount = pd.read_csv('/home/pdiachil/projects/annotation/jamesp/sax/tmp.tsv', sep='\t', usecols=['ID6_RV_Cavity',
       'ID6_RV_Cavity_5_thresholded', 'ID6_RV_Cavity_components',
       'ID6_RV_Cavity_5_thresholded_components', 'ID6_RV_Cavity_LongAxisAngle',
       'ID6_RV_Cavity_LongAxisPixels', 'ID6_RV_Cavity_ShortAxisPixels',
       'ID6_RV_Cavity_Eccentricity', 'ID6_RV_Cavity_CentroidX',
       'ID6_RV_Cavity_CentroidY', 'ID6_RV_Cavity_TopLeftX',
       'ID6_RV_Cavity_TopLeftY', 'ID6_RV_Cavity_BottomRightX',
       'ID6_RV_Cavity_BottomRightY'])
# %%
df_pixelcount = pd.read_csv('/home/pdiachil/projects/annotation/jamesp/sax/pixelcount.tsv', sep='\t', usecols=['dicom', 'ID6_RV_Cavity', 'ID6_RV_Cavity_5_thresholded'])
# %%
df_manifest = pd.read_csv('/home/pdiachil/projects/annotation/jamesp/sax/manifest.tsv', sep='\t', usecols=['sample_id', 'dicom_file', 'px_height_mm', 'px_width_mm', 'series', 'instance_number'])

# %%
df_pixelcount_manifest = df_manifest.merge(df_pixelcount, left_on='dicom_file', right_on='dicom')
# %%
df_iou = pd.read_csv('/home/pdiachil/projects/surface_reconstruction/all_RV_iou_v20201102.csv')
# %%
df_iou_long = pd.DataFrame
for i, row in df_iou.iterrows():

# %%
df_sam = pd.read_csv('/home/pdiachil/projects/surface_reconstruction/lvm_from_inlinevf_and_ml4h_segmentation.tsv', sep='\t')
df_sam_all = df_sam.merge(df_all, on='sample_id')
df_sam_all_petersen = df_sam_all.merge(df_petersen, on='sample_id')

# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['SAXLV_poisson_max', 'SAXLVM_poisson_max', 'LV_poisson_max', 'LVM_poisson_max', 'LVEDV', 'LVM']
df_inuse = df_sam_all_petersen.dropna(subset=subset)
for feat in subset:
    df_inuse = df_inuse[df_inuse[feat] < 400]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots(1, 5)
f.set_size_inches(18, 3)
for i, feat in enumerate(['LVM_poisson_max', 'SAXLVM_poisson_max', 'mean_LVM_poisson_max', 'inlinevf_lvm', 'ml4h_lvm']):
    ax[i].hexbin(df_inuse[feat], df_inuse['LVM'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
    ax[i].set_aspect('equal')
    ax[i].plot([0, 400], [0, 400], color='k')
    ax[i].set_xlabel(f'{feat} (ml)')
    ax[i].set_ylabel('LVM (ml)')
    ax[i].set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax[i].set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax[i].set_xlim([0, 400])
    ax[i].set_ylim([0, 400])
    pearson = scipy.stats.pearsonr(df_inuse[feat], df_inuse['LVM'])[0]
    ax[i].set_title(f'n={len(df_inuse)}, r={pearson:.2f}')
    plt.tight_layout()

f.savefig('petersen_sam_allLVM_sax_lax.png', dpi=500)
# %%
# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['RVEDV_poisson_20201102', 'RVEDV_poisson_20201119',
          'RVESV_poisson_20201102', 'RVESV_poisson_20201119',
          'RVEDV_poisson_20201102_ml4h', 'RVESV_poisson_20201102_ml4h',
          'RVEDV_poisson_20201119_noshift', 'RVESV_min_20201119_noshift',
          'RV_discs_max_20201122', 'RV_discs_min_20201122',
          'RV_discs_max_20201122', 'RV_discs_min_20201122'
          'RVEDV', 'RVESV', 'RVEF']
df_inuse = df_all_petersen.dropna(subset=subset)
for feat in subset:
    if 'RVEF' in feat:
        continue
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots(5, 3)
f.set_size_inches(9, 12)
for j, version in enumerate(['20201102', '20201102_ml4h', '20201119', '20201119_noshift', '20201122']):
    poi_or_discs = 'discs' if '20201122' in version else 'poisson'
    for i, (feat, feat_petersen, extent) in enumerate(zip([f'RV_{poi_or_discs}_max_{version}', f'RV_{poi_or_discs}_min_{version}', f'RVEF_{poi_or_discs}_{version}'],
                                                           ['RVEDV', 'RVESV', 'RVEF'],
                                                           [400, 200, 100],
                                                           )):
        ax[j, i].hexbin(df_inuse[feat], df_inuse[feat_petersen], extent=(0, extent, 0, extent), mincnt=1, cmap='gray')
        ax[j, i].set_aspect('equal')
        ax[j, i].plot([0, 400], [0, 400], color='k')
        ax[j, i].set_xlabel(f'{feat}')
        ax[j, i].set_ylabel(f'{feat_petersen}')
        #ax[i].set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        #ax[i].set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        ax[j, i].set_xlim([0, extent])
        ax[j, i].set_ylim([0, extent])
        pearson = scipy.stats.pearsonr(df_inuse[feat], df_inuse[feat_petersen])[0]
        spearman = scipy.stats.spearmanr(df_inuse[feat], df_inuse[feat_petersen])[0]
        ax[j, i].set_title(f'n={len(df_inuse)}, r={pearson:.2f}, s={spearman:.2f}')
plt.tight_layout()
f.savefig('RV_petersen_all_20201124.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
subset = ['RV_poisson_max_20201102', 'RV_poisson_min_20201102',
          'RV_discs_max_20201122', 'RV_discs_min_20201122',
          'RV_discs_max_20201122_dice', 'RV_discs_min_20201122_dice',
          'RVEDV', 'RVESV', 'RVEF']
df_inuse = df_all_petersen.dropna(subset=subset)
for feat in subset:
    if 'RVEF' in feat:
        continue
    df_inuse = df_inuse[df_inuse[feat] < 500]
    df_inuse = df_inuse[df_inuse[feat] > 5]
f, ax = plt.subplots(3, 3)
f.set_size_inches(9, 9)
for j, (version, label) in enumerate(['20201102', '20201122', '20201122_dice']):
    poi_or_discs = 'discs' if '20201122' in version else 'poisson'
    for i, (feat, feat_petersen, extent) in enumerate(zip([f'RV_{poi_or_discs}_max_{version}', f'RV_{poi_or_discs}_min_{version}', f'RVEF_{poi_or_discs}_{version}'],
                                                           ['RVEDV', 'RVESV', 'RVEF'],
                                                           [400, 200, 100]
                                                           )):
        ax[j, i].hexbin(df_inuse[feat], df_inuse[feat_petersen], extent=(0, extent, 0, extent), mincnt=1, cmap='gray')
        ax[j, i].set_aspect('equal')
        ax[j, i].plot([0, 400], [0, 400], color='k')
        ax[j, i].set_xlabel(f'{feat}')
        ax[j, i].set_ylabel(f'{feat_petersen}')
        #ax[i].set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        #ax[i].set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
        ax[j, i].set_xlim([0, extent])
        ax[j, i].set_ylim([0, extent])
        pearson = scipy.stats.pearsonr(df_inuse[feat], df_inuse[feat_petersen])[0]
        spearman = scipy.stats.spearmanr(df_inuse[feat], df_inuse[feat_petersen])[0]
        ax[j, i].set_title(f'n={len(df_inuse)}, r={pearson:.2f}, s={spearman:.2f}')
plt.tight_layout()
f.savefig('RV_petersen_all_20201124.png', dpi=500)
# %%
import numpy as np
df_inuse['RVEF_diff'] = np.abs(df_inuse['RVEF']-df_inuse['RVEF_discs_20201122'])
df_inuse_sort = df_inuse.sort_values(by=['RVEF_diff'], ascending=False)
df_inuse_sort.to_csv('df_worse_sorted_sax_v20201116_lax_v20201119.csv')
# %%
import h5py
import vtk
import imageio
from matplotlib.animation import FuncAnimation, PillowWriter
from vtk.util import numpy_support as ns
from google.cloud import storage
from ml4h.tensormap.ukb.mri_vtk import _mri_hd5_to_structured_grids, _mri_tensor_4d
from ml4h.defines import MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_2CH_SEGMENTED_CHANNEL_MAP, MRI_LAX_3CH_SEGMENTED_CHANNEL_MAP, MRI_FRAMES

def plot_all_annotations(row, out_name):
    sample_id = row['sample_id']
    views = ['lax_4ch']
    views += [f'sax_b{d}' for d in range(1, 13)]
    # views = [f'sax_b{d}' for d in range(1, 10)]
    view_format_string = 'cine_segmented_{view}'
    annot_format_string = 'cine_segmented_{view}_annotated'
    annot_time_format_string = 'cine_segmented_{view}_annotated_{t}'
    MRI_SAX_SEGMENTED_CHANNEL_MAP = {'RV_cavity': 6, 'LV_cavity': 5, 'LV_free_wall': 3, 'interventricular_septum': 2}
    # MRI_SAX_SEGMENTED_CHANNEL_MAP = {'RV_cavity': 5, 'LV_cavity': 4, 'LV_free_wall': 3, 'interventricular_septum': 2}
    channels = [
        [
            MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP['RV_cavity'],
        ],
    ]
    channels[0] += [MRI_SAX_SEGMENTED_CHANNEL_MAP['RV_cavity'] for d in range(1, 13)]
    chambers = ['RV']

    hd5 = f'/mnt/disks/segmented-sax-v20201116-lax-v20201119-petersen/2020-11-20/{sample_id}.hd5'
    hd5 = f'/mnt/disks/segmented-sax-lax-v20201102/2020-11-02/{sample_id}.hd5'
    sample_id = hd5.split('/')[-1].replace('.hd5', '')

    annot_datasets = []
    orig_datasets = []

    with h5py.File(hd5) as ff_trad:
        for iv, view in enumerate(views):
            if view_format_string.format(view=view) not in ff_trad['ukb_cardiac_mri']:
                views = views[:iv]
                channels[0] = channels[0][:iv]
                break
            annot_datasets.append(
                _mri_hd5_to_structured_grids(
                    ff_trad, annot_format_string.format(view=view),
                    view_name=view_format_string.format(view=view),
                    concatenate=True, annotation=True,
                    save_path=None, order='F',
                )[0],
            )
            orig_datasets.append(
                _mri_hd5_to_structured_grids(
                    ff_trad, view_format_string.format(view=view),
                    view_name=view_format_string.format(view=view),
                    concatenate=False, annotation=False,
                    save_path=None, order='F',
                )[0],
            )

    f, ax = plt.subplots(3, 7, constrained_layout=True)
    f.set_size_inches(16, 9)
    gs = ax[2, 0].get_gridspec()
    # remove the underlying axes
    for ax0 in ax[2, :]:
        ax0.remove()
    axbig = f.add_subplot(gs[2, :])
    for idata in range(14):
        ax[idata//7, idata%7].set_xticklabels([])
        ax[idata//7, idata%7].set_yticklabels([])

    axbig.plot(range(50), row[keys_rv_20201122], label='discs_sax_20201116_lax_20201122', linewidth=3)
    axbig.plot(range(50), row[keys_rv_20201119], label='poisson_sax_20201116_lax_20201119', linewidth=3)
    axbig.plot(range(50), row[keys_rv_20201102], label='poisson_sax_20201102_lax_20200816', linewidth=3)
    axbig.plot([0.0, 50.0], [row['RVEDV'], row['RVEDV']], 'k--')
    axbig.plot([0.0, 50.0], [row['RVESV'], row['RVESV']], 'k--')
    axbig.legend()
    axbig.set_xticklabels([])
    with imageio.get_writer(f'{out_name}_{sample_id}.gif', mode='I') as writer:
        for t in range(0, 50, 5):
            for idata, (orig_dataset, annot_dataset, view, channel) in enumerate(zip(orig_datasets, annot_datasets, views, channels[0])):
                arr_orig = ns.vtk_to_numpy(orig_dataset.GetCellData().GetArray(f'cine_segmented_{view}_{t}'))
                arr_annot = ns.vtk_to_numpy(annot_dataset.GetCellData().GetArray(f'cine_segmented_{view}_annotated_{t}'))
                orig_extent = orig_dataset.GetExtent()
                img_orig = np.zeros((256, 256))
                img_orig[:orig_extent[1], :orig_extent[3]] = arr_orig.reshape(orig_extent[1], orig_extent[3])
                img_annot = arr_annot.reshape(256, 256)
                ax[idata//7, idata%7].imshow(img_orig, cmap='gray', vmin=0, vmax=255)
                img_annot_ma = np.ma.masked_array(data=img_annot,
                                                  mask=(img_annot != channel))
                ax[idata//7, idata%7].imshow(img_annot_ma, alpha=0.5)
                ax[idata//7, idata%7].set_title(view)
            f.suptitle(f'RVEF: {row["RVEF"]}, RVEF_discs: {row["RVEF_discs_20201122"]:.1f}')
            f.savefig(f'tmp_{t}.png', dpi=100)

            image = imageio.imread(f'tmp_{t}.png')
            writer.append_data(image)

for i in range(5):
    plot_all_annotations(df_inuse_sort.iloc[i], out_name='rv_discs_worse')
for i in range(5):
    plot_all_annotations(df_inuse_sort.iloc[-i-1], out_name='rv_discs_best')
# %%
import matplotlib.pyplot as plt
discs = pd.read_csv('/home/pdiachil/ml/RV_discs_sax_v20201116_lax_v20201122_4_5.csv')

row = df_inuse_sort[df_inuse_sort['sample_id']==2923883].iloc[0]
f, ax = plt.subplots()

ax.plot(range(50), row[keys_rv_20201119])
ax.plot(range(50), discs.iloc[0][[f'RV_discs_{t}' for t in range(50)]])
ax.plot([0.0, 50.0], [row['RVEDV'], row['RVEDV']])
ax.plot([0.0, 50.0], [row['RVESV'], row['RVESV']])

# %%
discs
# %%
