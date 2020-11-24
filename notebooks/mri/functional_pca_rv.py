# %%
from statsmodels.multivariate.pca import PCA
import pandas as pd
# %%

fmt_string = '/home/pdiachil/projects/chambers/rv_fastai/RV_processed_v20201102_{d}_{dp1}.csv'
df_fastai = pd.concat([pd.read_csv(fmt_string.format(d=d, dp1=d+1)) for d in range(45000) if os.path.isfile(fmt_string.format(d=d, dp1=d+1))])

# %%
fmt_string = '/home/pdiachil/projects/chambers/rv_ml4h/RV_processed_v20201102_{d}_{dp1}.csv'
df_ml4h = pd.concat([pd.read_csv(fmt_string.format(d=d, dp1=d+1)) for d in range(45000) if os.path.isfile(fmt_string.format(d=d, dp1=d+1))])


# %%
df_fastai.to_csv('all_rv_boundary_raw_fastai_v20201102.csv', index=False)
df_ml4h.to_csv('all_rv_boundary_raw_ml4h_v20201102.csv', index=False)

# %%
keys = [f'RV_poisson_{d}' for d in range(50)]
df_fastai = pd.read_csv('all_rv_boundary_raw_fastai_v20201102.csv')
df_ml4h = pd.read_csv('all_rv_boundary_raw_ml4h_v20201102.csv')
df_petersen = pd.read_csv('returned_lv_mass.tsv', sep='\t')
df_fastai = df_fastai[df_fastai['sample_id']!=-1]
df_ml4h = df_ml4h[df_ml4h['sample_id']!=-1]
df_fastai['RV_poisson_max'] = df_fastai[keys].max(axis=1)
df_fastai['RV_poisson_min'] = df_fastai[keys].min(axis=1) 
df_ml4h['RV_poisson_max'] = df_ml4h[keys].max(axis=1)
df_ml4h['RV_poisson_min'] = df_ml4h[keys].min(axis=1)
df_fastai = df_fastai[df_fastai['RV_poisson_min']>5.]
df_fastai.to_csv('all_rv_boundary_v20201102.csv')
df_fastai = df_fastai[df_fastai['RV_poisson_max']<500.]
df_ml4h = df_ml4h[df_ml4h['RV_poisson_min']>5.]
df_ml4h.to_csv('all_rv_boundary_ml4h_v20201102.csv')
df_ml4h = df_ml4h[df_ml4h['RV_poisson_max']<500.]

df_fastai['RVEF_poisson'] = (df_fastai['RV_poisson_max']-df_fastai['RV_poisson_min'])/df_fastai['RV_poisson_max']
df_ml4h['RVEF_poisson'] = (df_ml4h['RV_poisson_max']-df_ml4h['RV_poisson_min'])/df_ml4h['RV_poisson_max']
df_fastai_ml4h = df_fastai.merge(df_ml4h, on='sample_id', suffixes=('_fastai', '_ml4h'))
for feat in ['RV_poisson_max', 'RV_poisson_min', 'RVEF_poisson']:
    df_fastai_ml4h[f'{feat}_consensus'] = 0.5*(df_fastai_ml4h[f'{feat}_ml4h']+df_fastai_ml4h[f'{feat}_fastai'])
df_fastai_ml4h.to_csv('all_boundary_rv_fastai_ml4h.csv', index=False)
# %%
df_fastai_petersen = df_fastai.merge(df_petersen[['RVEDV', 'RVESV', 'RVEF', 'sample_id']], on='sample_id')
df_ml4h_petersen = df_ml4h.merge(df_petersen[['RVEDV', 'RVESV', 'RVEF', 'sample_id']], on='sample_id')
df_fastai_petersen = df_fastai_petersen.dropna()
df_ml4h_petersen = df_ml4h_petersen.dropna()
df_fastai_ml4h_petersen = df_fastai_ml4h.merge(df_petersen[['RVEDV', 'RVESV', 'RVEF', 'sample_id']], on='sample_id')
df_fastai_ml4h_petersen.to_csv('all_boundary_rv_fastai_ml4h_petersen.csv', index=False)
# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_fastai_petersen['RV_poisson_max'], df_fastai_petersen['RVEDV'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'Max RV volume (3-D surf FASTAI) (ml)')
ax.set_ylabel('RVEDV (Petersen) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([50, 400])
ax.set_ylim([50, 400])
pearson = scipy.stats.pearsonr(df_fastai_petersen['RV_poisson_max'], df_fastai_petersen['RVEDV'])[0]
ax.set_title(f'n={len(df_fastai_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('petersen_poisson_fastai_rv_max.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_ml4h_petersen['RV_poisson_max'], df_ml4h_petersen['RVEDV'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'Max RV volume (3-D surf ML4H) (ml)')
ax.set_ylabel('RVEDV (Petersen) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([50, 400])
ax.set_ylim([50, 400])
pearson = scipy.stats.pearsonr(df_ml4h_petersen['RV_poisson_max'], df_ml4h_petersen['RVEDV'])[0]
ax.set_title(f'n={len(df_ml4h_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('petersen_poisson_ml4h_rv_max.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_fastai_petersen['RV_poisson_min'], df_fastai_petersen['RVESV'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'Min RV volume (3-D surf FASTAI) (ml)')
ax.set_ylabel('RVESV (Petersen) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([0, 200])
ax.set_ylim([0, 200])
pearson = scipy.stats.pearsonr(df_fastai_petersen['RV_poisson_min'], df_fastai_petersen['RVESV'])[0]
ax.set_title(f'n={len(df_fastai_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('petersen_poisson_fastai_rv_min.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_ml4h_petersen['RV_poisson_min'], df_ml4h_petersen['RVESV'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'Min RV volume (3-D surf ML4H) (ml)')
ax.set_ylabel('RVESV (Petersen) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([0, 200])
ax.set_ylim([0, 200])
pearson = scipy.stats.pearsonr(df_ml4h_petersen['RV_poisson_min'], df_ml4h_petersen['RVESV'])[0]
ax.set_title(f'n={len(df_ml4h_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('petersen_poisson_ml4h_rv_min.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_fastai_ml4h_petersen['RV_poisson_max_fastai'], df_fastai_ml4h_petersen['RV_poisson_max_ml4h'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'Max RV volume (3-D surf FASTAI) (ml)')
ax.set_ylabel(f'Max RV volume (3-D surf ML4H) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([50, 400])
ax.set_ylim([50, 400])
pearson = scipy.stats.pearsonr(df_fastai_ml4h_petersen['RV_poisson_max_fastai'], df_fastai_ml4h_petersen['RV_poisson_max_ml4h'])[0]
ax.set_title(f'n={len(df_fastai_ml4h_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('petersen_poisson_fastai_ml4h_rv_max.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_fastai_ml4h_petersen['RV_poisson_min_fastai'], df_fastai_ml4h_petersen['RV_poisson_min_ml4h'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'Min RV volume (3-D surf FASTAI) (ml)')
ax.set_ylabel(f'Min RV volume (3-D surf ML4H) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([0, 200])
ax.set_ylim([0, 200])
pearson = scipy.stats.pearsonr(df_fastai_ml4h_petersen['RV_poisson_min_fastai'], df_fastai_ml4h_petersen['RV_poisson_min_ml4h'])[0]
ax.set_title(f'n={len(df_fastai_ml4h_petersen)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('petersen_poisson_fastai_ml4h_rv_min.png', dpi=500)
# %%
# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_fastai_ml4h['RV_poisson_max_fastai'], df_fastai_ml4h['RV_poisson_max_ml4h'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'Max RV volume (3-D surf FASTAI) (ml)')
ax.set_ylabel(f'Max RV volume (3-D surf ML4H) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([50, 400])
ax.set_ylim([50, 400])
pearson = scipy.stats.pearsonr(df_fastai_ml4h['RV_poisson_max_fastai'], df_fastai_ml4h['RV_poisson_max_ml4h'])[0]
ax.set_title(f'n={len(df_fastai_ml4h)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('poisson_fastai_ml4h_rv_max.png', dpi=500)

# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_fastai_ml4h['RV_poisson_min_fastai'], df_fastai_ml4h['RV_poisson_min_ml4h'], extent=(0, 400, 0, 400), mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
ax.set_xlabel(f'Min RV volume (3-D surf FASTAI) (ml)')
ax.set_ylabel(f'Min RV volume (3-D surf ML4H) (ml)')
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
ax.set_xlim([0, 200])
ax.set_ylim([0, 200])
pearson = scipy.stats.pearsonr(df_fastai_ml4h['RV_poisson_min_fastai'], df_fastai_ml4h['RV_poisson_min_ml4h'])[0]
ax.set_title(f'n={len(df_fastai_ml4h)}, r={pearson:.2f}')
plt.tight_layout()
f.savefig('poisson_fastai_ml4h_rv_min.png', dpi=500)
# %%
df_fastai_ml4h_petersen['RVEF_fastai'] = (df_fastai_ml4h_petersen['RV_poisson_max_fastai'] - df_fastai_ml4h_petersen['RV_poisson_min_fastai'])/df_fastai_ml4h_petersen['RV_poisson_max_fastai']
df_fastai_ml4h_petersen['RVEF_ml4h'] = (df_fastai_ml4h_petersen['RV_poisson_max_ml4h'] - df_fastai_ml4h_petersen['RV_poisson_min_ml4h'])/df_fastai_ml4h_petersen['RV_poisson_max_ml4h']
df_fastai_ml4h_petersen = df_fastai_ml4h_petersen.dropna()
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_fastai_ml4h_petersen['RVEF_fastai'], df_fastai_ml4h_petersen['RVEF'], mincnt=1, cmap='gray')
# ax.set_aspect('equal')
# ax.plot([0, 400], [0, 400], color='k')
# ax.set_xlabel(f'Min RV volume (3-D surf FASTAI) (ml)')
# ax.set_ylabel(f'Min RV volume (3-D surf ML4H) (ml)')
# ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
# ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
# ax.set_xlim([0, 200])
# ax.set_ylim([0, 200])
pearson = scipy.stats.pearsonr(df_fastai_ml4h_petersen['RVEF_fastai'], df_fastai_ml4h_petersen['RVEF'])[0]
ax.set_title(f'n={len(df_fastai_ml4h_petersen)}, r={pearson:.2f}')
# plt.tight_layout()
# f.savefig('poisson_fastai_ml4h_rv_min.png', dpi=500)
# %%
# %%
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
f.set_size_inches(3, 3)
ax.hexbin(df_fastai_ml4h_petersen['RVEF_ml4h'], df_fastai_ml4h_petersen['RVEF'], mincnt=1, cmap='gray')
# ax.set_aspect('equal')
# ax.plot([0, 400], [0, 400], color='k')
# ax.set_xlabel(f'Min RV volume (3-D surf FASTAI) (ml)')
# ax.set_ylabel(f'Min RV volume (3-D surf ML4H) (ml)')
# ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
# ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
# ax.set_xlim([0, 200])
# ax.set_ylim([0, 200])
pearson = scipy.stats.pearsonr(0.5*(df_fastai_ml4h_petersen['RVEF_ml4h']+df_fastai_ml4h_petersen['RVEF_fastai']), df_fastai_ml4h_petersen['RVEF'])[0]
ax.set_title(f'n={len(df_fastai_ml4h_petersen)}, r={pearson:.2f}')
# plt.tight_layout()
# f.savefig('poisson_fastai_ml4h_rv_min.png', dpi=500)

# %%
import seaborn as sns
df_fastai_ml4h_petersen['RV_poisson_max_consensus'] = 0.5*(df_fastai_ml4h_petersen['RV_poisson_max_fastai']+df_fastai_ml4h_petersen['RV_poisson_max_ml4h'])
df_fastai_ml4h_petersen['RV_poisson_min_consensus'] = 0.5*(df_fastai_ml4h_petersen['RV_poisson_min_fastai']+df_fastai_ml4h_petersen['RV_poisson_min_ml4h'])

f, ax = plt.subplots()
f.set_size_inches(5, 4)
sns.heatmap(df_fastai_ml4h_petersen[['RV_poisson_max_fastai', 'RV_poisson_max_ml4h', 'RV_poisson_max_consensus', 'RVEDV']].corr(),
            cmap='gray_r', ax = ax, annot=True, cbar=False          
        )
ticklabels=['RV$_{max}$ (3-D surf FASTAI)', 'RV$_{max}$ (3-D surf ML4H)', 'RV$_{max}$ (3-D surf FASTAI+ML4H)', 'RVEDV (Petersen)']
ax.set_xticklabels(ticklabels, rotation=45, ha='right')
ax.set_yticklabels(ticklabels)
plt.tight_layout()
f.savefig('rv_heatmap_max.png', dpi=500)

# %%
import seaborn as sns
f, ax = plt.subplots()
f.set_size_inches(5, 4)
sns.heatmap(df_fastai_ml4h_petersen[['RV_poisson_min_fastai', 'RV_poisson_min_ml4h', 'RV_poisson_min_consensus', 'RVESV']].corr(),
            cmap='gray_r', ax = ax, annot=True, cbar=False            
        )
ticklabels=['RV$_{min}$ (3-D surf FASTAI)', 'RV$_{min}$ (3-D surf ML4H)', 'RV$_{min}$ (3-D surf FASTAI+ML4H)', 'RVESV (Petersen)']
ax.set_xticklabels(ticklabels, rotation=45, ha='right')
ax.set_yticklabels(ticklabels)
plt.tight_layout()
f.savefig('rv_heatmap_min.png', dpi=500)
# %%
