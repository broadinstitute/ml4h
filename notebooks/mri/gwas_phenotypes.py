# %%
import pandas as pd

pheno_poisson = pd.read_csv('pheno_poisson.tsv', sep='\t')
pheno_pixels = pd.read_csv('pheno_pixels.tsv', sep='\t')
pheno_pixels = pheno_pixels.merge(pheno_poisson[['FID', 'RV_poisson_max', 'invnorm_RV_poisson_max']], on='FID')
# %%
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots()
sns.distplot(pheno_poisson[pheno_poisson['RV_poisson_max']<300.]['RV_poisson_max'], kde=False)
# %%

f, ax = plt.subplots()
sns.distplot(pheno_poisson['invnorm_RV_poisson_max'], kde=False)

# %%
pheno_pixels = pheno_pixels.dropna(subset=['RV_poisson_max', 'max_rv'])
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
# f.set_size_inches(3, 3)
ax.hexbin(pheno_pixels[pheno_pixels['RV_poisson_max']<300.]['RV_poisson_max'], pheno_pixels[pheno_pixels['RV_poisson_max']<300.]['max_rv'],
          mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([0, 400], [0, 400], color='k')
# ax.set_xlabel(f'Max RV volume (3-D surf FASTAI) (ml)')
# ax.set_ylabel('RVEDV (Petersen) (ml)')
# ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
# ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
# ax.set_xlim([50, 400])
# ax.set_ylim([50, 400])
pearson = scipy.stats.pearsonr(pheno_pixels[pheno_pixels['RV_poisson_max']<3e6]['RV_poisson_max'], 
                               pheno_pixels[pheno_pixels['RV_poisson_max']<3e6]['max_rv'])[0]
ax.set_title(f'n={len(pheno_pixels)}, r={pearson:.2f}')
plt.tight_layout()
# f.savefig('petersen_poisson_fastai_rv_max.png', dpi=500)
# %%
pheno_pixels = pheno_pixels.dropna(subset=['RV_poisson_max', 'max_rv'])
import scipy.stats
import matplotlib.pyplot as plt
f, ax = plt.subplots()
# f.set_size_inches(3, 3)
ax.hexbin(pheno_pixels[pheno_pixels['RV_poisson_max']<300.]['invnorm_RV_poisson_max'], pheno_pixels[pheno_pixels['RV_poisson_max']<300.]['invnorm_max_rv'],
          mincnt=1, cmap='gray')
ax.set_aspect('equal')
ax.plot([-4, 4], [-4, 4], color='k')
# ax.set_xlabel(f'Max RV volume (3-D surf FASTAI) (ml)')
# ax.set_ylabel('RVEDV (Petersen) (ml)')
# ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
# ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
# ax.set_xlim([50, 400])
# ax.set_ylim([50, 400])
pearson = scipy.stats.pearsonr(pheno_pixels[pheno_pixels['RV_poisson_max']<300]['invnorm_RV_poisson_max'], 
                               pheno_pixels[pheno_pixels['RV_poisson_max']<300]['invnorm_max_rv'])[0]
ax.set_title(f'n={len(pheno_pixels)}, r={pearson:.2f}')
plt.tight_layout()
# %%
