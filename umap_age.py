
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap


paths = ['supervised_deep_9-16.tsv', 'ae_deep_9-18.tsv', 'vae_deep_9-18.tsv', 'simclr_deep_9-16.tsv']
#target = 'partners_ecg_age_newest'
target = 'partners_ecg_rate_md_newest'
output_folder = 'umap_rate_9-22'
os.makedirs(output_folder, exist_ok=True)


def name_latents(model: str):
    split = model.split('_')
    latents = int(split[-1])
    name = split[1]
    return name, latents


def plot_umaps(path: str):
    df = pd.read_csv(path, sep='\t')
    #df = df[(df[target] > 20) & (df[target] < 100)]
    model_to_latents = defaultdict(list)
    for column in df.columns:
        if 'latent' not in column:
            continue
        model, _ = column.split('_latent_')
        model_to_latents[model].append(column)
    for model, cols in model_to_latents.items():
        print(f'------------ {model} ------------')
        plt.figure(figsize=(7, 9))
        reducer = umap.UMAP()
        X = StandardScaler().fit_transform(df[cols].values)
        embed = reducer.fit_transform(X)
        #embed_df = pd.DataFrame({'UMAP 1': embed[:, 0], 'UMAP 2': embed[:, 1], 'age': df[target]})
        #sns.scatterplot(x='UMAP 1', y='UMAP 2', hue=embed_df['age'].to_list(), data=embed_df)
        sc = plt.scatter(embed[:, 0], embed[:, 1], c=df[target], s=1, cmap='rainbow')
        plt.axis('off')
        plt.colorbar(sc, orientation='horizontal', label='age')
        name, latents = name_latents(model)
        plt.title(f'UMAP of {name} with {latents} latents colored by age')
        plt.savefig(os.path.join(output_folder, f'{name}_{latents}.png'), dpi=300)
    plt.close('all')


list(map(plot_umaps, paths))
