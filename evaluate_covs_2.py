import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import cross_val_score


paths = ['supervised_deep_9-16.tsv', 'ae_deep_9-18.tsv', 'vae_deep_9-18.tsv', 'simclr_deep_9-16.tsv']
target = 'partners_ecg_age_newest'
reg = RidgeCV(normalize=True, alphas=np.logspace(-6, 5, 45))
output_folder = '9-18_compare'
os.makedirs(output_folder, exist_ok=True)


def name_latents(model: str):
    split = model.split('_')
    latents = int(split[-1])
    name = split[1]
    return name, latents


def get_score_df(path: str):
    df = pd.read_csv(path, sep='\t')

    model_to_latents = defaultdict(list)
    for column in df.columns:
        if 'latent' not in column:
            continue
        model, _ = column.split('_latent_')
        model_to_latents[model].append(column)

    model_to_scores = []
    for model, cols in model_to_latents.items():
        print(f'------------ {model} ------------')
        scores = cross_val_score(reg, df[cols], df[target], cv=10)
        print(f'Mean score for target {target} {scores.mean():.3f} - Score std {scores.std():.3f}')
        name, latents = name_latents(model)
        for score in scores:
            model_to_scores.append({'name': name, 'latents': latents, 'score': score})

    return pd.DataFrame(model_to_scores)


scores = pd.concat(map(get_score_df, paths))
plt.figure(figsize=(12, 7))
sns.catplot(data=scores, x='latents', y='score', hue='name', kind='point', capsize=.2)
plt.ylabel(f'Age $R^2$')
plt.savefig(os.path.join(output_folder, f'latents_R2_for_{target}'), dpi=300)
