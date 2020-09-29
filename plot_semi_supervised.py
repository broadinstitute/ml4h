import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.metrics import r2_score


N_BOOT = 1


def find_inference(folder: str):
    return glob.glob(f'{folder}/*/inference*.tsv')


def bootstrap_R2(y_true, y_pred, n=500):
    scores = []
    for _ in range(n):
        idx = np.random.randint(0, y_true.shape[0], y_true.shape[0]) 
        scores.append(r2_score(y_true[idx], y_pred[idx]))
    return scores
    


def get_score_df(path: str):
    df = pd.read_csv(path, sep='\t')

    actual = df['standardized_partners_ecg_age_newest_actual']
    
    score_dfs = []
    for column in df.columns:
        if 'prediction' not in column:
            continue
        split = column.split('_')
        percent = int(split[-2])
        latents = int(split[-5])
        model = split[-6]
        print(f'Scoring {model} with percent {percent} latents {latents}')
        scores = bootstrap_R2(actual, df[column], N_BOOT)
        score_df = pd.DataFrame({'score': scores})
        score_df['name'] = model
        score_df['training samples'] = int(percent / 100 * 193852)
        score_df['latents'] = latents
        score_dfs.append(score_df)

    return pd.concat(score_dfs)


def get_supervised_score_df(path: str):
    df = pd.read_csv(path, sep='\t')

    actual = df['standardized_partners_ecg_age_newest_actual']
    
    score_dfs = []
    for column in df.columns:
        if 'prediction' not in column:
            continue
        split = column.split('_')
        percent = int(split[-2])
        latents = int(split[-5])
        model = 'age supervised'
        print(f'Scoring {model} with percent {percent} latents {latents}')
        scores = bootstrap_R2(actual, df[column], N_BOOT)
        score_df = pd.DataFrame({'score': scores})
        score_df['name'] = model
        score_df['training samples'] = int(percent / 100 * 193852)
        score_df['latents'] = latents
        score_dfs.append(score_df)

    return pd.concat(score_dfs)


semi_supervised = find_inference('train_runs/semi_supervised')
supe = find_inference('train_runs/age_directly')
score_df = pd.concat(list(map(get_score_df, semi_supervised)) + list(map(get_supervised_score_df, supe)))

plt.figure(figsize=(7, 7))
score_df['Age $R^2$'] = score_df['score']
hue_order = 'supervised', 'ae', 'vae', 'simclr', 'age supervised'
#sns.catplot(data=score_df, x='percent', y='Age $R^2$', hue='name', kind='point', capsize=.2, col='latents', ci='sd')
sns.pointplot(data=score_df, x='training samples', y='Age $R^2$', hue='name', kind='point', ci=100, dodge=True, hue_order=hue_order)
plt.savefig(f'/afs/csail.mit.edu/u/n/ndiamant/public_html/semi_supervised_compare.png', dpi=300)

