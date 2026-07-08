# ML4HEN-COX

## Description

This repo contains code to perform feature selection in very large datasets --- in both number of samples and number of covariates --- using survival data.

## Requirements

This code was tested using python 3.7.
It can be used using virtual env.

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

Several files are provided:

* [coxnet_training_testing_evaluating.py](./coxnet_training_testing_evaluating.py): fitting CoxNet models
* [xgboost_training_testing_evaluating.py](./xgboost_training_testing_evaluating.py): fitting XgCox models
* [2020.11.30_analysis_cleaned2.r](./2020.11.30_analysis_cleaned2.r): downstream R code
* [2020.11.30_analysis_cleaned2.ipynb](./2020.11.30_analysis_cleaned2.ipynb): downstream notebook

### Model loading

The provided xgboost model can be loaded as follows:

```py
import xgboost as xgb
xgcox = xgb.Booster()
xgcox.load_model("models/xgcox_model.json")
```

and the CoxNet model as:

```py
import pickle
from sksurv.linear_model import CoxnetSurvivalAnalysis
xxx = pickle.load(open('models/coxnet_survival_05_final.pickle', 'rb'))
```

### Citation

**[Selection of 51 predictors from 13,782 candidate multimodal features using machine learning improves coronary artery disease prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8672148/)**, Saaket Agrawal, BS*, Marcus D. R. Klarqvist, PhD,  MSc, MSc*, Connor Emdin, DPhil, MD, Aniruddh P. Patel, MD, Manish D. Paranjpe, BA, Patrick T. Ellinor, MD, PhD, Anthony Philippakis, MD, PhD, Kenney Ng, PhD, Puneet Batra, PhD, Amit V. Khera, MD, MSc

