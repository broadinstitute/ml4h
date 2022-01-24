
# ML4H is released under the following BSD 3-Clause License:
#
# Copyright (c) 2020, Broad Institute, Inc. and The General Hospital Corporation.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name Broad Institute, Inc. or The General Hospital Corporation
#   nor the names of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Load the required dependencies
import pandas as pd
import numpy as np
import fastparquet as fp
import xgboost as xgb
from sklearn.model_selection import KFold
import sksurv.linear_model
from sksurv.metrics import concordance_index_censored
import gc


# 0. ----------------------
# Prepare xgboost data. This step may not be necessary if your data is small.
X_train         = fp.ParquetFile('development_data.pq').to_pandas()
train_cox_times = fp.ParquetFile('development_data_survival_times.pq').to_pandas()

# xgboost requires that censored times are negative
train_cox_times.time[train_cox_times.outcome == False] = train_cox_times.time[train_cox_times.outcome == False] * -1

# Convert into XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train, label=train_cox_times.time)
dtrain.save_binary(f'development_data.data')

# Perform 5-fold cross-validation using a fixed seed: this ascertains that we always get
# the same 5 train-test splits.
#
# Split training data into five folds and convert into DMatrix format
kf = KFold(n_splits=5, random_state=13372, shuffle=True).split(train_cox_times.outcome)
current_fold = 0

for train_index, test_index in kf:
    dtrain = xgb.DMatrix(X_train.iloc[train_index], label=train_cox_times.iloc[train_index].time)
    dtest  = xgb.DMatrix(X_train.iloc[test_index],  label=train_cox_times.iloc[test_index].time)
    dtrain.save_binary(f'development_data__fold_{current_fold}__train.data')
    dtest.save_binary(f'development_data__fold_{current_fold}__test.data')
    print(dtrain.num_col(), dtrain.num_row(), dtest.num_col(), dtest.num_row())
    del dtrain
    del dtest
    current_fold = current_fold + 1


# Holdout data
X_holdout = fp.ParquetFile('holdout_data.pq').to_pandas()
holdout_cox_times = fp.ParquetFile('holdout_data_survival_times').to_pandas()

# xgboost requires that censored times are negative
holdout_cox_times.time[holdout_cox_times.outcome == False] = holdout_cox_times.time[holdout_cox_times.outcome == False] * -1

# Convert into XGBoost DMatrix format
dholdout = xgb.DMatrix(X_holdout, label=holdout_cox_times.time)
dholdout.save_binary(f'holdout_data.data')

# 1. ------------Ò‰----------
# Search for optimal hyperparameters. There are many, many ways this is achieved so we
# will leave this section to the reader


# 2. ----------------------
# Evaluate a model given the optimal hyperparameters
# Example hyperparameters found:
space = {
    'booster': 'gbtree',
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.9500000000000001,
    'colsample_bytree': 1.0,
    'learning_rate': 0.13,
    'max_depth': 2,
    'min_child_weight': 6.0,
    'min_split_loss': 2.0,
    'objective': 'survival:cox',
    'reg_alpha': 8.5,
    'reg_lambda': 3.5,
    'scale_pos_weight': 30.0,
    'seed': 1137,
    'subsample': 1.0,
}


# Load training data and times
X_train         = fp.ParquetFile('development_data.pq').to_pandas()
train_cox_times = fp.ParquetFile('development_data_survival_times.pq').to_pandas()
train_cox_times['time_original'] = train_cox_times.time

# xgboost requires that censored times are negative
train_cox_times.time[train_cox_times.outcome == False] = train_cox_times.time[train_cox_times.outcome == False] * -1

# Run 5-fold cross-validation
kf = KFold(n_splits=5,random_state=13372,shuffle=True).split(train_cox_times.outcome)
current_fold = 0

# Save a variety of statistics
c_scores_train    = []
c_scores_test     = []
predictions_train = []
predictions_test  = []
O_train = []
O_test  = []
T_train = []
T_test  = []
abs_risk_train = []
abs_risk_test  = []
s_naughts      = [] # S0(t) for t=[10, 11, 12]
best_iteration = [] # Number of cycles used for each fold

# Train models
for train_index, test_index in kf:
    progress = dict()
    dtrain = xgb.DMatrix(f'development_data__fold_{current_fold}__train.data')
    dtest  = xgb.DMatrix(f'development_data__fold_{current_fold}__test.data')
    trained = xgb.train(
        space,
            dtrain,
            2000,
            early_stopping_rounds=20,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            verbose_eval=True,
    )
    #
    best_iteration.append(trained.best_iteration + 1)
    #
    # Predictions are returned as exp(X^tB) rather than the linear predictor X^tB
    # Predict train data
    predictions = trained.predict(dtrain, ntree_limit=trained.best_iteration + 1, output_margin=True)
    predictions = pd.DataFrame(predictions)
    predictions = predictions.reset_index(drop=True)
    pred_train  = predictions[0].values.flatten()
    predictions_train.append(pred_train)
    #
    O = train_cox_times.iloc[train_index].outcome
    T = train_cox_times.iloc[train_index].time_original
    O = O.reset_index(drop=True)
    T = T.reset_index(drop=True)
    O_train.append(O.values)
    T_train.append(T.values)
    #
    # Compute the baseline hazard S_0(t) at timepoint 10*365 (10 years)
    base_haz = sksurv.linear_model.CoxPHSurvivalAnalysis()._baseline_model.fit(pred_train, O.values, T.values)
    s0_10 = base_haz.baseline_survival_(10*365)
    #
    # Store S_0 and compute C-statistics
    s_naughts.append(np.array([s0_10]))
    result_train = concordance_index_censored(O, T, pred_train)
    c_scores_train.append(result_train[0])
    #
    # Predict test data
    predictions = trained.predict(dtest, ntree_limit=trained.best_iteration + 1, output_margin=True)
    predictions = pd.DataFrame(predictions)
    predictions = predictions.reset_index(drop=True)
    pred_test   = predictions[0].values.flatten()
    predictions_test.append(pred_test)
    #
    O = train_cox_times.iloc[test_index].outcome
    T = train_cox_times.iloc[test_index].time_original
    O = O.reset_index(drop=True)
    T = T.reset_index(drop=True)
    O_test.append(O.values)
    T_test.append(T.values)
    #
    result_test = concordance_index_censored(O, T, pred_test)
    c_scores_test.append(result_test[0])
    #
    # 2. Compute reference E(X^T * beta)
    reference_wtb_population = np.mean(pred_train)
    abs_risk_train.append([
        1-(s0_10**np.exp(pred_train)),
    ])
    abs_risk_test.append([
        1-(s0_10**np.exp(pred_test)),
    ])
    current_fold = current_fold + 1
    #
    # Release memory. This may not be necessary if your dataframes are small
    del dtrain
    del dtest
    gc.collect()


# Store C-statistcs for train and test
df = pd.DataFrame({'c_train': c_scores_train, 'c_test': c_scores_test})
df.to_csv("xgboost_cox_best_model_c.txt")

# Extract train and test predictions
ptrain = [pd.DataFrame(f) for f in predictions_train]
ptest  = [pd.DataFrame(f) for f in predictions_test]

# Set indices
kf = KFold(n_splits=5,random_state=13372,shuffle=True)
current_fold = 0
for train_index, test_index in kf.split(train_cox_times.outcome):
    ptrain[current_fold].index = train_cox_times.iloc[train_index].index
    ptest[current_fold].index  = train_cox_times.iloc[test_index].index
    current_fold = current_fold + 1


# Store statistics
# Training data
pjoint = []
current_fold = 0
for i,j,k,l in zip(ptrain,O_train,T_train,abs_risk_train):
    O = pd.DataFrame(j)
    O.index = i.index
    T = pd.DataFrame(k)
    T.index = i.index
    d = pd.DataFrame(l).transpose()
    d.index = i.index
    temp = pd.concat([i, O, T, d],axis=1)
    temp['fold'] = current_fold
    pjoint.append(temp)
    current_fold = current_fold + 1

pjoint = pd.concat(pjoint)
pjoint.columns = ['linear_score','outcome','time', 'predict_10', 'fold']
pjoint.to_csv("xgboost_cox_best_model_predictions_train.txt")

# Testing data
pjoint = []
current_fold = 0
for i,j,k,l in zip(ptest,O_test,T_test,abs_risk_test):
    O = pd.DataFrame(j)
    O.index = i.index
    T = pd.DataFrame(k)
    T.index = i.index
    d = pd.DataFrame(l).transpose()
    d.index = i.index
    temp = pd.concat([i, O, T, d],axis=1)
    temp['fold'] = current_fold
    pjoint.append(temp)
    current_fold = current_fold + 1


pjoint = pd.concat(pjoint)
pjoint.columns = ['linear_score','outcome','time', 'predict_10', 'fold']
pjoint.to_csv("xgboost_cox_best_model_predictions_test.txt")


# 3. ---------------------------------
# Train with the best hyperparameters on all data and evaluate on holdout

best_iteration = np.array(best_iteration)
dtrain = xgb.DMatrix('development_data.data')

# Fit model
trained_all = xgb.train(
    space,
        dtrain,
        num_boost_round=np.max(best_iteration), # Largest number of rounds from the 5-fold CV
        verbose_eval=True,
)

trained_all.save_model("xgcox_model.json")

# Make predictions on development data
predictions = trained_all.predict(dtrain, ntree_limit=trained_all.best_iteration + 1, output_margin=True)

# Compute C-index
train_c = concordance_index_censored(train_cox_times.outcome, train_cox_times.time_original, predictions)

# Load holdout data
dholdout  = xgb.DMatrix('holdout_data.data')
holdout_cox_times = fp.ParquetFile('development_data_survival_times.pq').to_pandas().reset_index().set_index('index')
holdout_cox_times['time_original'] = holdout_cox_times.time
holdout_cox_times.time[holdout_cox_times.outcome==False] = holdout_cox_times.time[holdout_cox_times.outcome==False]*-1

# Make predictions
predictions_holdout = trained_all.predict(dholdout, ntree_limit=trained_all.best_iteration + 1, output_margin=True)

# Compute C-index
test_c = concordance_index_censored(holdout_cox_times.outcome, holdout_cox_times.time_original, predictions_holdout)

# Store results
all_data = pd.DataFrame([train_c, test_c])
all_data.to_csv('xgboost_cox_best_model_predictions_all_data.txt',sep="\t")

# Load
# Load the hold-out data
X_holdout = fp.ParquetFile('holdout_data.pq').to_pandas()
holdout_cox_times = fp.ParquetFile('holdout_data_survival_times.pq').to_pandas()
X_holdout_unscaled = fp.ParquetFile('holdout_data_unscaled.pq').to_pandas()

# Prepare stratifications
y_holdout = holdout_cox_times[['outcome']]
y_holdout['age_under_55'] = X_holdout_unscaled['d_age']< 55
y_holdout['age_over_55']  = X_holdout_unscaled['d_age']>=55
y_holdout['sex']          = X_holdout_unscaled['c_sex']

# Compute the C-statistic for the entire holdout and for a variety
# of stratifications
c_scores = []
c_scores_under55 = []
c_scores_over55  = []
c_scores_male    = []
c_scores_female  = []

 # Make predictions on holdout data
predictions = trained_all.predict(dholdout, ntree_limit=trained_all.best_iteration + 1, output_margin=True)

# Compute the C-statistic for the test fold
result = concordance_index_censored(holdout_cox_times.outcome, holdout_cox_times.time_original, predictions_holdout)
c_scores.append(result[0])

# Under 55
result = concordance_index_censored(
    holdout_cox_times[y_holdout['age_under_55']==True]['outcome'],
    holdout_cox_times[y_holdout['age_under_55']==True]['time_original'],
    predictions[y_holdout['age_under_55']==True],
)

c_scores_under55.append(result[0])

# Over 55
result = concordance_index_censored(
    holdout_cox_times[y_holdout['age_over_55']==True]['outcome'],
    holdout_cox_times[y_holdout['age_over_55']==True]['time_original'],
    predictions[y_holdout['age_over_55']==True],
)

c_scores_over55.append(result[0])

# Males
result = concordance_index_censored(
    holdout_cox_times[y_holdout['sex']==1.0]['outcome'],
    holdout_cox_times[y_holdout['sex']==1.0]['time_original'],
    predictions[y_holdout['sex']==1.0],
)

c_scores_male.append(result[0])

# Females
result = concordance_index_censored(
    holdout_cox_times[y_holdout['sex']==0.0]['outcome'],
    holdout_cox_times[y_holdout['sex']==0.0]['time_original'],
    predictions[y_holdout['sex']==0.0],
)

c_scores_female.append(result[0])

# Construct dataframe and save to disk
holdout_results = pd.DataFrame({
        'all':     c_scores,
        'under55': c_scores_under55,
        'over55':  c_scores_over55,
        'male':    c_scores_male,
        'female':  c_scores_female,
})

holdout_results.to_csv('xgboost_cox_best_model_predictions_all_data_subgroups.txt',sep="\t")

# 4. ---------------------------------
# Compute statistics

# Compute baseline hazard function
base_haz = sksurv.linear_model.CoxPHSurvivalAnalysis()._baseline_model.fit(
    predictions_train,
    train_cox_times.outcome.values,
    train_cox_times.time_original.values,
)
s0_10 = base_haz.baseline_survival_(10 * 365)

# Store results
baseline_survival_data = pd.DataFrame({
    'time_days': base_haz.baseline_survival_.x,
    'baseline_survival': base_haz.baseline_survival_.y,
})
baseline_survival_data.to_csv("xgboost_cox_best_model_predictions_all_data__baseline_survival.txt")

# Compute C-index for training data
result_train = concordance_index_censored(
    train_cox_times.outcome.values,
    train_cox_times.time_original.values,
    predictions_train,
)

# Compute C-index for holdout data
predictions_holdout = trained_all.predict(dholdout, ntree_limit=trained_all.best_iteration + 1, output_margin=True)
result_test = concordance_index_censored(
    holdout_cox_times.outcome.values,
    holdout_cox_times.time_original.values,
    predictions_holdout,
)

# Prepare holdout risks
abs_risk_holdut = pd.DataFrame({
    'ukbid':  holdout_cox_times.index,
    'linear': predictions_holdout,
    'abs_risk_t10': 1 - (s0_10**np.exp(predictions_holdout)),
    'time':   holdout_cox_times.time_original.values,
    'O':      holdout_cox_times.outcome.values,
    'O_T10':  holdout_cox_times.outcome.values,
})
abs_risk_holdut.O_T10[abs_risk_holdut.time > (10 * 365)] = False

# Write to disk
abs_risk_holdut.to_csv("xgboost_cox_best_model_predictions_all_data__absolute_risk_with_ids.txt")

# Prepare training risks
abs_risk_train = pd.DataFrame({
    'ukbid': train_cox_times.index,
    'linear': predictions_train,
    'abs_risk_t10': 1 - (s0_10**np.exp(predictions_train)),
    'time': train_cox_times.time_original.values,
    'O': train_cox_times.outcome.values,
    'O_T10': train_cox_times.outcome.values,
})
abs_risk_train.O_T10[abs_risk_train.time > (10 * 365)] = False

# Write to disk
abs_risk_train.to_csv("xgboost_cox_best_model_predictions_all_data__absolute_risk_with_ids__development.txt")
