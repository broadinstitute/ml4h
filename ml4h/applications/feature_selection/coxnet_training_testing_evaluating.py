
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
from numpy import sqrt, argmax
import zstandard
import pyarrow.parquet as pq
import fastparquet as fp
import pickle
from timeit import default_timer as timer
from sksurv.linear_model import CoxnetSurvivalAnalysis # scikit-survival-0.15.0
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored


# 1. ----------------------
# Load development data
# X_train is a (number of samples, covariates) matrix
X_train = fp.ParquetFile('development_data.pq').to_pandas()
# Outcome matrix comprising of a boolean `outcome` and a numerical value `time`
# Example:
#          outcome  time
# index
# 1111111    False  1044
# 2222222    False  3437
# 3333333    True   4214
# 4444444    False  4794
# 5555555    False  5898
train_cox_times = fp.ParquetFile('development_data_survival_times.pq').to_pandas()


# 2. ----------------------
# Cross-validate model on development data
# 
# Perform 5-fold cross-validation using a fixed seed: this ascertains that we always get
# the same 5 train-test splits.
#
# Split training data into five folds.
kf = KFold(n_splits=5, random_state=13372, shuffle=True).split(train_cox_times.outcome)

# List of alpha hyperparameters to evaluate for each fold. Example (random range):
# array([1.65634558e-02, 1.50920040e-02, 1.37512720e-02, 1.25296469e-02,
#        1.14165476e-02, 1.04023329e-02, 9.47821836e-03, 8.63619956e-03,
#        7.86898340e-03, 7.16992460e-03, 6.53296826e-03, 5.95259736e-03,
#        5.42378502e-03, 4.94195091e-03, 4.50292161e-03, 4.10289446e-03,
#        3.73840463e-03, 3.40629506e-03, 3.10368920e-03, 2.82796600e-03,
#        2.57673730e-03, 2.34782706e-03, 2.13925257e-03, 1.94920727e-03,
#        1.77604507e-03, 1.61826612e-03, 1.47450382e-03, 1.34351296e-03,
#        1.22415897e-03, 1.11540807e-03, 1.01631829e-03, 9.26031371e-04,
#        8.43765297e-04, 7.68807513e-04, 7.00508773e-04, 6.38277504e-04,
#        5.81574690e-04, 5.29909197e-04, 4.82833525e-04, 4.39939925e-04,
#        4.00856875e-04, 3.65245855e-04, 3.32798420e-04, 3.03233525e-04,
#        2.76295095e-04, 2.51749800e-04, 2.29385042e-04, 2.09007107e-04,
#        1.90439491e-04, 1.73521372e-04, 1.58106211e-04, 1.44060491e-04,
#        1.31262554e-04, 1.19601551e-04, 1.08976479e-04, 9.92953086e-05,
#        9.04741871e-05, 8.24367096e-05, 7.51132596e-05, 6.84404046e-05,
#        6.23603477e-05, 5.68204263e-05, 5.17726562e-05, 4.71733161e-05])
#
# It is possible to fit the CoxNet survival model below without providing alphas by
# not specifying the alphas parameter.
alphas = [] # List of alpha hyperparameters to evaluate

current_fold = 0 # Keep track of which fold we're currently on in the loop

# Loop over train and test folds and fit the model. For the current application, we first
# fit all the model and save them to disk before proceeding.
for train_index, test_index in kf:
    # Slice out the training times and outcomes and convert into (outcome,time)-tuples
    # Example cox_tuples[0:5]:
    # array([(False, 4044.), (True, 4437.), (False, 4004.), (True, 3781.),
    #   (False, 4498.)], dtype=[('f0', '?'), ('f1', '<f8')])
    cox_tuples = np.array(train_cox_times.iloc[train_index].to_records(index=False).tolist(), dtype=np.dtype('bool,float'))
    
    # Define the CoxNet model we want to fit
    estimator = CoxnetSurvivalAnalysis(
            normalize=False, # Do not normalize, the data is already normalized in our case
            l1_ratio=0.5, # Ratio between L1 and L2 regression penalties
            verbose=True,
            fit_baseline_model=True, # Fit the baselines
            copy_X=False, # Do not copy the data in this case as our data frames are very large
            tol=1e-9,
            alphas=alphas)
    
    # Fit the model using the training set of individuals
    estimator.fit(X_train.iloc[train_index], cox_tuples)
    
    # Pickle the actual model to disk
    pickle.dump(estimator, open(f'coxnet_survival_05_fold{current_fold}.pickle', 'wb'))
    
    # Increment fold
    current_fold = current_fold + 1


# 3. ----------------------
# Foreach 5-fold ---
#   1. Compute predictions for each alpha
#   2. Compute C-statistics for that alpha
#   3. Compute absolute risk given that fold
class CoxNetEvaluateHelper:
    """This convience class computes the discrimination (C-index) for train 
    and test each level of regulariziation (parameter alpha) given a provided
    fitted model. The model interface must have the `predict` function available
    ---this is true for all sklearn and sksurv models which we explicitly target.
    """
    def __init__(self, estimator, coefs, alphas, data, train_index, test_index, cox_times):
        self._estimator = estimator
        self._coefs  = coefs
        self._data   = data
        self._train_index = train_index
        self._test_index  = test_index
        self._times  = cox_times
        self._alphas = alphas
        self._c_train    = []
        self._c_test     = []
        self._n_features = []
        self._abs_risk   = []
        self._base_haz   = []
    
    def predict(self, verbose=True):
        for a, i in zip(self._estimator.alphas_, range(len(self._estimator.alphas_))):
            if verbose:
                print(f"Alpha: {a} for {i}/{len(self._estimator.alphas_)}")
            try:
                self.run_cox(i)
            except Exception as e:
                print("Failed inner loop.")
                raise Exception(e)
    
    def run_cox(self, offset: int):
        # Select alpha at the current offset
        alpha = self._alphas[offset]
        
        # Coefficients at the given alpha
        p = self._coefs.loc[alpha]
        
        # Remove covariates with coefficients reduced to zero
        p = p[p!=0]
        
        # Make predictions on test data at a given alpha
        predictions = self._estimator.predict(self._data.iloc[self._test_index], alpha=alpha)
        
        # Select target individuals in the follow-up time table
        cox_times_local = self._times.iloc[self._test_index]
        
        # Compute the C-statistic for the test fold
        result = concordance_index_censored(cox_times_local["outcome"], 
                                            cox_times_local["time"], 
                                            predictions)
        
        # Predict on train data for reference
        predictions = self._estimator.predict(self._data.iloc[self._train_index], alpha=alpha)
        
        # Compute absolute risk at year 10.
        # Compute the baseline hazard S_0(t) at timepoint 10*365 (10 years)
        s0_10 = self._estimator._get_baseline_model(alpha).cum_baseline_hazard_(10*365)
        abs_risk = 1 - (s0_10**np.exp(predictions))

        # Select target individuals in the follow-up time table
        cox_times_local = self._times.iloc[self._train_index]
        
        # Compute the C-statistic for training data predicting training data
        result_train = concordance_index_censored(cox_times_local["outcome"], 
                                                    cox_times_local["time"], 
                                                    predictions)
        
        # Store the C-statistics for both test and train data and the number of 
        # non-zero features.
        self._c_test.append(result[0])
        self._c_train.append(result_train[0])
        self._n_features.append(len(p)) 
        self._abs_risk.append(abs_risk)
        self._base_haz.append(s0_10)
        return True


# Split training data into the exact five folds used above by using the exact seed as above.
kf = KFold(n_splits=5,random_state=13372,shuffle=True).split(train_cox_times.outcome)
data_classes = [] # Store the results
current_fold = 0 # Keep track of which fold we're currently on in the loop

# Iterate over the folds
for train_index, test_index in kf:
    # Load the fitted estimator as saved above for a given fold
    estimator = pickle.load(open(f'coxnet_survival_05_fold{current_fold}.pickle', 'rb'))
    
    # Create a coefficient matrix for each hyperparameter alpha
    coefs = pd.DataFrame(estimator.coef_)
    coefs.columns = estimator.alphas_
    coefs = coefs.transpose()
    coefs.columns = X_train.columns

    # Use the CoxNetEvaluateHelper helper class to compute predictions, C-statistics, and absolute
    # risk.
    cox_data_class = CoxNetEvaluateHelper(estimator, coefs, estimator.alphas_, X_train, train_index, test_index, train_cox_times)
    cox_data_class.predict(verbose=True)
    
    # Keep the results for each fold
    data_classes.append(cox_data_class)
    
    # Increment fold counter
    current_fold = current_fold + 1


# Pickle the model-dervied statistics to disk
pickle.dump(data_classes, open(f'coxnet_survival_05_fold_results.pickle', 'wb'))

# 4. ----------------------
# Bootstrap confidence interval in the final chosen model

boot_c_scores = []
boot_n_failed = 0

# Run 100-times out-of-bag predictions
for b in range(100):
    # Randomly draw N samples from the original dataset where N is the number
    # of samples.
    train_targets = np.random.choice(X_train.index.values, len(X_train), replace=True)
    
    # Samples not drawn from the sampling procedure are used as an out-of-bag
    # (OOB) dataset for testing.
    test_oob = list(set(train_targets).symmetric_difference(set(X_train.index.values)))

    # Grab the target times
    cox_tuples = train_cox_times.loc[train_targets].to_records(index=False).tolist()
    
    # Current bootstrapped dataset
    X_boot = X_train.loc[train_targets]
    
    # Coxnet estimator
    estimator = CoxnetSurvivalAnalysis(
            normalize=False,
            l1_ratio=0.5,
            verbose=True,
            fit_baseline_model=True,
            copy_X=False,
            tol=1e-9,
            alphas=[0.0011765691116882482])
    
    # Fit the model
    try:
        estimator.fit(X_boot, np.array(cox_tuples, dtype=np.dtype('bool,float')))
    except Exception as e:
        boot_n_failed = boot_n_failed + 1
        continue

    # Make predictions on the OOB dataset
    predictions = estimator.predict(X_train.loc[test_oob])

    # Compute the C-index
    boot_times = train_cox_times.loc[test_oob]
    result = concordance_index_censored(boot_times["outcome"], 
                                        boot_times["time"], 
                                        predictions)
    
    # Store results
    boot_c_scores.append(result[0])


# Compute onfidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(boot_c_scores, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(boot_c_scores, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


# 5. ----------------------
# Evaluate the final chosen model on holdout data.

target_alpha = 

# Fit the chosen model using the given the best hyperparameter alpha on all available
# development data
estimator = CoxnetSurvivalAnalysis(
        normalize=False,
        l1_ratio=0.5,
        verbose=True,
        fit_baseline_model=True,
        copy_X=False,
        tol=1e-9,
        alphas=[target_alpha])

cox_tuples = train_cox_times.to_records(index=False).tolist()
estimator.fit(X_train, np.array(cox_tuples, dtype=np.dtype('bool,float')))
coefs = pd.DataFrame(estimator.coef_)

# Get baseline hazard
s0_10 = estimator._get_baseline_model(target_alpha).cum_baseline_hazard_(10*365) # 10 years in days

# Load the hold-out data
X_holdout = fp.ParquetFile('holdout_data.pq').to_pandas()
holdout_cox_times = fp.ParquetFile('holdout_data_survival_times.pq').to_pandas()
X_holdout_unscaled = fp.ParquetFile('holdout_data_unscaled.pq').to_pandas()

# Prepare stratifications
y_holdout = holdout_cox_times[['outcome']]
y_holdout['age_under_55'] = X_holdout_unscaled['d_age']< 55
y_holdout['age_over_55']  = X_holdout_unscaled['d_age']>=55
y_holdout['sex']          = X_holdout_unscaled['c_sex']

# Make predictions
predictions_holdout = estimator.predict(X_holdout)

# Compute the C-statistic for the entire holdout and for a variety 
# of stratifications
c_scores = []
c_scores_under55 = []
c_scores_over55  = []
c_scores_male    = []
c_scores_female  = []

# Compute the C-statistic for the test fold
result = concordance_index_censored(holdout_cox_times["outcome"], 
                                    holdout_cox_times["time"], 
                                    predictions_holdout)
c_scores.append(result[0])

# Under 55
result = concordance_index_censored(holdout_cox_times[y_holdout['age_under_55']==True]['outcome'], 
                                    holdout_cox_times[y_holdout['age_under_55']==True]['time'], 
                                    predictions_holdout[y_holdout['age_under_55']==True])
c_scores_under55.append(result[0])

# Under 55
result = concordance_index_censored(holdout_cox_times[y_holdout['age_over_55']==True]['outcome'], 
                                    holdout_cox_times[y_holdout['age_over_55']==True]['time'], 
                                    predictions_holdout[y_holdout['age_over_55']==True])
c_scores_over55.append(result[0])

# Males
result = concordance_index_censored(holdout_cox_times[y_holdout['sex']==1.0]['outcome'], 
                                    holdout_cox_times[y_holdout['sex']==1.0]['time'], 
                                    predictions_holdout[y_holdout['sex']==1.0])
c_scores_male.append(result[0])

# Females
result = concordance_index_censored(holdout_cox_times[y_holdout['sex']==0.0]['outcome'], 
                                    holdout_cox_times[y_holdout['sex']==0.0]['time'], 
                                    predictions_holdout[y_holdout['sex']==0.0])
c_scores_female.append(result[0])


# Construct a data frame of the above results and save to disk
holdout_results = pd.concat([pd.DataFrame({
        'alpha':   estimator.alphas_,
        'all':     c_scores,
        'under55': c_scores_under55,
        'over55':  c_scores_over55,
        'male':    c_scores_male,
        'female':  c_scores_female
    }),
    (coefs!=0.0).sum().reset_index()], axis=1)

holdout_results.to_csv('coxnet_survival_05_all_training_data__holdout__c_scores.txt',sep="\t")

# Compute absolute risk per individual
abs_risk_holdut = pd.DataFrame({
    'ukbid':  holdout_cox_times.index,
    'linear': predictions_holdout,
    'abs_risk_t10': 1-(s0_10**np.exp(predictions_holdout)),
    'time':   holdout_cox_times.time.values,
    'O':      holdout_cox_times.outcome.values,
    'O_T10':  holdout_cox_times.outcome.values,
    'O_T11':  holdout_cox_times.outcome.values,
    'O_T12':  holdout_cox_times.outcome.values,
})
abs_risk_holdut.O_T10[abs_risk_holdut.time>(10*365)] = False

# Write to disk
abs_risk_holdut.to_csv("coxnet__absolute_risk_with_ids__holdout_model_0_0011765691116882482.txt")


# 6. ----------------------
# Compute feature importance.

## Initial prediction
result = concordance_index_censored(holdout_cox_times["outcome"], 
                                    holdout_cox_times["time"], 
                                    predictions_holdout)
reference_c_statistic = result[0]

# Foreach coulmn we randomly permute (shuffle) its values
# and make a prediction and assess how much it differes from
# the base prediction.
permute_importance_c = []
for c in X_holdout.columns:
    difference_c_stats = []
    # Perform 100-times permutations per feature
    for i in range(100):
        X_holdout_cox_internal    = X_holdout.copy()
        
        # Make permutation
        X_holdout_cox_internal[c] = X_holdout_cox_internal[c].sample(frac=1).values
        
        # Make predictions with this permuted matrix
        predictions = estimator.predict(X_holdout_cox_internal)
        
        # Compute C-statistics
        permutation_result = concordance_index_censored(holdout_cox_times["outcome"], 
                                            holdout_cox_times["time"], 
                                            predictions)
        
        # Store difference compared to the reference C-statistics
        difference_c_stats.append(reference_c_statistic - permutation_result[0])
    
    # Store results
    permute_importance_c.append({c: difference_c_stats})


# Construct a dataframe
permutation_importance = pd.concat([pd.DataFrame(f) for f in permute_importance_c],axis=1)

# Sort by absolute value
permutation_importance = permutation_importance[permutation_importance.mean().abs().sort_values(ascending=False).index]

# Save to disk
permutation_importance.to_csv("coxnet_final_model_permutation_importance__model_0_0011765691116882482.txt",sep="\t")
permutation_importance.melt().to_csv("coxnet_final_model_permutation_importance_melted__model_0_0011765691116882482.txt",sep="\t")

