# metrics.py
import json
import os
import time
import logging

import keras
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
import tensorflow.keras.backend as K
from sklearn.metrics import roc_curve, auc, average_precision_score
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.losses import LogCosh, CosineSimilarity, MSE, MAE, MAPE, Dice
from keras.saving import register_keras_serializable


STRING_METRICS = [
    'categorical_crossentropy','binary_crossentropy','mean_absolute_error','mae',
    'mean_squared_error', 'mse', 'cosine_similarity', 'log_cosh', 'sparse_categorical_crossentropy',
]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~ Metrics ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def weighted_crossentropy(weights, name='anonymous'):
    """A weighted version of tensorflow.keras.objectives.categorical_crossentropy

    Arguments:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        name: string identifying the loss to differentiate when models have multiple losses

    Returns:
        keras loss function named name+'_weighted_loss'

    """
    string_globe = 'global ' + name + '_weights\n'
    string_globe += 'global ' + name + '_kweights\n'
    string_globe += name + '_weights = np.array(weights)\n'
    string_globe += name + '_kweights = K.variable('+name+'_weights)\n'
    exec(string_globe, globals(), locals())
    fxn_postfix = '_weighted_loss'
    string_fxn = 'def ' + name + fxn_postfix + '(y_true, y_pred):\n'
    string_fxn += '\ty_pred /= K.sum(y_pred, axis=-1, keepdims=True)\n'
    string_fxn += '\ty_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())\n'
    string_fxn += '\tloss = y_true * K.log(y_pred) * ' + name + '_kweights\n'
    string_fxn += '\tloss = -K.sum(loss, -1)\n'
    string_fxn += '\treturn loss\n'
    exec(string_fxn, globals(), locals())
    loss_fxn = eval(name + fxn_postfix, globals(), locals())
    loss_fxn = register_keras_serializable()(loss_fxn)
    return loss_fxn


def sparse_cross_entropy(window_size: int):
    def _sparse_cross_entropy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, window_size))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none',
        )(y_true, y_pred)
        return tf.reduce_mean(loss)
    return _sparse_cross_entropy


def angle_between_batches(tensors):
    l0 = K.sqrt(K.sum(K.square(tensors[0]), axis=-1, keepdims=True) + K.epsilon())
    l1 = K.sqrt(K.sum(K.square(tensors[1]), axis=-1, keepdims=True) + K.epsilon())
    numerator = K.sum(tensors[0]*tensors[1], axis=-1, keepdims=True)
    return tf.acos(numerator / (l0*l1))


def two_batch_euclidean(tensors):
    return K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1, keepdims=True) + K.epsilon())


def custom_loss_keras(user_id, encodings):
    pairwise_diff = K.expand_dims(encodings, 0) - K.expand_dims(encodings, 1)
    pairwise_squared_distance = K.sum(K.square(pairwise_diff), axis=-1)
    pairwise_distance = K.sqrt(pairwise_squared_distance + K.epsilon())

    user_id = K.squeeze(user_id, axis=1)  # remove the axis added by Keras
    pairwise_equal = K.equal(K.expand_dims(user_id, 0), K.expand_dims(user_id, 1))

    pos_neg = K.cast(pairwise_equal, K.floatx()) * 2 - 1
    return K.sum(pairwise_distance * pos_neg, axis=-1) / 2


def euclid_dist(v):
    return (v[0] - v[1])**2


def angle_between_batches(tensors):
    l0 = K.sqrt(K.sum(K.square(tensors[0]), axis=-1, keepdims=True) + K.epsilon())
    l1 = K.sqrt(K.sum(K.square(tensors[1]), axis=-1, keepdims=True) + K.epsilon())
    numerator = K.sum(tensors[0]*tensors[1], axis=-1, keepdims=True)
    return tf.acos(numerator / (l0*l1))


def paired_angle_between_batches(tensors):
    l0 = K.sqrt(K.sum(K.square(tensors[0]), axis=-1, keepdims=True) + K.epsilon())
    l1 = K.sqrt(K.sum(K.square(tensors[1]), axis=-1, keepdims=True) + K.epsilon())
    numerator = K.sum(tensors[0]*tensors[1], axis=-1, keepdims=True)
    angle_w_self = tf.acos(numerator / (l0*l1))
    # This is very hacky! we assume batch sizes are odd and reverse the batch to compare to others.
    l1_other = K.sqrt(K.sum(K.square(K.reverse(tensors[1], 0)), axis=-1, keepdims=True) + K.epsilon())
    other_numerator = K.sum(tensors[0]*K.reverse(tensors[1], 0), axis=-1, keepdims=True)
    angle_w_other = tf.acos(other_numerator / (l0*l1_other))
    return angle_w_self - angle_w_other


def ignore_zeros_l2(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    return MSE(y_true * mask, y_pred * mask)


def ignore_zeros_logcosh(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    return LogCosh(y_true * mask, y_pred * mask)


def sentinel_logcosh_loss(sentinel: float):
    def ignore_sentinel_logcosh(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, sentinel), K.floatx())
        return LogCosh(y_true * mask, y_pred * mask)
    return ignore_sentinel_logcosh


def y_true_times_mse(y_true, y_pred):
    return K.maximum(y_true, 1.0)*MSE(y_true, y_pred)


def mse_10x(y_true, y_pred):
    return 10.0*MSE(y_true, y_pred)


def y_true_squared_times_mse(y_true, y_pred):
    return K.maximum(1.0+y_true, 1.0)*K.maximum(1.0+y_true, 1.0)*MSE(y_true, y_pred)


def y_true_cubed_times_mse(y_true, y_pred):
    return K.maximum(y_true, 1.0)*K.maximum(y_true, 1.0)*K.maximum(y_true, 1.0)*MSE(y_true, y_pred)


def y_true_squared_times_logcosh(y_true, y_pred):
    return K.maximum(1.0+y_true, 1.0)*K.maximum(1.0+y_true, 1.0)*LogCosh(y_true, y_pred)


def two_batch_euclidean(tensors):
    return K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1, keepdims=True) + K.epsilon())


def custom_loss_keras(user_id, encodings):
    pairwise_diff = K.expand_dims(encodings, 0) - K.expand_dims(encodings, 1)
    pairwise_squared_distance = K.sum(K.square(pairwise_diff), axis=-1)
    pairwise_distance = K.sqrt(pairwise_squared_distance + K.epsilon())

    user_id = K.squeeze(user_id, axis=1)  # remove the axis added by Keras
    pairwise_equal = K.equal(K.expand_dims(user_id, 0), K.expand_dims(user_id, 1))

    pos_neg = K.cast(pairwise_equal, K.floatx()) * 2 - 1
    return K.sum(pairwise_distance * pos_neg, axis=-1) / 2


def pearson(y_true, y_pred):
    # normalizing stage - setting a 0 mean.
    y_true -= K.mean(y_true, axis=-1)
    y_pred -= K.mean(y_pred, axis=-1)
    # normalizing stage - setting a 1 variance
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    # final result
    pearson_correlation = K.sum(y_true * y_pred, axis=-1)
    return pearson_correlation


def abs_pearson(y_true, y_pred):
    # normalizing stage - setting a 0 mean.
    y_true -= K.mean(y_true, axis=-1)
    y_pred -= K.mean(y_pred, axis=-1)
    # normalizing stage - setting a 1 variance
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    # final result
    pearson_correlation = K.sum(y_true * y_pred, axis=-1)
    return tf.math.abs(pearson_correlation)


def _make_riskset(follow_up_times):
    # sort in descending order
    follow_up_times_np = tf.make_ndarray(tf.make_tensor_proto(follow_up_times))
    o = np.argsort(-follow_up_times_np)
    n_samples = follow_up_times_np.shape[0]
    risk_set = np.zeros((n_samples, n_samples))

    for i_start, i_sort in enumerate(o):
        time_i_start = follow_up_times_np[i_sort]
        k = i_start
        while k < n_samples and time_i_start <= follow_up_times_np[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    risk_set_tf = tf.convert_to_tensor(risk_set)
    return risk_set_tf


def _softmax_masked(risk_scores, mask, axis=0, keepdims=None):
    """Compute logsumexp across `axis` for entries where `mask` is true."""
    mask_f = K.cast(mask, risk_scores.dtype)
    risk_scores_masked = risk_scores * mask_f
    # for numerical stability, subtract the maximum value before taking the exponential
    amax = K.max(risk_scores_masked, axis=axis, keepdims=True)
    risk_scores_shift = risk_scores_masked - amax

    exp_masked = K.exp(risk_scores_shift) * mask_f
    exp_sum = K.sum(exp_masked, axis=axis, keepdims=True)
    output = amax + K.log(exp_sum)
    if not keepdims:
        output = K.squeeze(output, axis=axis)
    return output


@tf.function
def cox_hazard_loss(y_true, y_pred):
    # move batch dimension to the end so predictions get broadcast row-wise when multiplying by riskset
    pred_t = K.transpose(y_pred[:, 0])
    events = y_true[:, 0]
    follow_up_times = y_true[:, 1]
    # compute log of sum over risk set for each row
    rr = _softmax_masked(pred_t, _make_riskset(follow_up_times), axis=1, keepdims=True)

    losses = events * (rr - y_pred[:, 0])
    loss = K.mean(losses)
    return loss


def survival_likelihood_loss(n_intervals):
    """Create custom Keras loss function for neural network survival model.

    This function is tightly coupled with the function _survival_tensor defined in tensor_from_file.py which builds the y_true tensor.

    Arguments
        n_intervals: the number of survival time intervals
    Returns
        Custom loss function that can be used with Keras
    """

    def loss(y_true, y_pred):
        """
        To play nicely with the Keras framework y_pred is the same shape as y_true.
        However we only consider the first half (n_intervals) of y_pred.
        Arguments
            y_true: Tensor.
              First half of the values are 1 if individual survived that interval, 0 if not.
              Second half of the values are for individuals who failed, and are 1 for time interval during which failure occurred, 0 for other intervals.
              For example given n_intervals = 3 a sample with prevalent disease will have y_true [0, 0, 0, 1, 0, 0]
              a sample with incident disease occurring in the last time bin will have y_true [1, 1, 0, 0, 0, 1]
              a sample who is lost to follow up (censored) in middle time bin will have y_true [1, 0, 0, 0, 0, 0]
            y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
        Returns
            Vector of losses for this minibatch.
        """
        failure_likelihood = 1. - (y_true[:, n_intervals:] * y_pred[:, 0:n_intervals])  # Loss only for individuals who failed
        survival_likelihood = y_true[:, 0:n_intervals] * y_pred[:, 0:n_intervals]  # Loss for intervals that were survived
        survival_likelihood += 1. - y_true[:, 0:n_intervals]  # No survival loss if interval was censored or failed
        return K.sum(-K.log(K.clip(K.concatenate((survival_likelihood, failure_likelihood)), K.epsilon(), None)), axis=-1)  # return -log likelihood

    return loss

def dice(y_true, y_pred):
    return Dice()(y_true, y_pred)
    return Dice(laplace_smoothing=1e-05).mean_loss(y_true, y_pred)

def per_class_dice(labels):
    dice_fxns = []
    for label_key in labels:
        label_idx = labels[label_key]
        fxn_name = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_name + '_dice(y_true, y_pred):\n'
        string_fxn += '\tdice = tf.keras.losses.Dice()(y_true, y_pred)\n'
        #string_fxn += '\tdice = K.mean(dice, axis=0)['+str(label_idx)+']\n'
        string_fxn += '\treturn dice'

        exec(string_fxn)
        dice_fxn = eval(fxn_name + '_dice')
        dice_fxn = register_keras_serializable()(dice_fxn)
        dice_fxns.append(dice_fxn)

    return dice_fxns

def euclid_dist(v):
    return (v[0] - v[1])**2


def per_class_recall(labels):
    recall_fxns = []
    for label_key in labels:
        label_idx = labels[label_key]
        fxn_name = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_name + '_recall(y_true, y_pred):\n'
        string_fxn += '\ttrue_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0)\n'
        string_fxn += '\tpossible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)\n'
        string_fxn += '\treturn true_positives['+str(label_idx)+'] / (possible_positives['+str(label_idx)+'] + K.epsilon())\n'

        exec(string_fxn)
        recall_fxn = eval(fxn_name + '_recall')
        recall_fxn = register_keras_serializable()(recall_fxn)
        recall_fxns.append(recall_fxn)

    return recall_fxns


def per_class_precision(labels):
    precision_fxns = []

    for label_key in labels:
        label_idx = labels[label_key]
        fxn_name = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_name + '_precision(y_true, y_pred):\n'
        string_fxn += '\ttrue_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0)\n'
        string_fxn += '\tpredicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)\n'
        string_fxn += '\treturn true_positives['+str(label_idx)+'] / (predicted_positives['+str(label_idx)+'] + K.epsilon())\n'

        exec(string_fxn)
        precision_fxn = eval(fxn_name + '_precision')
        precision_fxn = register_keras_serializable()(precision_fxn)
        precision_fxns.append(precision_fxn)

    return precision_fxns


def per_class_recall_3d(labels):
    recall_fxns = []

    for label_key in labels:
        label_idx = labels[label_key]
        fxn_prefix = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_prefix + '_recall(y_true, y_pred):\n'
        string_fxn += '\ttrue_positives = K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0)\n'
        string_fxn += '\tpossible_positives = K.sum(K.sum(K.round(K.clip(y_true, 0, 1)), axis=0), axis=0)\n'
        string_fxn += '\treturn true_positives['+str(label_idx)+'] / (possible_positives['+str(label_idx)+'] + K.epsilon())\n'

        exec(string_fxn)
        recall_fxn = eval(fxn_prefix + '_recall')
        recall_fxn = register_keras_serializable()(recall_fxn)
        recall_fxns.append(recall_fxn)

    return recall_fxns


def per_class_precision_3d(labels):
    precision_fxns = []

    for label_key in labels:
        label_idx = labels[label_key]
        fxn_prefix = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_prefix + '_precision(y_true, y_pred):\n'
        string_fxn += '\ttrue_positives = K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0)\n'
        string_fxn += '\tpredicted_positives = K.sum(K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0), axis=0)\n'
        string_fxn += '\treturn true_positives['+str(label_idx)+'] / (predicted_positives['+str(label_idx)+'] + K.epsilon())\n'

        exec(string_fxn)
        precision_fxn = eval(fxn_prefix + '_precision')
        precision_fxn = register_keras_serializable()(precision_fxn)
        precision_fxns.append(precision_fxn)

    return precision_fxns


def per_class_recall_4d(labels):
    recall_fxns = []

    for label_key in labels:
        label_idx = labels[label_key]
        fxn_prefix = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_prefix + '_recall(y_true, y_pred):\n'
        string_fxn += '\ttrue_positives = K.sum(K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0), axis=0)\n'
        string_fxn += '\tpossible_positives = K.sum(K.sum(K.sum(K.round(K.clip(y_true, 0, 1)), axis=0), axis=0), axis=0)\n'
        string_fxn += '\treturn true_positives['+str(label_idx)+'] / (possible_positives['+str(label_idx)+'] + K.epsilon())\n'

        exec(string_fxn)
        recall_fxn = eval(fxn_prefix + '_recall')
        recall_fxn = register_keras_serializable()(recall_fxn)
        recall_fxns.append(recall_fxn)

    return recall_fxns


def per_class_precision_4d(labels):
    precision_fxns = []

    for label_key in labels:
        label_idx = labels[label_key]
        fxn_prefix = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_prefix + '_precision(y_true, y_pred):\n'
        string_fxn += '\ttrue_positives = K.sum(K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0), axis=0)\n'
        string_fxn += '\tpredicted_positives = K.sum(K.sum(K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0), axis=0), axis=0)\n'
        string_fxn += '\treturn true_positives['+str(label_idx)+'] / (predicted_positives['+str(label_idx)+'] + K.epsilon())\n'

        exec(string_fxn)
        precision_fxn = eval(fxn_prefix + '_precision')
        precision_fxn = register_keras_serializable()(precision_fxn)

        precision_fxns.append(precision_fxn)

    return precision_fxns


def per_class_recall_5d(labels):
    recall_fxns = []

    for label_key in labels:
        label_idx = labels[label_key]
        fxn_prefix = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_prefix + '_recall(y_true, y_pred):\n'
        string_fxn += '\ttrue_positives = K.sum(K.sum(K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0), axis=0), axis=0)\n'
        string_fxn += '\tpossible_positives = K.sum(K.sum(K.sum(K.sum(K.round(K.clip(y_true, 0, 1)), axis=0), axis=0), axis=0), axis=0)\n'
        string_fxn += '\treturn true_positives['+str(label_idx)+'] / (possible_positives['+str(label_idx)+'] + K.epsilon())\n'

        exec(string_fxn)
        recall_fxn = eval(fxn_prefix + '_recall')
        recall_fxn = register_keras_serializable()(recall_fxn)
        recall_fxns.append(recall_fxn)

    return recall_fxns


def per_class_precision_5d(labels):
    precision_fxns = []

    for label_key in labels:
        label_idx = labels[label_key]
        fxn_prefix = label_key.replace('-', '_').replace(' ', '_')
        string_fxn = 'def ' + fxn_prefix + '_precision(y_true, y_pred):\n'
        string_fxn += '\ttrue_positives = K.sum(K.sum(K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0), axis=0), axis=0)\n'
        string_fxn += '\tpredicted_positives = K.sum(K.sum(K.sum(K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0), axis=0), axis=0), axis=0)\n'
        string_fxn += '\treturn true_positives['+str(label_idx)+'] / (predicted_positives['+str(label_idx)+'] + K.epsilon())\n'

        exec(string_fxn)
        precision_fxn = eval(fxn_prefix + '_precision')
        precision_fxn = register_keras_serializable()(precision_fxn)
        precision_fxns.append(precision_fxn)

    return precision_fxns


def get_metric_dict(output_tensor_maps):
    metrics = {}
    losses = []
    loss_weights = []
    for tm in output_tensor_maps:
        loss_weights.append(tm.loss_weight)
        for m in tm.metrics:
            if isinstance(m, str):
                metrics[m] = m
            elif hasattr(m, '__name__'):
                metrics[m.__name__] = m

        if tm.loss == 'categorical_crossentropy':
            losses.append(categorical_crossentropy)
        elif tm.loss == 'sparse_categorical_crossentropy':
            losses.append(sparse_categorical_crossentropy)
        elif tm.loss == 'binary_crossentropy':
            losses.append(binary_crossentropy)
        elif tm.loss == 'mean_absolute_error' or tm.loss == 'mae':
            losses.append(MSE)
        elif tm.loss == 'mean_squared_error' or tm.loss == 'mse':
            losses.append(MSE)
        elif tm.loss == 'cosine_similarity':
            losses.append(CosineSimilarity)
        elif tm.loss == 'log_cosh':
            losses.append(LogCosh)
        elif tm.loss == 'mape':
            losses.append(MAPE)
        elif hasattr(tm.loss,  '__name__'):
            metrics[tm.loss.__name__] = tm.loss
            losses.append(tm.loss)
        else:
            losses.append(tm.loss)

    def loss(y_true, y_pred):
        my_loss = 0
        for my_loss_fxn, loss_weight in zip(losses, loss_weights):
            my_loss += loss_weight * my_loss_fxn(y_true, y_pred)
        return my_loss
    metrics['loss'] = loss

    return metrics


def get_roc_aucs(predictions, truth, labels):
    """Compute ROC AUC for each label of each given model"""
    aucs = dict()

    for model in predictions.keys():
        roc_auc = dict()
        for label_name in labels.keys():
            label_encoding = labels[label_name]
            fpr, tpr,  _ = roc_curve(truth[:, label_encoding], predictions[model][:, label_encoding])
            roc_auc[label_name] = auc(fpr, tpr)
        aucs[model] = roc_auc

    return aucs


def get_precision_recall_aucs(predictions, truth, labels):
    """Compute Precision-Recall AUC for each label of each given model"""
    aucs = dict()

    for model in predictions.keys():
        average_precision = dict()
        for label_name in labels.keys():
            label_encoding = labels[label_name]
            average_precision[label_name] = average_precision_score(truth[:, label_encoding], predictions[model][:, label_encoding])
        aucs[model] = average_precision

    return aucs


def log_aucs(**aucs):
    """Log and tabulate AUCs given as nested dictionaries in the format '{model: {label: auc}}'"""
    def dashes(n): return '-' * n

    header = "{:<40} {:<28} {:<15}"
    row = "{:<40} {:<28} {:<15.4f}"
    width = 90
    logging.info(dashes(width))

    for auc_name, auc_value in aucs.items():
        logging.info(header.format('Model', 'Label', auc_name+' AUC'))
        triplets = []
        for model, model_value in auc_value.items():
            for label, auc in model_value.items():
                triplets.append((model, label, auc))
        for model, label, auc in sorted(triplets, key=lambda x: x[1]):
            logging.info(row.format(model, label, auc))
        logging.info(dashes(width))


def coefficient_of_determination(truth, predictions, eps=1e-6):
    true_mean = np.mean(truth)
    total_sum_of_squares = np.sum((truth - true_mean) * (truth - true_mean))
    residual_sum_of_squares = np.sum((predictions - truth) * (predictions - truth))
    r_squared = 1 - (residual_sum_of_squares / (total_sum_of_squares + eps))
    return r_squared


def get_pearson_coefficients(predictions, truth):
    coefs = dict()
    for model in predictions.keys():
        # corrcoef() returns full covariance matrix
        pearson = np.corrcoef(predictions[model].flatten(), truth.flatten())[1, 0]
        coefs[model] = pearson

    return coefs


def log_pearson_coefficients(coefs, label):
    def dashes(n): return '-' * n

    header = "{:<30} {:<25} {:<15} {:<15}"
    row = "{:<30} {:<25} {:<15.10f} {:<15.10f}"
    width = 85
    logging.info(dashes(width))
    logging.info(header.format('Model', 'Label', 'Pearson R', 'Pearson R^2'))
    for model, coef in coefs.items():
        logging.info(row.format(model, label, coef, coef*coef))
    logging.info(dashes(width))


def _unpack_truth_into_events(truth, intervals):
    event_time = np.argmin(np.diff(truth[:, :intervals]), axis=-1)
    event_time[truth[:, intervals-1] == 1] = intervals-1  # If the sample is never censored set event time to max time
    event_indicator = np.sum(truth[:, intervals:], axis=-1).astype(bool)
    return event_indicator, event_time


def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def concordance_index(prediction, truth, tied_tol=1e-8):
    intervals = truth.shape[-1] // 2
    event_indicator, event_time = _unpack_truth_into_events(truth, intervals)
    estimate = np.cumprod(prediction[:, :intervals], axis=-1)[:, -1]
    order = np.argsort(event_time)
    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()

        # an event should have a higher score
        con = est > est_i
        n_con = con[~ties].sum()

        numerator += n_con + 0.5 * n_ties
        denominator += mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def _estimate_concordance_index(event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    order = np.argsort(event_time)
    comparable, tied_time = _get_comparable(event_indicator, event_time, order)
    if len(comparable) == 0:
        raise ValueError("Data has no comparable pairs, cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-8):
    """Concordance index for right-censored data
    The concordance index is defined as the proportion of all comparable pairs
    in which the predictions and outcomes are concordant.
    Two samples are comparable if (i) both of them experienced an event (at different times),
    or (ii) the one with a shorter observed survival time experienced an event, in which case
    the event-free subject "outlived" the other. A pair is not comparable if they experienced
    events at the same time.
    Concordance intuitively means that two samples were ordered correctly by the model.
    More specifically, two samples are concordant, if the one with a higher estimated
    risk score has a shorter actual survival time.
    When predicted risks are identical for a pair, 0.5 rather than 1 is added to the count
    of concordant pairs.
    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.
    Parameters
    ----------
    event_indicator : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred
    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring
    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event
    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.
    Returns
    -------
    cindex : float
        Concordance index
    concordant : int
        Number of concordant pairs
    discordant : int
        Number of discordant pairs
    tied_risk : int
        Number of pairs having tied estimated risks
    tied_time : int
        Number of comparable pairs sharing the same time
    See also
    --------
    concordance_index_ipcw
        Alternative estimator of the concordance index with less bias.
    References
    ----------
    .. [1] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L., Rosati, R.A,
           "Multivariable prognostic models: issues in developing models,
           evaluating assumptions and adequacy, and measuring and reducing errors",
           Statistics in Medicine, 15(4), 361-87, 1996.
    """
    w = np.ones_like(estimate)
    return _estimate_concordance_index(event_indicator, event_time, estimate, w, tied_tol)


class KernelInceptionDistance(keras.metrics.Metric):
    def __init__(self, name, input_shape, kernel_image_size, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=input_shape), # TODO: handle multi-channel
                keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3])),
                keras.layers.Rescaling(255.0),
                keras.layers.Resizing(height=kernel_image_size, width=kernel_image_size),
                keras.layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kernel_image_size, kernel_image_size, 3),
                    weights="imagenet",
                ),
                keras.layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


class InceptionScore(keras.metrics.Metric):
    def __init__(self, name, input_shape, kernel_image_size, **kwargs):
        super().__init__(name=name, **kwargs)

        # Inception score is estimated per batch and averaged across batches
        self.is_tracker = keras.metrics.Mean(name="is_tracker")

        # A pretrained InceptionV3 is used without its classification layer
        # We preprocess the images as during the pretraining of InceptionV3
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=input_shape),  # TODO: handle multi-channel
                keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3])),  # Ensure 3 channels
                keras.layers.Rescaling(255.0),
                keras.layers.Resizing(height=kernel_image_size, width=kernel_image_size),
                keras.layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=True,  # Include the classification layer for IS
                    input_shape=(kernel_image_size, kernel_image_size, 3),
                    weights="imagenet",
                ),
            ],
            name="inception_encoder",
        )

    def update_state(self, generated_images, sample_weight=None):
        # Get the predicted class probabilities from the InceptionV3 model
        logits = self.encoder(generated_images, training=False)
        softmax_probs = tf.nn.softmax(logits, axis=-1)

        # Compute p(y|x) for each generated image (softmax probabilities)
        p_y_given_x = softmax_probs

        # Compute the marginal distribution p(y) across all generated images
        p_y = tf.reduce_mean(p_y_given_x, axis=0)

        # Compute KL divergence between p(y|x) and p(y) for each image
        kl_divergence = tf.reduce_sum(p_y_given_x * (tf.math.log(p_y_given_x) - tf.math.log(p_y)), axis=-1)

        # Inception score is the exponentiation of the mean KL divergence
        is_score = tf.exp(tf.reduce_mean(kl_divergence))

        # Update the average IS estimate
        self.is_tracker.update_state(is_score)

    def result(self):
        return self.is_tracker.result()

    def reset_state(self):
        self.is_tracker.reset_state()


class MultiScaleSSIM(keras.metrics.Metric):
    def __init__(self, name="multi_scale_ssim", **kwargs):
        super(MultiScaleSSIM, self).__init__(name=name, **kwargs)
        self.total_ssim = self.add_weight(name="total_ssim", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, max_val, sample_weight=None):
        # Calculate MS-SSIM for the batch
        ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=max_val, power_factors=[0.25, 0.25, 0.25, 0.25])
        if sample_weight is not None:
            ssim = tf.multiply(ssim, sample_weight)

        # Update total MS-SSIM and count
        self.total_ssim.assign_add(tf.reduce_sum(ssim))
        self.count.assign_add(tf.cast(tf.size(ssim), tf.float32))

    def result(self):
        # Return the mean MS-SSIM over all batches
        return tf.divide(self.total_ssim, self.count)

    def reset_state(self):
        # Reset the metric state variables
        self.total_ssim.assign(0.0)
        self.count.assign(0.0)


def calculate_kid(real, generated):
    """
    Compute the Kernel Inception Distance (KID) between two sets of images.

    Parameters:
      real: np.ndarray of shape (N, 224, 224, 1) – real images (grayscale)
      generated: np.ndarray of shape (N, 224, 224, 1) – generated images (grayscale)

    Returns:
      kid_value: float – the estimated KID value.

    Note: This function assumes that the image pixel values are in the [0, 255] range.
    """
    # Convert grayscale images to 3 channels by repeating the channel dimension
    if real.shape[-1] == 1:
        real = np.repeat(real, 3, axis=-1)
    if generated.shape[-1] == 1:
        generated = np.repeat(generated, 3, axis=-1)

    # Convert to TensorFlow tensors and resize images to 299x299 (the InceptionV3 input size)
    real_tensor = tf.convert_to_tensor(real, dtype=tf.float32)
    generated_tensor = tf.convert_to_tensor(generated, dtype=tf.float32)

    real_resized = tf.image.resize(real_tensor, (299, 299)).numpy()
    generated_resized = tf.image.resize(generated_tensor, (299, 299)).numpy()

    # Preprocess images for InceptionV3
    # real_preprocessed = preprocess_input(real_resized)
    # generated_preprocessed = preprocess_input(generated_resized)

    # Load InceptionV3 model for feature extraction.
    # Using include_top=False with global average pooling gives a 2048-D feature vector per image.
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Extract features for both sets.
    features_real = model.predict(real_resized, verbose=0)
    features_generated = model.predict(generated_resized, verbose=0)

    # Determine feature dimension
    d = features_real.shape[1]

    # Define the third-order polynomial kernel: k(x, y) = (x^T y / d + 1)^3
    def poly_kernel(X, Y):
        return (np.dot(X, Y.T) / d + 1) ** 3

    subset_real = features_real
    subset_generated = features_generated

    # Compute the kernel matrices
    K_rr = poly_kernel(subset_real, subset_real)
    K_gg = poly_kernel(subset_generated, subset_generated)
    K_rg = poly_kernel(subset_real, features_generated)

    m = subset_real.shape[0]

    # For an unbiased estimator, exclude the diagonal elements from the intra-set kernel sums.
    sum_K_rr = np.sum(K_rr) - np.sum(np.diag(K_rr))
    sum_K_gg = np.sum(K_gg) - np.sum(np.diag(K_gg))

    # Unbiased MMD^2 estimate (i.e. KID) as described in the literature:
    kid_value = (sum_K_rr / (m * (m - 1)) +
                 sum_K_gg / (m * (m - 1)) -
                 2 * np.mean(K_rg))

    return kid_value


def calculate_fid(real, generated):
    """
    Calculate the Frechet Inception Distance (FID) between two sets of images.

    Parameters:
        real (np.ndarray): Array of real images.
        generated (np.ndarray): Array of generated images.
        num_subsets (int): Number of subsets to sample for calculating mean FID.
        subset_size (int): Number of images in each subset.

    Returns:
        float: Mean FID value over the subsets.
    """

    # Convert grayscale images to 3 channels by repeating the channel dimension
    if real.shape[-1] == 1:
        real = np.repeat(real, 3, axis=-1)
    if generated.shape[-1] == 1:
        generated = np.repeat(generated, 3, axis=-1)

    # Convert to TensorFlow tensors and resize images to 299x299 (the InceptionV3 input size)
    real_tensor = tf.convert_to_tensor(real, dtype=tf.float32)
    generated_tensor = tf.convert_to_tensor(generated, dtype=tf.float32)

    real_resized = tf.image.resize(real_tensor, (299, 299)).numpy()
    generated_resized = tf.image.resize(generated_tensor, (299, 299)).numpy()

    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    n_real = real.shape[0]
    n_generated = generated.shape[0]

    subset_real = real_resized
    subset_generated = generated_resized

    # Ensure images are resized to (299,299) as expected by InceptionV3.
    # If they are already 299x299, this step will have minimal cost.
    if subset_real.shape[1] != 299 or subset_real.shape[2] != 299:
        subset_real = tf.image.resize(subset_real, (299, 299)).numpy()
    if subset_generated.shape[1] != 299 or subset_generated.shape[2] != 299:
        subset_generated = tf.image.resize(subset_generated, (299, 299)).numpy()

    # Preprocess the images as required by InceptionV3 (scaling pixels to [-1, 1])
    subset_real = preprocess_input(subset_real)
    subset_generated = preprocess_input(subset_generated)

    # Extract features using the InceptionV3 model.
    features_real = model.predict(subset_real, verbose=0)
    features_generated = model.predict(subset_generated, verbose=0)

    # Compute mean and covariance for each set of features.
    mu_real = np.mean(features_real, axis=0)
    mu_generated = np.mean(features_generated, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_generated = np.cov(features_generated, rowvar=False)

    # Compute the squared difference between the means.
    diff = mu_real - mu_generated
    diff_squared = np.sum(diff ** 2)

    # Compute the square root of the product of covariances.
    covmean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
    # Handle numerical errors: if imaginary numbers appear, discard the imaginary part.
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the FID for the subset.
    fid = diff_squared + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * np.trace(covmean)
    return fid

class JsonlMetricsCallback(tf.keras.callbacks.Callback):
    """
    Writes one JSON line per epoch with training + validation metrics.

    Example output line:
    {"epoch": 7, "time": 12.34, "loss": 0.42, "val_loss": 0.38, "val_auc_mean": 0.81}
    """

    def __init__(self, output_dir, filename="loss_metrics.json", flush=True):
        super().__init__()
        self.output_dir = output_dir
        self.filename = filename
        self.flush = flush
        self._epoch_start_time = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(self.output_dir, self.filename)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed = None
        if self._epoch_start_time is not None:
            elapsed = time.time() - self._epoch_start_time

        record = {
            "epoch": int(epoch),
            "time": elapsed,
        }

        # Convert tensors / numpy scalars → Python scalars
        for k, v in logs.items():
            try:
                record[k] = float(v)
            except Exception:
                pass

        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            if self.flush:
                f.flush()
                os.fsync(f.fileno())




def _register_all(module_globals):
    for name, obj in module_globals.items():
        if callable(obj) and not name.startswith("_"):
            module_globals[name] = register_keras_serializable()(obj)

_register_all(globals())
