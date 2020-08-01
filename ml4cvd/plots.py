# Imports: standard library
import os
import re
import math
import hashlib
import logging
from typing import Dict, List, Tuple, Union, Callable, Optional
from datetime import datetime
from collections import Counter, OrderedDict, defaultdict

# Imports: third party
import h5py
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import manifold
from sklearn.metrics import (
    auc,
    roc_curve,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
)
from matplotlib.ticker import NullFormatter
from sklearn.calibration import calibration_curve
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# Imports: first party
from ml4cvd.defines import (
    PDF_EXT,
    IMAGE_EXT,
    TENSOR_EXT,
    ECG_REST_LEADS,
    ECG_DATE_FORMAT,
    ECG_DATETIME_FORMAT,
)
from ml4cvd.metrics import concordance_index, coefficient_of_determination
from ml4cvd.TensorMap import TensorMap

# fmt: off
# need matplotlib -> Agg -> pyplot
import matplotlib  # isort:skip
matplotlib.use("Agg")  # isort:skip
from matplotlib import pyplot as plt  # isort:skip
# fmt: on


RECALL_LABEL = "Sensitivity | True Positive Rate | TP/(TP+FN)"
FALLOUT_LABEL = "1 - Specificity | False Positive Rate | FP/(FP+TN)"
PRECISION_LABEL = "Precision | Positive Predictive Value | TP/(TP+FP)"

SUBPLOT_SIZE = 8

COLOR_ARRAY = [
    "tan",
    "indigo",
    "cyan",
    "pink",
    "purple",
    "blue",
    "chartreuse",
    "deepskyblue",
    "green",
    "salmon",
    "aqua",
    "magenta",
    "aquamarine",
    "red",
    "coral",
    "tomato",
    "grey",
    "black",
    "maroon",
    "hotpink",
    "steelblue",
    "orange",
    "papayawhip",
    "wheat",
    "chocolate",
    "darkkhaki",
    "gold",
    "orange",
    "crimson",
    "slategray",
    "violet",
    "cadetblue",
    "midnightblue",
    "darkorchid",
    "paleturquoise",
    "plum",
    "lime",
    "teal",
    "peru",
    "silver",
    "darkgreen",
    "rosybrown",
    "firebrick",
    "saddlebrown",
    "dodgerblue",
    "orangered",
]

ECG_REST_PLOT_DEFAULT_YRANGE = 3.0
ECG_REST_PLOT_MAX_YRANGE = 10.0
ECG_REST_PLOT_LEADS = [
    ["strip_I", "strip_aVR", "strip_V1", "strip_V4"],
    ["strip_II", "strip_aVL", "strip_V2", "strip_V5"],
    ["strip_III", "strip_aVF", "strip_V3", "strip_V6"],
]
ECG_REST_PLOT_MEDIAN_LEADS = [
    ["median_I", "median_aVR", "median_V1", "median_V4"],
    ["median_II", "median_aVL", "median_V2", "median_V5"],
    ["median_III", "median_aVF", "median_V3", "median_V6"],
]
ECG_REST_PLOT_AMP_LEADS = [
    [0, 3, 6, 9],
    [1, 4, 7, 10],
    [2, 5, 8, 11],
]


def evaluate_predictions(
    tm: TensorMap,
    y_predictions: np.ndarray,
    y_truth: np.ndarray,
    title: str,
    folder: str,
    test_paths: List[str] = None,
    max_melt: int = 30000,
    rocs: List[Tuple[np.ndarray, np.ndarray, Dict[str, int]]] = [],
    scatters: List[Tuple[np.ndarray, np.ndarray, str, List[str]]] = [],
    data_split: str = "test",
) -> Dict[str, float]:
    """Evaluate predictions for a given TensorMap with truth data and plot the appropriate metrics.
    Accumulates data in the rocs and scatters lists to facilitate subplotting.

    :param tm: The TensorMap predictions to evaluate
    :param y_predictions: The predictions
    :param y_truth: The truth
    :param title: A title for the plots
    :param folder: The folder to save the plots at
    :param test_paths: The tensor paths that were predicted
    :param max_melt: For multi-dimensional prediction the maximum number of prediction to allow in the flattened array
    :param rocs: (output) List of Tuples which are inputs for ROC curve plotting to allow subplotting downstream
    :param scatters: (output) List of Tuples which are inputs for scatter plots to allow subplotting downstream
    :param data_split: The data split being evaluated (train, valid, or test)
    :return: Dictionary of performance metrics with string keys for labels and float values
    """
    performance_metrics = {}
    if tm.is_categorical() and tm.static_axes() == 1:
        logging.info(
            f"{tm.name} has channel map: {tm.channel_map}"
            f" with {y_predictions.shape[0]} examples in the test set.\n"
            f"Sum Truth:{np.sum(y_truth, axis=0)} \nSum pred"
            f" :{np.sum(y_predictions, axis=0)}",
        )
        plot_precision_recall_per_class(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            prefix=folder,
            data_split=data_split,
        )
        plot_prediction_calibration(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            prefix=folder,
            data_split=data_split,
        )
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.is_categorical() and tm.static_axes() == 2:
        melt_shape = (
            y_predictions.shape[0] * y_predictions.shape[1],
            y_predictions.shape[2],
        )
        idx = np.random.choice(
            np.arange(melt_shape[0]), min(melt_shape[0], max_melt), replace=False,
        )
        y_predictions = y_predictions.reshape(melt_shape)[idx]
        y_truth = y_truth.reshape(melt_shape)[idx]
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        performance_metrics.update(
            plot_precision_recall_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        plot_prediction_calibration(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            prefix=folder,
            data_split=data_split,
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.is_categorical() and tm.static_axes() == 3:
        melt_shape = (
            y_predictions.shape[0] * y_predictions.shape[1] * y_predictions.shape[2],
            y_predictions.shape[3],
        )
        idx = np.random.choice(
            np.arange(melt_shape[0]), min(melt_shape[0], max_melt), replace=False,
        )
        y_predictions = y_predictions.reshape(melt_shape)[idx]
        y_truth = y_truth.reshape(melt_shape)[idx]
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        performance_metrics.update(
            plot_precision_recall_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        plot_prediction_calibration(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            prefix=folder,
            data_split=data_split,
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.is_categorical() and tm.static_axes() == 4:
        melt_shape = (
            y_predictions.shape[0]
            * y_predictions.shape[1]
            * y_predictions.shape[2]
            * y_predictions.shape[3],
            y_predictions.shape[4],
        )
        idx = np.random.choice(
            np.arange(melt_shape[0]), min(melt_shape[0], max_melt), replace=False,
        )
        y_predictions = y_predictions.reshape(melt_shape)[idx]
        y_truth = y_truth.reshape(melt_shape)[idx]
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        performance_metrics.update(
            plot_precision_recall_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        plot_prediction_calibration(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            prefix=folder,
            data_split=data_split,
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.is_survival_curve():
        performance_metrics.update(
            plot_survival(
                y_predictions,
                y_truth,
                title,
                days_window=tm.days_window,
                prefix=folder,
            ),
        )
        plot_survival_curves(
            y_predictions,
            y_truth,
            title,
            days_window=tm.days_window,
            prefix=folder,
            paths=test_paths,
        )
        time_steps = tm.shape[-1] // 2
        days_per_step = 1 + tm.days_window // time_steps
        predictions_at_end = (
            1 - np.cumprod(y_predictions[:, :time_steps], axis=-1)[:, -1]
        )
        events_at_end = np.cumsum(y_truth[:, time_steps:], axis=-1)[:, -1]
        follow_up = np.cumsum(y_truth[:, :time_steps], axis=-1)[:, -1] * days_per_step
        logging.info(
            f"Shapes event {events_at_end.shape}, preds shape"
            f" {predictions_at_end.shape} new ax shape"
            f" {events_at_end[:, np.newaxis].shape}",
        )
        calibration_title = f"{title}_at_{tm.days_window}_days"
        plot_prediction_calibration(
            prediction=predictions_at_end[:, np.newaxis],
            truth=events_at_end[:, np.newaxis],
            labels={tm.name: 0},
            title=calibration_title,
            prefix=folder,
            data_split=data_split,
        )
        plot_survivorship(
            events_at_end,
            follow_up,
            predictions_at_end,
            tm.name,
            folder,
            tm.days_window,
        )
    elif tm.is_language():
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        performance_metrics.update(
            plot_precision_recall_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                prefix=folder,
                data_split=data_split,
            ),
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.static_axes() > 1 or tm.is_mesh():
        prediction_flat = tm.rescale(y_predictions).flatten()[:max_melt]
        truth_flat = tm.rescale(y_truth).flatten()[:max_melt]
        if prediction_flat.shape[0] == truth_flat.shape[0]:
            performance_metrics.update(
                plot_scatter(prediction_flat, truth_flat, title, prefix=folder),
            )
    elif tm.is_continuous():
        if tm.sentinel is not None:
            y_predictions = y_predictions[y_truth != tm.sentinel, np.newaxis]
            y_truth = y_truth[y_truth != tm.sentinel, np.newaxis]
        performance_metrics.update(
            plot_scatter(
                tm.rescale(y_predictions),
                tm.rescale(y_truth),
                title,
                prefix=folder,
                paths=test_paths,
            ),
        )
        scatters.append(
            (tm.rescale(y_predictions), tm.rescale(y_truth), title, test_paths),
        )
    else:
        logging.warning(f"No evaluation clause for tensor map {tm.name}")

    if tm.name == "median":
        plot_waves(y_predictions, y_truth, "median_waves_" + title, folder)

    return performance_metrics


def plot_metric_history(history, training_steps: int, title: str, prefix="./figures/"):
    plt.rcParams["font.size"] = 14
    row = 0
    col = 0
    total_plots = int(
        len(history.history) / 2,
    )  # divide by 2 because we plot validation and train histories together
    cols = max(2, int(math.ceil(math.sqrt(total_plots))))
    rows = max(2, int(math.ceil(total_plots / cols)))
    f, axes = plt.subplots(
        rows, cols, figsize=(int(cols * SUBPLOT_SIZE), int(rows * SUBPLOT_SIZE)),
    )
    for k in sorted(history.history.keys()):
        if not k.startswith("val_") and not k.startswith("no_"):
            if isinstance(history.history[k][0], LearningRateSchedule):
                history.history[k] = [
                    history.history[k][0](i * training_steps)
                    for i in range(len(history.history[k]))
                ]
            axes[row, col].plot(
                list(range(1, len(history.history[k]) + 1)), history.history[k],
            )
            k_split = str(k).replace("output_", "").split("_")
            k_title = " ".join(OrderedDict.fromkeys(k_split))
            axes[row, col].set_title(k_title)
            axes[row, col].set_xlabel("epoch")
            if "val_" + k in history.history:
                axes[row, col].plot(
                    list(range(1, len(history.history["val_" + k]) + 1)),
                    history.history["val_" + k],
                )
                labels = ["train", "valid"]
            else:
                labels = [k]
            axes[row, col].legend(labels, loc="upper left")

            row += 1
            if row == rows:
                row = 0
                col += 1
                if col >= cols:
                    break

    plt.tight_layout()
    figure_path = os.path.join(prefix, "metric_history_" + title + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path, bbox_inches="tight")
    plt.clf()
    logging.info(f"Saved learning curves at: {figure_path}")


def _find_negative_label_index(labels: Dict[str, int], key_prefix: str = "no_") -> int:
    """Given a set of labels and their values, return the index of the negative label"""
    negative_label_index = 0
    for index, label in enumerate(labels):
        if label.startswith(key_prefix):
            negative_label_index = index
    return negative_label_index


def plot_prediction_calibration(
    prediction: np.ndarray,
    truth: np.ndarray,
    labels: Dict[str, int],
    title: str,
    prefix: str = "./figures/",
    n_bins: int = 10,
    data_split: str = "test",
):
    """Plot calibration performance and compute Brier Score.

    :param prediction: Array of probabilistic predictions with shape (num_samples, num_classes)
    :param truth: The true classifications of each class, one hot encoded of shape (num_samples, num_classes)
    :param labels: Dictionary mapping strings describing each class to their corresponding index in the arrays
    :param title: The name of this plot
    :param prefix: Optional path prefix where the plot will be saved
    :param n_bins: Number of bins to quantize predictions into
    :param data_split: The data split being plotted (train, valid, or test)
    """
    plt.rcParams["font.size"] = 14
    _, (ax1, ax3, ax2) = plt.subplots(3, figsize=(SUBPLOT_SIZE, 2 * SUBPLOT_SIZE))

    true_sums = np.sum(truth, axis=0)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated Brier score: 0.0")
    ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated Brier score: 0.0")

    if len(labels) == 2:
        negative_label_idx = _find_negative_label_index(labels=labels, key_prefix="no_")

    for idx, label in enumerate(labels):
        if len(labels) == 2 and idx == negative_label_idx:
            continue

        y_true = truth[..., labels[label]]
        y_prob = prediction[..., labels[label]]
        color = _hash_string_to_color(label)
        brier_score = brier_score_loss(
            y_true, prediction[..., labels[label]], pos_label=1,
        )
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins,
        )
        ax3.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"{label} Brier score: {brier_score:0.3f}",
            color=color,
        )
        ax2.hist(
            y_prob,
            range=(0, 1),
            bins=n_bins,
            label=f"{label} n={true_sums[labels[label]]:.0f}",
            histtype="step",
            lw=2,
            color=color,
        )

        bins = stats.mstats.mquantiles(y_prob, np.arange(0.0, 1.0, 1.0 / n_bins))
        binids = np.digitize(y_prob, bins) - 1

        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        ax1.plot(
            prob_pred,
            prob_true,
            "s-",
            label=f"{label} Brier score: {brier_score:0.3f}",
            color=color,
        )
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'{title.replace("_", " ")}\nCalibration plot (equally sized bins)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    ax3.set_title("Calibration plot (equally spaced bins)")
    plt.tight_layout()

    figure_path = os.path.join(
        prefix, "calibrations_" + title + "_" + data_split + IMAGE_EXT,
    )
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path, bbox_inches="tight")
    logging.info(f"Saved calibration plot at: {figure_path}")
    plt.clf()


def plot_scatter(
    prediction, truth, title, prefix="./figures/", paths=None, top_k=3, alpha=0.5,
):
    margin = float((np.max(truth) - np.min(truth)) / 100)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(SUBPLOT_SIZE, 2 * SUBPLOT_SIZE))
    ax1.plot(
        [np.min(truth), np.max(truth)], [np.min(truth), np.max(truth)], linewidth=2,
    )
    ax1.plot(
        [np.min(prediction), np.max(prediction)],
        [np.min(prediction), np.max(prediction)],
        linewidth=4,
    )
    pearson = np.corrcoef(prediction.flatten(), truth.flatten())[
        1, 0,
    ]  # corrcoef returns full covariance matrix
    big_r_squared = coefficient_of_determination(truth, prediction)
    logging.info(
        f"Pearson:{pearson:0.3f} r^2:{pearson*pearson:0.3f} R^2:{big_r_squared:0.3f}",
    )
    ax1.scatter(
        prediction,
        truth,
        label=(
            f"Pearson:{pearson:0.3f} r^2:{pearson*pearson:0.3f}"
            f" R^2:{big_r_squared:0.3f}"
        ),
        marker=".",
        alpha=alpha,
    )
    if paths is not None:
        diff = np.abs(prediction - truth)
        arg_sorted = diff[:, 0].argsort()
        # The path of the best prediction, ie the inlier
        _text_on_plot(
            ax1,
            prediction[arg_sorted[0]] + margin,
            truth[arg_sorted[0]] + margin,
            os.path.basename(paths[arg_sorted[0]]),
        )
        # Plot the paths of the worst predictions ie the outliers
        for idx in arg_sorted[-top_k:]:
            _text_on_plot(
                ax1,
                prediction[idx] + margin,
                truth[idx] + margin,
                os.path.basename(paths[idx]),
            )

    ax1.set_xlabel("Predictions")
    ax1.set_ylabel("Actual")
    ax1.set_title(title + "\n")
    ax1.legend(loc="lower right")

    sns.distplot(prediction, label="Predicted", color="r", ax=ax2)
    sns.distplot(truth, label="Truth", color="b", ax=ax2)
    ax2.legend(loc="upper left")

    figure_path = os.path.join(prefix, "scatter_" + title + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    logging.info("Try to save scatter plot at: {}".format(figure_path))
    plt.savefig(figure_path)
    plt.clf()
    return {title + "_pearson": pearson}


def plot_scatters(
    predictions, truth, title, prefix="./figures/", paths=None, top_k=3, alpha=0.5,
):
    margin = float((np.max(truth) - np.min(truth)) / 100)
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))
    plt.plot([np.min(truth), np.max(truth)], [np.min(truth), np.max(truth)])
    for k in predictions:
        color = _hash_string_to_color(k)
        pearson = np.corrcoef(predictions[k].flatten(), truth.flatten())[
            1, 0,
        ]  # corrcoef returns full covariance matrix
        r2 = pearson * pearson
        big_r2 = coefficient_of_determination(truth.flatten(), predictions[k].flatten())
        plt.plot(
            [np.min(predictions[k]), np.max(predictions[k])],
            [np.min(predictions[k]), np.max(predictions[k])],
            color=color,
        )
        plt.scatter(
            predictions[k],
            truth,
            color=color,
            label=str(k) + f" Pearson:{pearson:0.3f} r^2:{r2:0.3f} R^2:{big_r2:0.3f}",
            marker=".",
            alpha=alpha,
        )
        if paths is not None:
            diff = np.abs(predictions[k] - truth)
            arg_sorted = diff[:, 0].argsort()
            _text_on_plot(
                plt,
                predictions[k][arg_sorted[0]] + margin,
                truth[arg_sorted[0]] + margin,
                os.path.basename(paths[arg_sorted[0]]),
            )
            for idx in arg_sorted[-top_k:]:
                _text_on_plot(
                    plt,
                    predictions[k][idx] + margin,
                    truth[idx] + margin,
                    os.path.basename(paths[idx]),
                )
    plt.xlabel("Predictions")
    plt.ylabel("Actual")
    plt.title(title + "\n")
    plt.legend(loc="upper left")

    figure_path = os.path.join(prefix, "scatters_" + title + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    logging.info("Saved scatter plot at: {}".format(figure_path))


def subplot_scatters(
    scatters: List[Tuple[np.ndarray, np.ndarray, str, Optional[List[str]]]],
    prefix: str = "./figures/",
    top_k: int = 3,
    alpha: float = 0.5,
):
    row = 0
    col = 0
    total_plots = len(scatters)
    cols = max(2, int(math.ceil(math.sqrt(total_plots))))
    rows = max(2, int(math.ceil(total_plots / cols)))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * SUBPLOT_SIZE, rows * SUBPLOT_SIZE),
    )
    for prediction, truth, title, paths in scatters:
        axes[row, col].plot(
            [np.min(truth), np.max(truth)], [np.min(truth), np.max(truth)],
        )
        axes[row, col].plot(
            [np.min(prediction), np.max(prediction)],
            [np.min(prediction), np.max(prediction)],
        )
        axes[row, col].scatter(prediction, truth, marker=".", alpha=alpha)
        margin = float((np.max(truth) - np.min(truth)) / 100)

        # If tensor paths are provided, plot file names of top_k outliers and #1 inlier
        if paths is not None:
            diff = np.abs(prediction - truth)
            arg_sorted = diff[:, 0].argsort()
            # The path of the best prediction, ie the inlier
            _text_on_plot(
                axes[row, col],
                prediction[arg_sorted[0]] + margin,
                truth[arg_sorted[0]] + margin,
                os.path.basename(paths[arg_sorted[0]]),
            )
            # Plot the paths of the worst predictions ie the outliers
            for idx in arg_sorted[-top_k:]:
                _text_on_plot(
                    axes[row, col],
                    prediction[idx] + margin,
                    truth[idx] + margin,
                    os.path.basename(paths[idx]),
                )
        axes[row, col].set_xlabel("Predictions")
        axes[row, col].set_ylabel("Actual")
        axes[row, col].set_title(title + "\n")
        pearson = np.corrcoef(prediction.flatten(), truth.flatten())[1, 0]
        r2 = pearson * pearson
        big_r2 = coefficient_of_determination(truth.flatten(), prediction.flatten())
        axes[row, col].text(
            0,
            1,
            f"Pearson:{pearson:0.3f} r^2:{r2:0.3f} R^2:{big_r2:0.3f}",
            verticalalignment="bottom",
            transform=axes[row, col].transAxes,
        )

        row += 1
        if row == rows:
            row = 0
            col += 1
            if col >= cols:
                break

    figure_path = prefix + "scatters_together" + IMAGE_EXT
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    logging.info(f"Saved scatters together at: {figure_path}")


def subplot_comparison_scatters(
    scatters: List[Tuple[Dict[str, np.ndarray], np.ndarray, str, Optional[List[str]]]],
    prefix: str = "./figures/",
    top_k: int = 3,
    alpha: float = 0.5,
):
    row = 0
    col = 0
    total_plots = len(scatters)
    cols = max(2, int(math.ceil(math.sqrt(total_plots))))
    rows = max(2, int(math.ceil(total_plots / cols)))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * SUBPLOT_SIZE, rows * SUBPLOT_SIZE),
    )
    for predictions, truth, title, paths in scatters:
        for k in predictions:
            c = _hash_string_to_color(title + k)
            pearson = np.corrcoef(predictions[k].flatten(), truth.flatten())[1, 0]
            r2 = pearson * pearson
            big_r2 = coefficient_of_determination(
                truth.flatten(), predictions[k].flatten(),
            )
            axes[row, col].plot(
                [np.min(predictions[k]), np.max(predictions[k])],
                [np.min(predictions[k]), np.max(predictions[k])],
                color=c,
            )
            axes[row, col].scatter(
                predictions[k],
                truth,
                color=c,
                label=f"{k} r:{pearson:0.3f} r^2:{r2:0.3f} R^2:{big_r2:0.3f}",
                marker=".",
                alpha=alpha,
            )
            axes[row, col].legend(loc="upper left")
            if paths is not None:
                margin = float((np.max(truth) - np.min(truth)) / 100)
                diff = np.abs(predictions[k] - truth)
                arg_sorted = diff[:, 0].argsort()
                _text_on_plot(
                    axes[row, col],
                    predictions[k][arg_sorted[0]] + margin,
                    truth[arg_sorted[0]] + margin,
                    os.path.basename(paths[arg_sorted[0]]),
                )
                for idx in arg_sorted[-top_k:]:
                    _text_on_plot(
                        axes[row, col],
                        predictions[k][idx] + margin,
                        truth[idx] + margin,
                        os.path.basename(paths[idx]),
                    )
        axes[row, col].set_xlabel("Predictions")
        axes[row, col].set_ylabel("Actual")
        axes[row, col].set_title(title + "\n")

        row += 1
        if row == rows:
            row = 0
            col += 1
            if col >= cols:
                break

    figure_path = os.path.join(prefix, "scatters_compared_together" + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    logging.info(f"Saved scatter comparisons together at: {figure_path}")


def plot_survivorship(
    events: np.ndarray,
    days_follow_up: np.ndarray,
    predictions: np.ndarray,
    title: str,
    prefix: str = "./figures/",
    days_window: int = 1825,
):
    """Plot Kaplan-Meier survivorship curves and stratify by median model prediction.
    All input arrays have the same shape: (num_samples,)

    :param events: Array indicating if each sample had an event (1) or not (0) by the end of follow up
    :param days_follow_up: Array with the total days of follow up for each sample
    :param predictions: Array with model predictions of an event before the end of follow up.
    :param title: Title for the plot
    :param prefix: Path prefix where plot will be saved
    :param days_window: Maximum days of follow up
    """
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))
    days_sorted_index = np.argsort(days_follow_up)
    days_sorted = days_follow_up[days_sorted_index]
    alive_per_step = len(events)
    sick_per_step = 0
    censored = 0
    survivorship = [1.0]
    real_survivorship = [1.0]
    for cur_day, day_index in enumerate(days_sorted_index):
        if days_follow_up[day_index] > days_window:
            break
        sick_per_step += events[day_index]
        censored += 1 - events[day_index]
        alive_per_step -= events[day_index]
        survivorship.append(1 - (sick_per_step / (alive_per_step + sick_per_step)))
        real_survivorship.append(
            real_survivorship[cur_day] * (1 - (events[day_index] / alive_per_step)),
        )
    logging.info(
        f"Cur day {cur_day} totL {len(real_survivorship)} totL {len(days_sorted)} First"
        f" day {days_sorted[0]} Last day, day {days_follow_up[day_index]}, censored"
        f" {censored}",
    )
    plt.plot(
        [0] + days_sorted[: cur_day + 1],
        real_survivorship[: cur_day + 1],
        marker=".",
        label="Survivorship",
    )
    groups = ["High risk", "Low risk"]
    predicted_alive = {g: len(events) // 2 for g in groups}
    predicted_sick = {g: 0 for g in groups}
    predicted_days = defaultdict(list)
    predicted_survival = defaultdict(list)
    threshold = np.median(predictions)
    for cur_day, day_index in enumerate(days_sorted_index):
        if days_follow_up[day_index] > days_window:
            break
        group = "High risk" if predictions[day_index] > threshold else "Low risk"
        predicted_sick[group] += events[day_index]
        predicted_survival[group].append(
            1
            - (
                predicted_sick[group] / (predicted_alive[group] + predicted_sick[group])
            ),
        )
        predicted_alive[group] -= events[day_index]
        predicted_days[group].append(days_follow_up[day_index])

    for group in groups:
        plt.plot(
            [0] + predicted_days[group],
            [1] + predicted_survival[group],
            color="r" if "High" in group else "g",
            marker="o",
            label=f"{group} group had {predicted_sick[group]} events",
        )
    plt.title(
        f"{title}\nEnrolled: {len(events)}, Censored: {censored:.0f},"
        f" {100 * (censored / len(events)):2.1f}%, Events: {sick_per_step:.0f},"
        f" {100 * (sick_per_step / len(events)):2.1f}%\nMax follow up: {days_window}"
        f" days, {days_window // 365} years.",
    )
    plt.xlabel("Follow up time (days)")
    plt.ylabel("Proportion Surviving")
    plt.legend(loc="lower left")

    figure_path = os.path.join(
        prefix, f"survivorship_fu_{days_window}_{title}{IMAGE_EXT}",
    )
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    logging.info(f"Try to save survival plot at: {figure_path}")
    plt.savefig(figure_path)
    return {}


def plot_survival(
    prediction: np.ndarray,
    truth: np.ndarray,
    title: str,
    days_window: int,
    prefix: str = "./figures/",
) -> Dict[str, float]:
    """Plot Kaplan-Meier survivorship and predicted proportion surviving, calculate and return C-Index

    :param prediction: Array with model predictions of an event at each time step, with shape (num_samples, intervals*2).
    :param truth: Array with survival at each time step followed by events, shape is (num_samples, intervals*2)
    :param title: Title for the plot
    :param days_window: Maximum days of follow up
    :param prefix: Path prefix where plot will be saved

    :return: Dictionary mapping metric names to their floating point values
    """
    c_index, concordant, discordant, tied_risk, tied_time = concordance_index(
        prediction, truth,
    )
    logging.info(
        f"C-index:{c_index} concordant:{concordant} discordant:{discordant}"
        f" tied_risk:{tied_risk} tied_time:{tied_time}",
    )
    intervals = truth.shape[-1] // 2
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))

    cumulative_sick = np.cumsum(np.sum(truth[:, intervals:], axis=0))
    cumulative_censored = (
        truth.shape[0] - np.sum(truth[:, :intervals], axis=0)
    ) - cumulative_sick
    alive_per_step = np.sum(truth[:, :intervals], axis=0)
    sick_per_step = np.sum(truth[:, intervals:], axis=0)
    survivorship = np.cumprod(1 - (sick_per_step / alive_per_step))
    logging.info(f"Sick per step is: {sick_per_step} out of {truth.shape[0]}")
    logging.info(
        "Predicted sick per step is:"
        f" {list(map(int, np.sum(1-prediction[:, :intervals], axis=0)))} out of"
        f" {truth.shape[0]}",
    )
    logging.info(f"Survivors at each step is: {alive_per_step} out of {truth.shape[0]}")
    logging.info(
        f"Cumulative Censored: {cumulative_censored} or"
        f" {np.max(truth[:, :intervals]+truth[:, intervals:])}",
    )
    predicted_proportion = (
        np.sum(np.cumprod(prediction[:, :intervals], axis=1), axis=0) / truth.shape[0]
    )

    plt.plot(
        range(0, days_window, 1 + days_window // intervals),
        predicted_proportion,
        marker="o",
        label=f"Predicted Proportion C-Index:{c_index:0.3f}",
    )
    plt.plot(
        range(0, days_window, 1 + days_window // intervals),
        survivorship,
        marker="o",
        label="Survivorship",
    )
    plt.xlabel("Follow up time (days)")
    plt.ylabel("Proportion Surviving")
    plt.title(
        f"{title}\nEnrolled: {truth.shape[0]}, Censored: {cumulative_censored[-1]:.0f},"
        f" {100 * (cumulative_censored[-1] / truth.shape[0]):2.1f}%, Events:"
        f" {cumulative_sick[-1]:.0f},"
        f" {100 * (cumulative_sick[-1] / truth.shape[0]):2.1f}%\nMax follow up:"
        f" {days_window} days, {days_window // 365} years.",
    )
    plt.legend(loc="upper right")

    figure_path = os.path.join(prefix, "proportional_hazards_" + title + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    logging.info(f"Try to save survival plot at: {figure_path}")
    plt.savefig(figure_path)
    return {
        "c_index": c_index,
        "concordant": concordant,
        "discordant": discordant,
        "tied_risk": tied_risk,
        "tied_time": tied_time,
    }


def plot_survival_curves(
    prediction,
    truth,
    title,
    days_window,
    prefix="./figures/",
    num_curves=30,
    paths=None,
):
    intervals = truth.shape[-1] // 2
    plt.figure(figsize=(SUBPLOT_SIZE * 2, SUBPLOT_SIZE * 2))
    predicted_survivals = np.cumprod(prediction[:, :intervals], axis=1)
    sick = np.sum(truth[:, intervals:], axis=-1)
    censor_periods = np.argmin(truth[:, :intervals], axis=-1)
    x_days = range(0, days_window, 1 + days_window // intervals)
    cur_sick = 0
    cur_healthy = 0
    min_sick = num_curves * 0.1
    for i in range(truth.shape[0]):
        p = os.path.basename(paths[i]).replace(TENSOR_EXT, "")
        if sick[i] == 1:
            sick_period = np.argmax(truth[i, intervals:])
            sick_day = sick_period * (days_window // intervals)
            plt.plot(
                x_days[: sick_period + 2],
                predicted_survivals[i, : sick_period + 2],
                label=f"Failed:{p} p:{predicted_survivals[i, sick_period]:0.2f}",
                color="red",
            )
            plt.text(
                sick_day,
                predicted_survivals[i, sick_period],
                f"Diagnosed day:{sick_day} id:{p}",
            )
            cur_sick += 1
            if cur_sick >= min_sick and i >= num_curves:
                break
        elif censor_periods[i] != 0:  # individual was censored before failure
            plt.plot(
                x_days[: censor_periods[i]],
                predicted_survivals[i, : censor_periods[i]],
                label=(
                    f"Censored:{p} p:{predicted_survivals[i, censor_periods[i]]:0.2f}"
                ),
                color="blue",
            )
        elif cur_healthy < num_curves:
            plt.plot(
                x_days,
                predicted_survivals[i],
                label=f"Survived:{p} p:{predicted_survivals[i, -1]:0.2f}",
                color="green",
            )
            cur_healthy += 1
    plt.title(title + "\n")
    plt.legend(loc="lower left")
    plt.xlabel("Follow up time (days)")
    plt.ylabel("Survival Curve Prediction")
    figure_path = os.path.join(prefix, "survival_curves_" + title + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    logging.info("Try to save survival plot at: {}".format(figure_path))
    plt.savefig(figure_path)
    return {}


def _plot_ecg_text(
    data: Dict[str, Union[np.ndarray, str, Dict]], fig: plt.Figure, w: float, h: float,
) -> None:
    # top text
    dt = datetime.strptime(data["datetime"], ECG_DATETIME_FORMAT)
    dob = data["dob"]
    if dob != "":
        dob = datetime.strptime(dob, ECG_DATE_FORMAT)
        dob = f"{dob:%d-%b-%Y}".upper()
    age = -1
    if not np.isnan(data["age"]):
        age = int(data["age"])
    sex = {value: key for key, value in data["sex"].items()}

    fig.text(
        0.17 / w, 8.04 / h, f"{data['lastname']}, {data['firstname']}", weight="bold",
    )
    fig.text(3.05 / w, 8.04 / h, f"ID:{data['patientid']}", weight="bold")
    fig.text(4.56 / w, 8.04 / h, f"{dt:%d-%b-%Y %H:%M:%S}".upper(), weight="bold")
    fig.text(6.05 / w, 8.04 / h, f"{data['sitename']}", weight="bold")

    fig.text(0.17 / w, 7.77 / h, f"{dob} ({age} yr)", weight="bold")  # TODO age units
    fig.text(0.17 / w, 7.63 / h, f"{sex[1]}".title(), weight="bold")
    fig.text(0.17 / w, 7.35 / h, f"Room: ", weight="bold")  # TODO room?
    fig.text(0.17 / w, 7.21 / h, f"Loc: {data['location']}", weight="bold")

    fig.text(2.15 / w, 7.77 / h, f"Vent. rate", weight="bold")
    fig.text(2.15 / w, 7.63 / h, f"PR interval", weight="bold")
    fig.text(2.15 / w, 7.49 / h, f"QRS duration", weight="bold")
    fig.text(2.15 / w, 7.35 / h, f"QT/QTc", weight="bold")
    fig.text(2.15 / w, 7.21 / h, f"P-R-T axes", weight="bold")

    fig.text(3.91 / w, 7.77 / h, f"{int(data['rate_md'])}", weight="bold", ha="right")
    fig.text(3.91 / w, 7.63 / h, f"{int(data['pr_md'])}", weight="bold", ha="right")
    fig.text(3.91 / w, 7.49 / h, f"{int(data['qrs_md'])}", weight="bold", ha="right")
    fig.text(
        3.91 / w,
        7.35 / h,
        f"{int(data['qt_md'])}/{int(data['qtc_md'])}",
        weight="bold",
        ha="right",
    )
    fig.text(
        3.91 / w,
        7.21 / h,
        f"{int(data['paxis_md'])}   {int(data['raxis_md'])}",
        weight="bold",
        ha="right",
    )

    fig.text(4.30 / w, 7.77 / h, f"BPM", weight="bold", ha="right")
    fig.text(4.30 / w, 7.63 / h, f"ms", weight="bold", ha="right")
    fig.text(4.30 / w, 7.49 / h, f"ms", weight="bold", ha="right")
    fig.text(4.30 / w, 7.35 / h, f"ms", weight="bold", ha="right")
    fig.text(4.30 / w, 7.21 / h, f"{int(data['taxis_md'])}", weight="bold", ha="right")

    fig.text(4.75 / w, 7.21 / h, f"{data['read_md']}", wrap=True, weight="bold")

    # TODO tensorize these values from XML
    fig.text(1.28 / w, 6.65 / h, f"Technician: {''}", weight="bold")
    fig.text(1.28 / w, 6.51 / h, f"Test ind: {''}", weight="bold")
    fig.text(4.75 / w, 6.25 / h, f"Referred by: {''}", weight="bold")
    fig.text(7.63 / w, 6.25 / h, f"Electronically Signed By: {''}", weight="bold")


def _plot_ecg_full(voltage: Dict[str, np.ndarray], ax: plt.Axes) -> None:
    full_voltage = np.full((12, 2500), np.nan)
    for i, lead in enumerate(voltage):
        full_voltage[i] = voltage[lead]

    # convert voltage to millivolts
    full_voltage /= 1000

    # calculate space between leads
    min_y, max_y = ax.get_ylim()
    y_offset = (max_y - min_y) / len(voltage)

    text_xoffset = 5
    text_yoffset = -0.01

    # plot signal and add labels
    for i, lead in enumerate(voltage):
        this_offset = (len(voltage) - i - 0.5) * y_offset
        ax.plot(full_voltage[i] + this_offset, color="black", linewidth=0.375)
        ax.text(
            0 + text_xoffset,
            this_offset + text_yoffset,
            lead,
            ha="left",
            va="top",
            weight="bold",
        )


def _plot_ecg_clinical(voltage: Dict[str, np.ndarray], ax: plt.Axes) -> None:
    # get voltage in clinical chunks
    clinical_voltage = np.full((6, 2500), np.nan)
    halfgap = 5

    clinical_voltage[0][0 : 625 - halfgap] = voltage["I"][0 : 625 - halfgap]
    clinical_voltage[0][625 + halfgap : 1250 - halfgap] = voltage["aVR"][
        625 + halfgap : 1250 - halfgap
    ]
    clinical_voltage[0][1250 + halfgap : 1875 - halfgap] = voltage["V1"][
        1250 + halfgap : 1875 - halfgap
    ]
    clinical_voltage[0][1875 + halfgap : 2500] = voltage["V4"][1875 + halfgap : 2500]

    clinical_voltage[1][0 : 625 - halfgap] = voltage["II"][0 : 625 - halfgap]
    clinical_voltage[1][625 + halfgap : 1250 - halfgap] = voltage["aVL"][
        625 + halfgap : 1250 - halfgap
    ]
    clinical_voltage[1][1250 + halfgap : 1875 - halfgap] = voltage["V2"][
        1250 + halfgap : 1875 - halfgap
    ]
    clinical_voltage[1][1875 + halfgap : 2500] = voltage["V5"][1875 + halfgap : 2500]

    clinical_voltage[2][0 : 625 - halfgap] = voltage["III"][0 : 625 - halfgap]
    clinical_voltage[2][625 + halfgap : 1250 - halfgap] = voltage["aVF"][
        625 + halfgap : 1250 - halfgap
    ]
    clinical_voltage[2][1250 + halfgap : 1875 - halfgap] = voltage["V3"][
        1250 + halfgap : 1875 - halfgap
    ]
    clinical_voltage[2][1875 + halfgap : 2500] = voltage["V6"][1875 + halfgap : 2500]

    clinical_voltage[3] = voltage["V1"]
    clinical_voltage[4] = voltage["II"]
    clinical_voltage[5] = voltage["V5"]

    voltage = clinical_voltage

    # convert voltage to millivolts
    voltage /= 1000

    # calculate space between leads
    min_y, max_y = ax.get_ylim()
    y_offset = (max_y - min_y) / len(voltage)

    text_xoffset = 5
    text_yoffset = -0.1

    # plot signal and add labels
    for i in range(len(voltage)):
        this_offset = (len(voltage) - i - 0.5) * y_offset
        ax.plot(voltage[i] + this_offset, color="black", linewidth=0.375)
        if i == 0:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "I",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                625 + text_xoffset,
                this_offset + text_yoffset,
                "aVR",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1250 + text_xoffset,
                this_offset + text_yoffset,
                "V1",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1875 + text_xoffset,
                this_offset + text_yoffset,
                "V4",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 1:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "II",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                625 + text_xoffset,
                this_offset + text_yoffset,
                "aVL",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1250 + text_xoffset,
                this_offset + text_yoffset,
                "V2",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1875 + text_xoffset,
                this_offset + text_yoffset,
                "V5",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 2:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "III",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                625 + text_xoffset,
                this_offset + text_yoffset,
                "aVF",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1250 + text_xoffset,
                this_offset + text_yoffset,
                "V3",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1875 + text_xoffset,
                this_offset + text_yoffset,
                "V6",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 3:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "V1",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 4:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "II",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 5:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "V5",
                ha="left",
                va="top",
                weight="bold",
            )


def _plot_ecg_figure(
    data: Dict[str, Union[np.ndarray, str, Dict]],
    plot_signal_function: Callable[[Dict[str, np.ndarray], plt.Axes], None],
    plot_mode: str,
    output_folder: str,
    run_id: str,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9.5

    w, h = 11, 8.5
    fig = plt.figure(figsize=(w, h), dpi=100)

    # patient info and ecg text
    _plot_ecg_text(data, fig, w, h)

    # define plot area in inches
    left = 0.17
    bottom = h - 7.85
    width = w - 2 * left
    height = h - bottom - 2.3

    # ecg plot area
    ax = fig.add_axes([left / w, bottom / h, width / w, height / h])

    # voltage is in microvolts
    # the entire plot area is 5.55 inches tall, 10.66 inches wide (141 mm, 271 mm)
    # the resolution on the y-axis is 10 mm/mV
    # the resolution on the x-axis is 25 mm/s
    inch2mm = lambda inches: inches * 25.4

    # 1. set y-limit to max 14.1 mV
    y_res = 10  # mm/mV
    max_y = inch2mm(height) / y_res
    min_y = 0
    ax.set_ylim(min_y, max_y)

    # 2. set x-limit to max 10.8 s, center 10 s leads
    sampling_frequency = 250  # Hz
    x_res = 25  # mm/s
    max_x = inch2mm(width) / x_res
    x_buffer = (max_x - 10) / 2
    max_x -= x_buffer
    min_x = -x_buffer
    max_x *= sampling_frequency
    min_x *= sampling_frequency
    ax.set_xlim(min_x, max_x)

    # 3. set ticks for every 0.1 mV or every 1/25 s
    y_tick = 1 / y_res
    x_tick = 1 / x_res * sampling_frequency
    x_major_ticks = np.arange(min_x, max_x, x_tick * 5)
    x_minor_ticks = np.arange(min_x, max_x, x_tick)
    y_major_ticks = np.arange(min_y, max_y, y_tick * 5)
    y_minor_ticks = np.arange(min_y, max_y, y_tick)

    ax.set_xticks(x_major_ticks)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)

    ax.tick_params(
        which="both", left=False, bottom=False, labelleft=False, labelbottom=False,
    )
    ax.grid(b=True, color="r", which="major", lw=0.5)
    ax.grid(b=True, color="r", which="minor", lw=0.2)

    # signal plot
    voltage = data["2500_raw"]
    plot_signal_function(voltage, ax)

    # bottom text
    fig.text(
        0.17 / w,
        0.46 / h,
        f"{x_res}mm/s    {y_res}mm/mV    {sampling_frequency}Hz",
        ha="left",
        va="center",
        weight="bold",
    )

    # save both pdf and png
    title = re.sub(r"[:/. ]", "", f'{plot_mode}_{data["patientid"]}_{data["datetime"]}')
    plt.savefig(os.path.join(output_folder, run_id, f"{title}{PDF_EXT}"))
    plt.savefig(os.path.join(output_folder, run_id, f"{title}{IMAGE_EXT}"))
    plt.close(fig)


def plot_ecg(args):
    plot_tensors = [
        "ecg_patientid",
        "ecg_firstname",
        "ecg_lastname",
        "ecg_sex",
        "ecg_dob",
        "ecg_age",
        "ecg_datetime",
        "ecg_sitename",
        "ecg_location",
        "ecg_read_md",
        "ecg_taxis_md",
        "ecg_rate_md",
        "ecg_pr_md",
        "ecg_qrs_md",
        "ecg_qt_md",
        "ecg_paxis_md",
        "ecg_raxis_md",
        "ecg_qtc_md",
    ]
    voltage_tensor = "12_lead_ecg_2500_raw"
    # Imports: first party
    from ml4cvd.tensor_maps_ecg import TMAPS

    tensor_maps_in = [TMAPS[it] for it in plot_tensors + [voltage_tensor]]
    tensor_paths = [
        os.path.join(args.tensors, tp)
        for tp in os.listdir(args.tensors)
        if os.path.splitext(tp)[-1].lower() == TENSOR_EXT
    ]

    if "clinical" == args.plot_mode:
        plot_signal_function = _plot_ecg_clinical
    elif "full" == args.plot_mode:
        plot_signal_function = _plot_ecg_full
    else:
        raise ValueError(f"Unsupported plot mode: {args.plot_mode}")

    # TODO use TensorGenerator here
    # Get tensors for all hd5
    for tp in tensor_paths:
        try:
            with h5py.File(tp, "r") as hd5:
                skip_hd5 = False
                tdict = defaultdict(dict)
                for tm in tensor_maps_in:
                    key = tm.name.split(
                        "12_lead_ecg_" if key == voltage_tensor else "ecg_",
                    )[1]
                    try:
                        tensors = tm.tensor_from_file(tm, hd5)

                        if tm.shape[0] is not None:
                            # If not a multi-tensor tensor, wrap in array to loop through
                            tensors = np.array([tensors])
                        for i, tensor in enumerate(tensors):
                            if tm.channel_map:
                                tdict[i][key] = dict()
                                for cm in tm.channel_map:
                                    tdict[i][key][cm] = (
                                        tensor[:, tm.channel_map[cm]]
                                        if tm.name == voltage_tensor
                                        else tensor[tm.channel_map[cm]]
                                    )
                            else:
                                if 1 == (
                                    tm.shape[0]
                                    if tm.shape[0] is not None
                                    else tm.shape[1]
                                ):
                                    tensor = tensor.item()
                                tdict[i][key] = tensor
                    except (
                        IndexError,
                        KeyError,
                        ValueError,
                        OSError,
                        RuntimeError,
                    ) as e:
                        logging.warning(
                            f"Could not obtain {tm.name}. Skipping plotting for all"
                            f" ECGs at {tp}",
                        )
                        skip_hd5 = True
                    if skip_hd5:
                        break
                if skip_hd5:
                    continue

                # plot each ecg
                for i in tdict:
                    _plot_ecg_figure(
                        data=tdict[i],
                        plot_signal_function=plot_signal_function,
                        plot_mode=args.plot_mode,
                        output_folder=args.output_folder,
                        run_id=args.id,
                    )
        except:
            logging.exception(f"Broken tensor at: {tp}")


def plot_cross_reference(
    args, xref_df, title, time_description, window_start, window_end,
):
    # TODO make this work with multiple time windows
    if xref_df.empty:
        logging.info(f'No cross reference found for "{title}"')
        return

    title = title.replace(" ", "_")

    # compute day diffs
    day_diffs = np.array(
        xref_df.apply(
            lambda row: (row[args.time_tensor] - row[window_end]).days, axis=1,
        ),
    )

    plt.rcParams["font.size"] = 18
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    binwidth = 5
    ax.hist(
        day_diffs, bins=range(day_diffs.min(), day_diffs.max() + binwidth, binwidth),
    )
    ax.set_xlabel("Days relative to event")
    ax.set_ylabel("Number of patients")
    ax.set_title(
        f"Distribution of {args.tensors_name} {time_description}: N={len(day_diffs)}",
    )

    ax.text(0.05, 0.90, f"Min: {day_diffs.min()}", transform=ax.transAxes)
    ax.text(0.05, 0.85, f"Max: {day_diffs.max()}", transform=ax.transAxes)
    ax.text(0.05, 0.80, f"Median: {np.median(day_diffs):.0f}", transform=ax.transAxes)
    plt.tight_layout()

    fpath = os.path.join(
        args.output_folder, args.id, f"distribution_{title}{IMAGE_EXT}",
    )
    fig.savefig(fpath)
    logging.info(f"Saved histogram of days relative to {window_end} to {fpath}")


def plot_roc_per_class(
    prediction, truth, labels, title, prefix="./figures/", data_split="test",
):
    plt.rcParams["font.size"] = 14
    lw = 2
    labels_to_areas = {}
    true_sums = np.sum(truth, axis=0)
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))
    fpr, tpr, roc_auc = get_fpr_tpr_roc_pred(prediction, truth, labels)

    if len(labels) == 2:
        negative_label_idx = _find_negative_label_index(labels=labels, key_prefix="no_")

    for idx, label in enumerate(labels):
        if len(labels) == 2 and idx == negative_label_idx:
            continue

        labels_to_areas[label] = roc_auc[labels[label]]
        color = _hash_string_to_color(label)
        label_text = (
            f"{label} = {roc_auc[labels[label]]:.3f}, n={true_sums[labels[label]]:.0f}"
        )
        plt.plot(
            fpr[labels[label]],
            tpr[labels[label]],
            color=color,
            lw=lw,
            label=label_text,
        )
        logging.info(
            f"ROC AUC for {label_text}, "
            f"Truth shape {truth.shape}, "
            f"True sums {true_sums}",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([-0.02, 1.03])
    plt.ylabel(RECALL_LABEL)
    plt.xlabel(FALLOUT_LABEL)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "k:", lw=0.5)
    plt.title(f"ROC curve: {title}, n={truth.shape[0]:.0f}\n")

    figure_path = os.path.join(
        prefix, "per_class_roc_" + title + "_" + data_split + IMAGE_EXT,
    )
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path, bbox_inches="tight")
    plt.clf()
    logging.info("Saved ROC curve at: {}".format(figure_path))
    return labels_to_areas


def plot_rocs(predictions, truth, labels, title, prefix="./figures/"):
    plt.rcParams["font.size"] = 14
    lw = 2
    true_sums = np.sum(truth, axis=0)
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))

    for p in predictions:
        fpr, tpr, roc_auc = get_fpr_tpr_roc_pred(predictions[p], truth, labels)
        if len(labels) == 2:
            negative_label_idx = _find_negative_label_index(
                labels=labels, key_prefix="no_",
            )
        for idx, label in enumerate(labels):
            if len(labels) == 2 and idx == negative_label_idx:
                continue

            color = _hash_string_to_color(p + label)
            label_text = (
                f"{p}_{label} area:{roc_auc[labels[label]]:.3f}"
                f" n={true_sums[labels[label]]:.0f}"
            )
            plt.plot(
                fpr[labels[label]],
                tpr[labels[label]],
                color=color,
                lw=lw,
                label=label_text,
            )
            logging.info(f"ROC label {label_text}")

    plt.xlim([0.0, 1.0])
    plt.ylim([-0.02, 1.03])
    plt.ylabel(RECALL_LABEL)
    plt.xlabel(FALLOUT_LABEL)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "k:", lw=0.5)
    plt.title(f"ROC curve: {title}, n={np.sum(true_sums):.0f}\n")

    figure_path = os.path.join(prefix, "per_class_roc_" + title + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    plt.clf()
    logging.info("Saved ROC curve at: {}".format(figure_path))


def subplot_rocs(
    rocs: List[Tuple[np.ndarray, np.ndarray, Dict[str, int]]],
    prefix: str = "./figures/",
):
    """Log and tabulate AUCs given as nested dictionaries in the format '{model: {label: auc}}'"""
    lw = 2
    row = 0
    col = 0
    total_plots = len(rocs)
    cols = max(2, int(math.ceil(math.sqrt(total_plots))))
    rows = max(2, int(math.ceil(total_plots / cols)))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * SUBPLOT_SIZE, rows * SUBPLOT_SIZE),
    )
    for predicted, truth, labels in rocs:
        true_sums = np.sum(truth, axis=0)
        fpr, tpr, roc_auc = get_fpr_tpr_roc_pred(predicted, truth, labels)

        if len(labels) == 2:
            negative_label_idx = _find_negative_label_index(
                labels=labels, key_prefix="no_",
            )

        for idx, label in enumerate(labels):
            if len(labels) == 2 and idx == negative_label_idx:
                continue

            color = _hash_string_to_color(label)
            label_text = (
                f"{label} area: {roc_auc[labels[label]]:.3f}"
                f" n={true_sums[labels[label]]:.0f}"
            )
            axes[row, col].plot(
                fpr[labels[label]],
                tpr[labels[label]],
                color=color,
                lw=lw,
                label=label_text,
            )
            logging.info(f"ROC Label {label_text}")
        axes[row, col].set_xlim([0.0, 1.0])
        axes[row, col].set_ylim([-0.02, 1.03])
        axes[row, col].set_ylabel(RECALL_LABEL)
        axes[row, col].set_xlabel(FALLOUT_LABEL)
        axes[row, col].legend(loc="lower right")
        axes[row, col].plot([0, 1], [0, 1], "k:", lw=0.5)
        axes[row, col].set_title(f"ROC n={np.sum(true_sums):.0f}")

        row += 1
        if row == rows:
            row = 0
            col += 1
            if col >= cols:
                break

    figure_path = prefix + "rocs_together" + IMAGE_EXT
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def subplot_comparison_rocs(
    rocs: List[Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int]]],
    prefix: str = "./figures/",
):
    """Log and tabulate AUCs given as nested dictionaries in the format '{model: {label: auc}}'"""
    lw = 3
    row = 0
    col = 0
    total_plots = len(rocs)
    cols = max(2, int(math.ceil(math.sqrt(total_plots))))
    rows = max(2, int(math.ceil(total_plots / cols)))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * SUBPLOT_SIZE, rows * SUBPLOT_SIZE),
    )
    for predictions, truth, labels in rocs:
        true_sums = np.sum(truth, axis=0)
        for p in predictions:
            fpr, tpr, roc_auc = get_fpr_tpr_roc_pred(predictions[p], truth, labels)
            if len(labels) == 2:
                negative_label_idx = _find_negative_label_index(
                    labels=labels, key_prefix="no_",
                )
            for idx, label in enumerate(labels):
                if len(labels) == 2 and idx == negative_label_idx:
                    continue

                color = _hash_string_to_color(p + label)
                label_text = (
                    f"{p}_{label} area:{roc_auc[labels[label]]:.3f}"
                    f" n={true_sums[labels[label]]:.0f}"
                )
                axes[row, col].plot(
                    fpr[labels[label]],
                    tpr[labels[label]],
                    color=color,
                    lw=lw,
                    label=label_text,
                )
                logging.info(f"ROC Label {label_text}")

        axes[row, col].set_xlim([0.0, 1.0])
        axes[row, col].set_ylim([-0.02, 1.03])
        axes[row, col].set_ylabel(RECALL_LABEL)
        axes[row, col].set_xlabel(FALLOUT_LABEL)
        axes[row, col].legend(loc="lower right")
        axes[row, col].plot([0, 1], [0, 1], "k:", lw=0.5)
        axes[row, col].set_title(f"ROC n={np.sum(true_sums):.0f}\n")

        row += 1
        if row == rows:
            row = 0
            col += 1
            if col >= cols:
                break

    figure_path = os.path.join(prefix, "rocs_compared_together" + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)


def plot_precision_recall_per_class(
    prediction, truth, labels, title, prefix="./figures/", data_split="test",
):
    plt.rcParams["font.size"] = 14
    lw = 2.0
    labels_to_areas = {}
    true_sums = np.sum(truth, axis=0)
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))

    if len(labels) == 2:
        negative_label_idx = _find_negative_label_index(labels=labels, key_prefix="no_")
    for idx, label in enumerate(labels):
        if len(labels) == 2 and idx == negative_label_idx:
            continue

        precision, recall, _ = precision_recall_curve(
            truth[:, labels[label]], prediction[:, labels[label]],
        )
        average_precision = average_precision_score(
            truth[:, labels[label]], prediction[:, labels[label]],
        )
        labels_to_areas[label] = average_precision
        color = _hash_string_to_color(label)
        label_text = (
            f"{label} mean precision: {average_precision:.3f},"
            f" n={true_sums[labels[label]]:.0f}"
        )
        plt.plot(recall, precision, lw=lw, color=color, label=label_text)
        logging.info(f"prAUC Label {label_text}")

    plt.xlim([0.0, 1.0])
    plt.ylim([-0.02, 1.03])
    plt.xlabel(RECALL_LABEL)
    plt.ylabel(PRECISION_LABEL)
    plt.legend(loc="lower right")
    plt.title(f"PR curve: {title}, n={np.sum(true_sums):.0f}\n")

    figure_path = os.path.join(
        prefix, "precision_recall_" + title + "_" + data_split + IMAGE_EXT,
    )
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path, bbox_inches="tight")
    plt.clf()
    logging.info(f"Saved Precision Recall curve at: {figure_path}")
    return labels_to_areas


def plot_precision_recalls(predictions, truth, labels, title, prefix="./figures/"):
    # Compute Precision-Recall and plot curve for each model
    lw = 2.0
    true_sums = np.sum(truth, axis=0)
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))

    for p in predictions:
        if len(labels) == 2:
            negative_label_idx = _find_negative_label_index(
                labels=labels, key_prefix="no_",
            )
        for idx, label in enumerate(labels):
            if len(labels) == 2 and idx == negative_label_idx:
                continue

            c = _hash_string_to_color(p + label)
            precision, recall, _ = precision_recall_curve(
                truth[:, labels[label]], predictions[p][:, labels[label]],
            )
            average_precision = average_precision_score(
                truth[:, labels[label]], predictions[p][:, labels[label]],
            )
            label_text = (
                f"{p}_{label} mean precision:{average_precision:.3f}"
                f" n={true_sums[labels[label]]:.0f}"
            )
            plt.plot(recall, precision, lw=lw, color=c, label=label_text)
            logging.info(f"prAUC Label {label_text}")

    plt.xlim([0.0, 1.00])
    plt.ylim([-0.02, 1.03])
    plt.xlabel(RECALL_LABEL)
    plt.ylabel(PRECISION_LABEL)
    plt.legend(loc="lower left")
    plt.title(f"{title} n={np.sum(true_sums):.0f}")

    figure_path = os.path.join(prefix, "precision_recall_" + title + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    plt.clf()
    logging.info("Saved Precision Recall curve at: {}".format(figure_path))


def get_fpr_tpr_roc_pred(y_pred, test_truth, labels):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for k in labels:
        cur_idx = labels[k]
        aser = roc_curve(test_truth[:, cur_idx], y_pred[:, cur_idx])
        fpr[labels[k]], tpr[labels[k]], _ = aser
        roc_auc[labels[k]] = auc(fpr[labels[k]], tpr[labels[k]])

    return fpr, tpr, roc_auc


def plot_waves(predicted_waves, true_waves, title, plot_path, rows=6, cols=6):
    row = 0
    col = 0
    f, axes = plt.subplots(rows, cols, sharex=True, figsize=(36, 36))
    for i in range(true_waves.shape[0]):
        axes[row, col].plot(true_waves[i, :, 0], color="blue", label="Actual Wave")
        if predicted_waves is not None:
            axes[row, col].plot(
                predicted_waves[i, :, 0], color="green", label="Predicted",
            )
        axes[row, col].set_xlabel("time")
        row += 1
        if row == rows:
            row = 0
            col += 1
            if col >= cols:
                break
    plt.legend(loc="lower left")
    figure_path = os.path.join(plot_path, title + IMAGE_EXT)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    plt.clf()
    logging.info("Saved waves at: {}".format(figure_path))


def plot_tsne(
    x_embed,
    categorical_labels,
    continuous_labels,
    gene_labels,
    label_dict,
    figure_path,
    alpha,
):
    x_embed = np.array(x_embed)
    if len(x_embed.shape) > 2:
        x_embed = np.reshape(x_embed, (x_embed.shape[0], np.prod(x_embed.shape[1:])))

    n_components = 2
    rows = max(2, len(label_dict))
    perplexities = [25, 75]
    (fig, subplots) = plt.subplots(
        rows,
        len(perplexities),
        figsize=(len(perplexities) * SUBPLOT_SIZE * 2, rows * SUBPLOT_SIZE * 2),
    )

    p2y = {}
    for i, p in enumerate(perplexities):
        tsne = manifold.TSNE(
            n_components=n_components,
            init="pca",
            random_state=123,
            perplexity=p,
            learning_rate=20,
            n_iter_without_progress=500,
        )
        p2y[p] = tsne.fit_transform(x_embed)

    j = -1
    for tm in label_dict:
        j += 1
        if j == rows:
            break
        categorical_subsets = {}
        categorical_counts = Counter()
        if tm in categorical_labels + gene_labels:
            for c in tm.channel_map:
                categorical_subsets[tm.channel_map[c]] = (
                    label_dict[tm] == tm.channel_map[c]
                )
                categorical_counts[tm.channel_map[c]] = np.sum(
                    categorical_subsets[tm.channel_map[c]],
                )
        elif tm in continuous_labels:
            colors = label_dict[tm]
        for i, p in enumerate(perplexities):
            ax = subplots[j, i]
            ax.set_title(f"{tm.name} | t-SNE perplexity:{p}")
            if tm in categorical_labels + gene_labels:
                color_labels = []
                for c in tm.channel_map:
                    channel_index = tm.channel_map[c]
                    color = _hash_string_to_color(c)
                    color_labels.append(
                        f"{c} n={categorical_counts[tm.channel_map[c]]}",
                    )
                    ax.scatter(
                        p2y[p][categorical_subsets[channel_index], 0],
                        p2y[p][categorical_subsets[channel_index], 1],
                        c=color,
                        alpha=alpha,
                    )
                ax.legend(color_labels, loc="lower left")
            elif tm in continuous_labels:
                points = ax.scatter(
                    p2y[p][:, 0], p2y[p][:, 1], c=colors, alpha=alpha, cmap="jet",
                )
                if i == len(perplexities) - 1:
                    fig.colorbar(points, ax=ax)

            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis("tight")

    figure_path += "tsne_plot" + IMAGE_EXT
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    plt.clf()
    logging.info(f"Saved T-SNE plot at: {figure_path}")


def plot_find_learning_rate(
    learning_rates: List[float],
    losses: List[float],
    smoothed_losses: List[float],
    picked_learning_rate: Optional[float],
    figure_path: str,
):
    plt.figure(figsize=(2 * SUBPLOT_SIZE, SUBPLOT_SIZE))
    plt.title("Learning rate finder")
    cutoff = smoothed_losses[0]
    plt.ylim(min(smoothed_losses), cutoff * 1.05)
    plt.axhline(
        cutoff, linestyle="--", color="k", label=f"Deltas ignored above {cutoff:.2f}",
    )
    learning_rates = np.log(learning_rates) / np.log(10)
    plt.plot(learning_rates, losses, label="Loss", c="r")
    plt.plot(learning_rates, smoothed_losses, label="Smoothed loss", c="b")
    if picked_learning_rate is not None:
        plt.axvline(
            np.log(picked_learning_rate) / np.log(10),
            label=f"Learning rate found {picked_learning_rate:.2E}",
            color="g",
            linestyle="--",
        )
    plt.xlabel("Log_10 learning rate")
    plt.legend()
    plt.savefig(os.path.join(figure_path, f"find_learning_rate{IMAGE_EXT}"))
    plt.clf()


def plot_saliency_maps(
    data: np.ndarray, gradients: np.ndarray, paths: List, prefix: str,
):
    """Plot saliency maps of a batch of input tensors.

    Saliency maps for each input tensor in the batch will be saved at the file path indicated by prefix.
    Also creates a mean saliency map across the batch
    2D tensors are assumed to be ECGs and 3D tensors are plotted with each slice as an RGB image.
    The red channel indicates negative gradients, and the green channel positive ones.

    :param data: A batch of input tensors
    :param gradients: A corresponding batch of gradients for those inputs, must be the same shape as data
    :param paths: A List of paths corresponding to each input tensor
    :param prefix: file path prefix where saliency maps will be saved
    """
    if data.shape[-1] == 1:
        data = data[..., 0]
        gradients = gradients[..., 0]

    mean_saliency = np.zeros(data.shape[1:4] + (3,))
    for batch_i, path in enumerate(paths):
        sample_id = os.path.basename(path).replace(TENSOR_EXT, "")
        if len(data.shape) == 3:
            ecgs = {f"{sample_id}_raw": data[batch_i], "gradients": gradients[batch_i]}
            _plot_ecgs(ecgs, f"{prefix}_{sample_id}_saliency_{batch_i}{IMAGE_EXT}")
        elif len(data.shape) == 4:
            cols = max(2, int(math.ceil(math.sqrt(data.shape[-1]))))
            rows = max(2, int(math.ceil(data.shape[-1] / cols)))
            title = f"{prefix}_{sample_id}_saliency_{batch_i}{IMAGE_EXT}"
            _plot_3d_tensor_slices_as_rgb(
                _saliency_map_rgb(data[batch_i], gradients[batch_i]), title, cols, rows,
            )
            saliency = _saliency_blurred_and_scaled(
                gradients[batch_i], blur_radius=5.0, max_value=1.0 / data.shape[0],
            )
            mean_saliency[..., 0] -= saliency
            mean_saliency[..., 1] += saliency
        elif len(data.shape) == 5:
            for j in range(data.shape[-1]):
                cols = max(2, int(math.ceil(math.sqrt(data.shape[-2]))))
                rows = max(2, int(math.ceil(data.shape[-2] / cols)))
                name = f"{prefix}_saliency_{batch_i}_channel_{j}{IMAGE_EXT}"
                _plot_3d_tensor_slices_as_rgb(
                    _saliency_map_rgb(
                        data[batch_i, ..., j], gradients[batch_i, ..., j],
                    ),
                    name,
                    cols,
                    rows,
                )
                saliency = _saliency_blurred_and_scaled(
                    gradients[batch_i, ..., j],
                    blur_radius=5.0,
                    max_value=1.0 / data.shape[0],
                )
                mean_saliency[..., 0] -= saliency
                mean_saliency[..., 1] += saliency
        else:
            logging.warning(f"No method to plot saliency for data shape: {data.shape}")

    if len(data.shape) == 4:
        _plot_3d_tensor_slices_as_rgb(
            _scale_tensor_inplace(mean_saliency),
            f"{prefix}_batch_mean_saliency{IMAGE_EXT}",
            cols,
            rows,
        )
    logging.info(f"Saved saliency maps at:{prefix}")


def _scale_tensor_inplace(tensor, min_value=0.0, max_value=1.0):
    tensor -= tensor.min()
    tensor *= (max_value - min_value) / tensor.max()
    tensor += min_value
    return tensor


def _saliency_blurred_and_scaled(gradients, blur_radius, max_value=1.0):
    blurred = gaussian_filter(gradients, sigma=blur_radius)
    _scale_tensor_inplace(blurred, max_value=max_value)
    blurred -= blurred.mean()
    return blurred


def _saliency_map_rgb(image, gradients, blur_radius=0):
    _scale_tensor_inplace(image)
    rgb_map = np.zeros(image.shape + (3,))
    blurred = _saliency_blurred_and_scaled(gradients, blur_radius)
    rgb_map[..., 0] = image - blurred
    rgb_map[..., 1] = image + blurred
    rgb_map[..., 2] = image
    rgb_map = np.clip(rgb_map, 0, 1)
    # _scale_tensor_inplace(rgb_map)
    return rgb_map


def _plot_ecgs(
    ecgs,
    figure_path,
    rows=3,
    cols=4,
    time_interval=2.5,
    raw_scale=0.005,
    hertz=500,
    lead_dictionary=ECG_REST_LEADS,
):
    index2leads = {v: k for k, v in lead_dictionary.items()}
    _, axes = plt.subplots(rows, cols, figsize=(18, 16), sharey=True)
    for i in range(rows):
        for j in range(cols):
            start = int(i * time_interval * hertz)
            stop = int((i + 1) * time_interval * hertz)
            axes[i, j].set_xlim(start, stop)
            for label in ecgs:
                axes[i, j].plot(
                    range(start, stop),
                    ecgs[label][start:stop, j + i * cols] * raw_scale,
                    label=label,
                )
            axes[i, j].legend(loc="lower right")
            axes[i, j].set_xlabel("milliseconds")
            axes[i, j].set_ylabel("mV")
            axes[i, j].set_title(index2leads[j + i * cols])
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    plt.clf()


def _plot_3d_tensor_slices_as_rgb(tensor, figure_path, cols=3, rows=10):
    _, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for i in range(tensor.shape[-2]):
        axes[i // cols, i % cols].imshow(tensor[:, :, i, :])
        axes[i // cols, i % cols].set_yticklabels([])
        axes[i // cols, i % cols].set_xticklabels([])

    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    plt.clf()


def _hash_string_to_color(string):
    """Hash a string to color (using hashlib and not the built-in hash for consistency between runs)"""
    return COLOR_ARRAY[
        int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % len(COLOR_ARRAY)
    ]


def _text_on_plot(axes, x, y, text, alpha=0.8, background="white"):
    t = axes.text(x, y, text)
    t.set_bbox({"facecolor": background, "alpha": alpha, "edgecolor": background})
