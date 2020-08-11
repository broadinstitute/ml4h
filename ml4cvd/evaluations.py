# Imports: standard library
import os
import logging
from typing import Dict, List, Tuple, Union, Optional

# Imports: third party
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model

# Imports: first party
from ml4cvd.plots import plot_tsne, subplot_rocs, subplot_scatters, evaluate_predictions
from ml4cvd.models import embed_model_predict
from ml4cvd.TensorMap import TensorMap
from ml4cvd.definitions import Path, Paths, Inputs, Outputs, Predictions
from ml4cvd.tensor_generators import (
    BATCH_INPUT_INDEX,
    BATCH_PATHS_INDEX,
    BATCH_OUTPUT_INDEX,
    TensorGenerator,
)


def predict_and_evaluate(
    model: Model,
    data: Union[TensorGenerator, Tuple[Inputs, Outputs], Tuple[Inputs, Outputs, Paths]],
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    plot_path: Path,
    data_split: str,
    save_coefficients: bool = False,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    hidden_layer: Optional[str] = None,
    embed_visualization: Optional[str] = None,
    alpha: Optional[float] = None,
) -> Dict:
    """
    Evaluate model on dataset, save plots, and return performance metrics

    :param model: Model
    :param data: TensorGenerator or tuple of inputs, outputs, and optionally paths
    :param tensor_maps_in: Input maps
    :param tensor_maps_out: Output maps
    :param plot_path: Path to directory to save plots to
    :param data_split: Name of data split
    :param save_coefficients: Save model coefficients
    :param steps: Number of batches to use, required if data is a TensorGenerator
    :param batch_size: Number of samples to use in a batch, required if data is a tuple input and output numpy arrays
    :param hidden_layer: Name of hidden layer for embedded visualization
    :param embed_visualization: Type of embedded visualization
    :param alpha: Float of transparency for embedded visualization
    :return: Dictionary of performance metrics
    """
    layer_names = [layer.name for layer in model.layers]
    performance_metrics = {}
    scatters = []
    rocs = []

    if save_coefficients:
        # Get coefficients from model layers
        coefficients = [c[0].round(3) for c in model.layers[-1].get_weights()[0]]

        # Get feature names from TMaps
        feature_names = []
        for tm in tensor_maps_in:
            # Use append to add single string to list
            if tm.channel_map is None:
                feature_names.append(tm.name)
            # Use extend to add list items to list
            else:
                feature_names.extend(tm.channel_map)

        if len(coefficients) != len(feature_names):
            raise ValueError("Number of coefficient values and names differ!")

        # Create dataframe of features
        df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        df = df.iloc[(-df["coefficient"]).argsort()].reset_index(drop=True)

        # Save dataframe
        fname = os.path.join(plot_path, "coefficients" + ".csv")
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        df.round(3).to_csv(path_or_buf=fname, index=False)

    y_predictions, output_data, data_paths = _get_predictions_from_data(
        model=model, data=data, steps=steps, batch_size=batch_size,
    )

    for y, tm in zip(y_predictions, tensor_maps_out):
        if tm.output_name() not in layer_names:
            continue
        y_truth = np.array(output_data[tm.output_name()])
        performance_metrics.update(
            evaluate_predictions(
                tm=tm,
                y_predictions=y,
                y_truth=y_truth,
                title=tm.name,
                folder=plot_path,
                test_paths=data_paths,
                rocs=rocs,
                scatters=scatters,
                data_split=data_split,
            ),
        )

    if len(rocs) > 1:
        subplot_rocs(rocs, data_split, plot_path)
    if len(scatters) > 1:
        subplot_scatters(scatters, data_split, plot_path)

    if embed_visualization == "tsne":
        if isinstance(data, TensorGenerator):
            raise NotImplementedError(
                f"Embedded visualization is not currently supported with generators.",
            )
        output_data_1d = {
            tm: np.array(output_data[tm.output_name()])
            for tm in tensor_maps_out
            if tm.output_name() in output_data
        }
        _tsne_wrapper(
            model=model,
            hidden_layer=hidden_layer,
            alpha=alpha,
            plot_path=plot_path,
            test_paths=data_paths,
            test_labels=output_data_1d,
            test_data=data[BATCH_INPUT_INDEX],
            tensor_maps_in=tensor_maps_in,
            batch_size=batch_size,
        )

    return performance_metrics


def _get_predictions_from_data(
    model: Model,
    data: Union[TensorGenerator, Tuple[Inputs, Outputs], Tuple[Inputs, Outputs, Paths]],
    steps: Optional[int],
    batch_size: Optional[int],
) -> Tuple[Predictions, Outputs, Optional[Paths]]:
    """
    Get model predictions, output data, and paths from data source

    :param model: Model
    :param data: TensorGenerator or tuple of inputs, outputs, and optionally paths
    :param steps: Number of batches to use, required if data is a TensorGenerator
    :param batch_size: Number of samples to use in a batch, required if data is a tuple input and output numpy arrays
    :return: Tuple of predictions as a list of numpy arrays, a dictionary of output data, and optionally paths
    """

    if isinstance(data, Tuple):
        if len(data) == 2:
            input_data, output_data = data
            paths = None
        elif len(data) == 3:
            input_data, output_data, paths = data
        else:
            raise ValueError(f"Expected 2 or 3 elements to data tuple, got {len(data)}")

        if batch_size is None:
            raise ValueError(
                f"When providing data as tuple of inputs and outputs, batch_size is required, got {batch_size}",
            )

        y_predictions = model.predict(input_data, batch_size=batch_size)
        if not isinstance(y_predictions, list):
            y_predictions = [y_predictions]
    elif isinstance(data, TensorGenerator):
        if steps is None:
            raise ValueError(
                f"When providing data as a generator, steps is required, got {steps}",
            )

        data.reset(deterministic=True)
        batch_size = data.batch_size
        data_length = steps * batch_size
        y_predictions = [
            np.zeros((data_length,) + tm.static_shape) for tm in data.output_maps
        ]
        output_data = {
            tm.output_name(): np.zeros((data_length,) + tm.static_shape)
            for tm in data.output_maps
        }
        paths = [] if data.keep_paths else None
        for step in range(steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            batch = next(data)

            # for single output models, prediction is an ndarray
            # for multi output models, predictions are a list of ndarrays
            batch_y_predictions = model.predict(batch[BATCH_INPUT_INDEX])
            if not isinstance(batch_y_predictions, list):
                batch_y_predictions = [batch_y_predictions]
            for i in range(len(y_predictions)):
                y_predictions[i][start_idx:end_idx] = batch_y_predictions[i]

            for output_name, output_tensor in batch[BATCH_OUTPUT_INDEX].items():
                output_data[output_name][start_idx:end_idx] = output_tensor

            if data.keep_paths:
                paths.extend(batch[BATCH_PATHS_INDEX])
    else:
        raise NotImplementedError(
            f"Cannot get data for inference from data of type {type(data).__name__}: {data}",
        )

    # predictions returned by this function are lists of numpy arrays
    return y_predictions, output_data, paths


def _test_labels_to_label_map(
    test_labels: Dict[TensorMap, np.ndarray], examples: int,
) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    label_dict = {tm: np.zeros((examples,)) for tm in test_labels}
    categorical_labels = []
    continuous_labels = []

    for tm in test_labels:
        for i in range(examples):
            if tm.is_continuous() and tm.static_axes() == 1:
                label_dict[tm][i] = tm.rescale(test_labels[tm][i])
                continuous_labels.append(tm)
            elif tm.is_categorical() and tm.static_axes() == 1:
                label_dict[tm][i] = np.argmax(test_labels[tm][i])
                categorical_labels.append(tm)

    return label_dict, categorical_labels, continuous_labels


def _tsne_wrapper(
    model: Model,
    hidden_layer: str,
    alpha: float,
    plot_path: Path,
    test_paths: Paths,
    test_labels: Dict[TensorMap, np.ndarray],
    test_data: Optional[Inputs] = None,
    tensor_maps_in: Optional[List[TensorMap]] = None,
    batch_size: Optional[int] = None,
    embeddings: Optional[np.ndarray] = None,
):
    """Plot 2D t-SNE of a model's hidden layer colored by many different co-variates.

    Callers must provide either model's embeddings or test_data on which embeddings will be inferred

    :param model: Keras model
    :param hidden_layer: String name of the hidden layer whose embeddings will be visualized
    :param alpha: Transparency of each data point
    :param plot_path: Image file name and path for the t_SNE plot
    :param test_paths: Paths for hd5 file containing each sample
    :param test_labels: Dictionary mapping TensorMaps to numpy arrays of labels (co-variates) to color code the t-SNE plots with
    :param test_data: Input data for the model necessary if embeddings is None
    :param tensor_maps_in: Input TensorMaps of the model necessary if embeddings is None
    :param batch_size: Batch size necessary if embeddings is None
    :param embeddings: (optional) Model's embeddings
    :return: None
    """
    if hidden_layer_name not in [layer.name for layer in model.layers]:
        logging.warning(
            f"Can't compute t-SNE, layer:{hidden_layer_name} not in provided model.",
        )
        return

    if embeddings is None:
        embeddings = embed_model_predict(
            model, tensor_maps_in, hidden_layer_name, test_data, batch_size,
        )

    gene_labels = []
    label_dict, categorical_labels, continuous_labels = _test_labels_to_label_map(
        test_labels, len(test_paths),
    )
    if (
        len(categorical_labels) > 0
        or len(continuous_labels) > 0
        or len(gene_labels) > 0
    ):
        plot_tsne(
            embeddings,
            categorical_labels,
            continuous_labels,
            gene_labels,
            label_dict,
            plot_path,
            alpha,
        )
