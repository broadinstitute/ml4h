# Imports: standard library
import os
import logging
from typing import Dict, List, Tuple

# Imports: third party
import numpy as np
import pandas as pd

# Imports: first party
from ml4cvd.plots import plot_tsne, subplot_rocs, subplot_scatters, evaluate_predictions
from ml4cvd.models import embed_model_predict
from ml4cvd.TensorMap import TensorMap
from ml4cvd.tensor_generators import (
    BATCH_INPUT_INDEX,
    BATCH_PATHS_INDEX,
    BATCH_OUTPUT_INDEX,
)


def predict_and_evaluate(
    model,
    test_data,
    test_labels,
    tensor_maps_in,
    tensor_maps_out,
    batch_size,
    hidden_layer,
    plot_path,
    test_paths,
    embed_visualization,
    alpha,
    data_split="test",
    save_coefficients=False,
):
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

    y_predictions = model.predict(test_data, batch_size=batch_size)
    for y, tm in zip(y_predictions, tensor_maps_out):

        # When models have one output, model.predict returns ndarray
        # otherwise it returns a list
        if tm.output_name() not in layer_names:
            continue
        if not isinstance(y_predictions, list):
            y = y_predictions
        y_truth = np.array(test_labels[tm.output_name()])
        performance_metrics.update(
            evaluate_predictions(
                tm=tm,
                y_predictions=y,
                y_truth=y_truth,
                title=tm.name,
                folder=plot_path,
                test_paths=test_paths,
                rocs=rocs,
                scatters=scatters,
                data_split=data_split,
            ),
        )

    if len(rocs) > 1:
        subplot_rocs(rocs, plot_path)
    if len(scatters) > 1:
        subplot_scatters(scatters, plot_path)

    test_labels_1d = {
        tm: np.array(test_labels[tm.output_name()])
        for tm in tensor_maps_out
        if tm.output_name() in test_labels
    }
    if embed_visualization == "tsne":
        _tsne_wrapper(
            model,
            hidden_layer,
            alpha,
            plot_path,
            test_paths,
            test_labels_1d,
            test_data=test_data,
            tensor_maps_in=tensor_maps_in,
            batch_size=batch_size,
        )

    return performance_metrics


def predict_scalars_and_evaluate_from_generator(
    model,
    generate_test,
    tensor_maps_in,
    tensor_maps_out,
    steps,
    hidden_layer,
    plot_path,
    alpha,
):
    layer_names = [layer.name for layer in model.layers]
    model_predictions = [
        tm.output_name() for tm in tensor_maps_out if tm.output_name() in layer_names
    ]
    scalar_predictions = {
        tm.output_name(): []
        for tm in tensor_maps_out
        if len(tm.shape) == 1 and tm.output_name() in layer_names
    }
    test_labels = {tm.output_name(): [] for tm in tensor_maps_out if len(tm.shape) == 1}

    logging.info(
        f" in scalar predict {model_predictions} scalar predict names:"
        f" {scalar_predictions.keys()} test labels: {test_labels.keys()}",
    )
    embeddings = []
    test_paths = []
    for i in range(steps):
        batch = next(generate_test)
        input_data, output_data, tensor_paths = (
            batch[BATCH_INPUT_INDEX],
            batch[BATCH_OUTPUT_INDEX],
            batch[BATCH_PATHS_INDEX],
        )
        y_predictions = model.predict(input_data)
        test_paths.extend(tensor_paths)
        if hidden_layer in layer_names:
            x_embed = embed_model_predict(
                model, tensor_maps_in, hidden_layer, input_data, 2,
            )
            embeddings.extend(
                np.copy(
                    np.reshape(x_embed, (x_embed.shape[0], np.prod(x_embed.shape[1:]))),
                ),
            )

        for tm_output_name in test_labels:
            test_labels[tm_output_name].extend(np.copy(output_data[tm_output_name]))

        for y, tm_output_name in zip(y_predictions, model_predictions):
            if not isinstance(
                y_predictions, list,
            ):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
                y = y_predictions
            if tm_output_name in scalar_predictions:
                scalar_predictions[tm_output_name].extend(np.copy(y))

    performance_metrics = {}
    scatters = []
    rocs = []
    for tm in tensor_maps_out:
        if tm.output_name() in scalar_predictions:
            y_predict = np.array(scalar_predictions[tm.output_name()])
            y_truth = np.array(test_labels[tm.output_name()])
            performance_metrics.update(
                evaluate_predictions(
                    tm,
                    y_predict,
                    y_truth,
                    tm.name,
                    plot_path,
                    test_paths,
                    rocs=rocs,
                    scatters=scatters,
                ),
            )

    if len(rocs) > 1:
        subplot_rocs(rocs, plot_path)
    if len(scatters) > 1:
        subplot_scatters(scatters, plot_path)
    if len(embeddings) > 0:
        test_labels_1d = {
            tm: np.array(test_labels[tm.output_name()])
            for tm in tensor_maps_out
            if tm.output_name() in test_labels
        }
        _tsne_wrapper(
            model,
            hidden_layer,
            alpha,
            plot_path,
            test_paths,
            test_labels_1d,
            embeddings=embeddings,
        )

    return performance_metrics


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
    model,
    hidden_layer_name,
    alpha,
    plot_path,
    test_paths,
    test_labels,
    test_data=None,
    tensor_maps_in=None,
    batch_size=16,
    embeddings=None,
):
    """Plot 2D t-SNE of a model's hidden layer colored by many different co-variates.

    Callers must provide either model's embeddings or test_data on which embeddings will be inferred

    :param model: Keras model
    :param hidden_layer_name: String name of the hidden layer whose embeddings will be visualized
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
