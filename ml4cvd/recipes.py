# Imports: standard library
import os
import csv
import copy
import logging
import argparse
from timeit import default_timer as timer
from typing import Dict
from functools import reduce
from collections import Counter, defaultdict

# Imports: third party
import numpy as np
import tensorflow as tf

# Imports: first party
from ml4cvd.plots import (
    plot_ecg,
    plot_rocs,
    plot_scatters,
    plot_roc_per_class,
    plot_saliency_maps,
    plot_precision_recalls,
    subplot_comparison_rocs,
    subplot_comparison_scatters,
)
from ml4cvd.models import (
    saliency_map,
    make_shallow_model,
    make_siamese_model,
    make_hidden_layer_model,
    get_model_inputs_outputs,
    train_model_from_generators,
    make_multimodal_multitask_model,
)
from ml4cvd.defines import MODEL_EXT, TENSOR_EXT
from ml4cvd.metrics import (
    log_aucs,
    get_roc_aucs,
    get_pearson_coefficients,
    log_pearson_coefficients,
    get_precision_recall_aucs,
)
from ml4cvd.arguments import parse_args
from ml4cvd.evaluations import (
    predict_and_evaluate,
    predict_scalars_and_evaluate_from_generator,
)
from ml4cvd.explorations import explore, cross_reference
from ml4cvd.hyperparameters import hyperoptimize
from ml4cvd.tensor_generators import (
    BATCH_INPUT_INDEX,
    BATCH_PATHS_INDEX,
    BATCH_OUTPUT_INDEX,
    TensorGenerator,
    get_verbose_stats_string,
    big_batch_from_minibatch_generator,
    train_valid_test_tensor_generators,
)
from ml4cvd.tensor_writer_ecg import write_tensors_ecg


def run(args):
    start_time = timer()  # Keep track of elapsed execution time
    try:
        if "train" == args.mode:
            train_multimodal_multitask(args)
        elif "test" == args.mode:
            test_multimodal_multitask(args)
        elif "tensorize" == args.mode:
            write_tensors_ecg(args.xml_folder, args.tensors, args.num_workers)
        elif "explore" == args.mode:
            explore(args)
        elif "cross_reference" == args.mode:
            cross_reference(args)
        elif "compare" == args.mode:
            compare_multimodal_multitask_models(args)
        elif "infer" == args.mode:
            infer_multimodal_multitask(args)
        elif "infer_hidden" == args.mode:
            infer_hidden_layer_multimodal_multitask(args)
        elif "test_scalar" == args.mode:
            test_multimodal_scalar_tasks(args)
        elif "compare_scalar" == args.mode:
            compare_multimodal_scalar_task_models(args)
        elif "plot_saliency" == args.mode:
            saliency_maps(args)
        elif "plot_ecg" == args.mode:
            plot_ecg(args)
        elif "train_shallow" == args.mode:
            train_shallow_model(args)
        elif "train_siamese" == args.mode:
            train_siamese_model(args)
        elif "hyperoptimize" == args.mode:
            hyperoptimize(args)
        else:
            raise ValueError("Unknown mode:", args.mode)

    except Exception as e:
        logging.exception(e)

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info(
        "Executed the '{}' operation in {:.2f} seconds".format(args.mode, elapsed_time),
    )


def train_multimodal_multitask(args):
    generate_train, generate_valid, generate_test = train_valid_test_tensor_generators(
        **args.__dict__
    )
    model = make_multimodal_multitask_model(**args.__dict__)
    model, history = train_model_from_generators(
        model=model,
        generate_train=generate_train,
        generate_valid=generate_valid,
        training_steps=args.training_steps,
        validation_steps=args.validation_steps,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate_patience=args.learning_rate_patience,
        learning_rate_reduction=args.learning_rate_reduction,
        output_folder=args.output_folder,
        run_id=args.id,
        return_history=True,
    )
    out_path = os.path.join(args.output_folder, args.id + "/")

    train_data, train_labels = big_batch_from_minibatch_generator(
        generate_train, args.training_steps,
    )
    predict_and_evaluate(
        model=model,
        test_data=train_data,
        test_labels=train_labels,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        batch_size=args.batch_size,
        hidden_layer=args.hidden_layer,
        plot_path=out_path,
        test_paths=None,
        embed_visualization=args.embed_visualization,
        alpha=args.alpha,
        data_split="train",
    )

    generate_train.kill_workers()
    generate_valid.kill_workers()

    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(
        generate_test, args.test_steps,
    )
    generate_test.kill_workers()

    logging.info(f"Model trained for {len(history.history['loss'])} epochs")

    if isinstance(generate_train, TensorGenerator):
        logging.info(
            get_verbose_stats_string(
                {
                    "train": generate_train,
                    "valid": generate_valid,
                    "test": generate_test,
                },
            ),
        )

    performance_metrics = predict_and_evaluate(
        model=model,
        test_data=test_data,
        test_labels=test_labels,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        batch_size=args.batch_size,
        hidden_layer=args.hidden_layer,
        plot_path=out_path,
        test_paths=test_paths,
        embed_visualization=args.embed_visualization,
        alpha=args.alpha,
        data_split="test",
    )
    return performance_metrics


def test_multimodal_multitask(args):
    _, _, generate_test = train_valid_test_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    out_path = os.path.join(args.output_folder, args.id + "/")
    data, labels, paths = big_batch_from_minibatch_generator(
        generate_test, args.test_steps,
    )
    return predict_and_evaluate(
        model,
        data,
        labels,
        args.tensor_maps_in,
        args.tensor_maps_out,
        args.batch_size,
        args.hidden_layer,
        out_path,
        paths,
        args.embed_visualization,
        args.alpha,
    )


def test_multimodal_scalar_tasks(args):
    _, _, generate_test = train_valid_test_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    p = os.path.join(args.output_folder, args.id + "/")
    return predict_scalars_and_evaluate_from_generator(
        model,
        generate_test,
        args.tensor_maps_in,
        args.tensor_maps_out,
        args.test_steps,
        args.hidden_layer,
        p,
        args.alpha,
    )


def compare_multimodal_multitask_models(args):
    _, _, generate_test = train_valid_test_tensor_generators(**args.__dict__)
    models_inputs_outputs = get_model_inputs_outputs(
        args.model_files, args.tensor_maps_in, args.tensor_maps_out,
    )
    input_data, output_data, paths = big_batch_from_minibatch_generator(
        generate_test, args.test_steps,
    )
    common_outputs = _get_common_outputs(models_inputs_outputs, "output")
    predictions = _get_predictions(
        args, models_inputs_outputs, input_data, common_outputs, "input", "output",
    )
    _calculate_and_plot_prediction_stats(args, predictions, output_data, paths)


def compare_multimodal_scalar_task_models(args):
    _, _, generate_test = train_valid_test_tensor_generators(**args.__dict__)
    models_io = get_model_inputs_outputs(
        args.model_files, args.tensor_maps_in, args.tensor_maps_out,
    )
    outs = _get_common_outputs(models_io, "output")
    predictions, labels, paths = _scalar_predictions_from_generator(
        args, models_io, generate_test, args.test_steps, outs, "input", "output",
    )
    _calculate_and_plot_prediction_stats(args, predictions, labels, paths)


def _make_tmap_nan_on_fail(tmap):
    """
    Builds a copy TensorMap with a tensor_from_file that returns nans on errors instead of raising an error
    """
    new_tmap = copy.deepcopy(tmap)
    new_tmap.validator = lambda _, x, hd5: x  # prevent failure caused by validator

    def _tff(tm, hd5, dependents=None):
        try:
            return tmap.tensor_from_file(tm, hd5, dependents)
        except (IndexError, KeyError, ValueError, OSError, RuntimeError):
            return np.full(shape=tm.shape, fill_value=np.nan)

    new_tmap.tensor_from_file = _tff
    return new_tmap


def inference_file_name(output_folder: str, id_: str) -> str:
    return os.path.join(output_folder, id_, "inference_" + id_ + ".tsv")


# TODO fix this
def infer_multimodal_multitask(args):
    stats = Counter()
    tensor_paths_inferred = set()
    inference_tsv = inference_file_name(args.output_folder, args.id)
    tsv_style_is_genetics = "genetics" in args.tsv_style
    tensor_paths = [
        os.path.join(args.tensors, tp)
        for tp in sorted(os.listdir(args.tensors))
        if os.path.splitext(tp)[-1].lower() == TENSOR_EXT
    ]
    model = make_multimodal_multitask_model(**args.__dict__)
    no_fail_tmaps_out = [_make_tmap_nan_on_fail(tmap) for tmap in args.tensor_maps_out]
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    generate_test = TensorGenerator(
        1,
        args.tensor_maps_in,
        no_fail_tmaps_out,
        tensor_paths,
        num_workers=0,
        cache_size=0,
        keep_paths=True,
        mixup=args.mixup_alpha,
    )
    generate_test.set_worker_paths(tensor_paths)
    with open(inference_tsv, mode="w") as inference_file:
        # TODO: csv.DictWriter is much nicer for this
        inference_writer = csv.writer(
            inference_file, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL,
        )
        header = ["sample_id"]
        if tsv_style_is_genetics:
            header = ["FID", "IID"]
        for ot, otm in zip(args.output_tensors, args.tensor_maps_out):
            if len(otm.shape) == 1 and otm.is_continuous():
                header.extend([ot + "_prediction", ot + "_actual"])
            elif len(otm.shape) == 1 and otm.is_categorical():
                channel_columns = []
                for k in otm.channel_map:
                    channel_columns.append(ot + "_" + k + "_prediction")
                    channel_columns.append(ot + "_" + k + "_actual")
                header.extend(channel_columns)
        inference_writer.writerow(header)

        while True:
            batch = next(generate_test)
            input_data, output_data, tensor_paths = (
                batch[BATCH_INPUT_INDEX],
                batch[BATCH_OUTPUT_INDEX],
                batch[BATCH_PATHS_INDEX],
            )
            if tensor_paths[0] in tensor_paths_inferred:
                next(generate_test)  # this prints end of epoch info
                logging.info(
                    f"Inference on {stats['count']} tensors finished. Inference TSV"
                    f" file at: {inference_tsv}",
                )
                break

            prediction = model.predict(input_data)
            if len(no_fail_tmaps_out) == 1:
                prediction = [prediction]

            csv_row = [
                os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, ""),
            ]  # extract sample id
            if tsv_style_is_genetics:
                csv_row *= 2
            for y, tm in zip(prediction, no_fail_tmaps_out):
                if len(tm.shape) == 1 and tm.is_continuous():
                    csv_row.append(
                        str(tm.rescale(y)[0][0]),
                    )  # first index into batch then index into the 1x1 structure
                    if (
                        tm.sentinel is not None
                        and tm.sentinel == output_data[tm.output_name()][0][0]
                    ) or np.isnan(output_data[tm.output_name()][0][0]):
                        csv_row.append("NA")
                    else:
                        csv_row.append(
                            str(tm.rescale(output_data[tm.output_name()])[0][0]),
                        )
                elif len(tm.shape) == 1 and tm.is_categorical():
                    for k, i in tm.channel_map.items():
                        csv_row.append(str(y[0][tm.channel_map[k]]))
                        actual = output_data[tm.output_name()][0][i]
                        csv_row.append("NA" if np.isnan(actual) else str(actual))

            inference_writer.writerow(csv_row)
            tensor_paths_inferred.add(tensor_paths[0])
            stats["count"] += 1
            if stats["count"] % 250 == 0:
                logging.info(
                    f"Wrote:{stats['count']} rows of inference.  Last"
                    f" tensor:{tensor_paths[0]}",
                )


def hidden_inference_file_name(output_folder: str, id_: str) -> str:
    return os.path.join(output_folder, id_, "hidden_inference_" + id_ + ".tsv")


def infer_hidden_layer_multimodal_multitask(args):
    stats = Counter()
    args.num_workers = 0
    inference_tsv = hidden_inference_file_name(args.output_folder, args.id)
    tsv_style_is_genetics = "genetics" in args.tsv_style
    tensor_paths = [
        os.path.join(args.tensors, tp)
        for tp in sorted(os.listdir(args.tensors))
        if os.path.splitext(tp)[-1].lower() == TENSOR_EXT
    ]
    # hard code batch size to 1 so we can iterate over file names and generated tensors together in the tensor_paths for loop
    generate_test = TensorGenerator(
        1,
        args.tensor_maps_in,
        args.tensor_maps_out,
        tensor_paths,
        num_workers=0,
        cache_size=args.cache_size,
        keep_paths=True,
        mixup=args.mixup_alpha,
    )
    generate_test.set_worker_paths(tensor_paths)
    full_model = make_multimodal_multitask_model(**args.__dict__)
    embed_model = make_hidden_layer_model(
        full_model, args.tensor_maps_in, args.hidden_layer,
    )
    dummy_input = {
        tm.input_name(): np.zeros(
            (1,) + full_model.get_layer(tm.input_name()).input_shape[0][1:],
        )
        for tm in args.tensor_maps_in
    }
    dummy_out = embed_model.predict(dummy_input)
    latent_dimensions = int(np.prod(dummy_out.shape[1:]))
    logging.info(
        f"Dummy output shape is: {dummy_out.shape} latent dimensions:"
        f" {latent_dimensions}",
    )
    with open(inference_tsv, mode="w") as inference_file:
        inference_writer = csv.writer(
            inference_file, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL,
        )
        header = ["FID", "IID"] if tsv_style_is_genetics else ["sample_id"]
        header += [f"latent_{i}" for i in range(latent_dimensions)]
        inference_writer.writerow(header)

        while True:
            batch = next(generate_test)
            input_data, tensor_paths = (
                batch[BATCH_INPUT_INDEX],
                batch[BATCH_PATHS_INDEX],
            )
            if tensor_paths[0] in stats:
                next(generate_test)  # this prints end of epoch info
                logging.info(
                    f"Latent space inference on {stats['count']} tensors finished."
                    f" Inference TSV file at: {inference_tsv}",
                )
                break

            sample_id = os.path.basename(tensor_paths[0]).replace(TENSOR_EXT, "")
            prediction = embed_model.predict(input_data)
            prediction = np.reshape(prediction, (latent_dimensions,))
            csv_row = [sample_id, sample_id] if tsv_style_is_genetics else [sample_id]
            csv_row += [f"{prediction[i]}" for i in range(latent_dimensions)]
            inference_writer.writerow(csv_row)
            stats[tensor_paths[0]] += 1
            stats["count"] += 1
            if stats["count"] % 500 == 0:
                logging.info(
                    f"Wrote:{stats['count']} rows of latent space inference.  Last"
                    f" tensor:{tensor_paths[0]}",
                )


def train_shallow_model(args: argparse.Namespace) -> Dict[str, float]:
    """
    Train a shallow model (e.g. linear or logistic regression) and return performance metrics.
    """
    generate_train, generate_valid, generate_test = train_valid_test_tensor_generators(
        **args.__dict__
    )
    model = make_shallow_model(
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        training_steps=args.training_steps,
        model_file=args.model_file,
        model_layers=args.model_layers,
    )
    model = train_model_from_generators(
        model=model,
        generate_train=generate_train,
        generate_valid=generate_valid,
        training_steps=args.training_steps,
        validation_steps=args.validation_steps,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate_patience=args.learning_rate_patience,
        learning_rate_reduction=args.learning_rate_reduction,
        output_folder=args.output_folder,
        run_id=args.id,
    )

    p = os.path.join(args.output_folder, args.id + "/")

    train_data, train_labels = big_batch_from_minibatch_generator(
        generate_train, args.training_steps,
    )
    generate_train.kill_workers()
    generate_valid.kill_workers()
    predict_and_evaluate(
        model=model,
        test_data=train_data,
        test_labels=train_labels,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        batch_size=args.batch_size,
        hidden_layer=args.hidden_layer,
        plot_path=p,
        test_paths=None,
        embed_visualization=args.embed_visualization,
        alpha=args.alpha,
        data_split="train",
    )
    test_data, test_labels, test_paths = big_batch_from_minibatch_generator(
        generate_test, args.test_steps,
    )
    generate_test.kill_workers()
    performance_metrics = predict_and_evaluate(
        model=model,
        test_data=test_data,
        test_labels=test_labels,
        tensor_maps_in=args.tensor_maps_in,
        tensor_maps_out=args.tensor_maps_out,
        batch_size=args.batch_size,
        hidden_layer=args.hidden_layer,
        plot_path=p,
        test_paths=test_paths,
        embed_visualization=args.embed_visualization,
        alpha=args.alpha,
        save_coefficients=True,
        data_split="test",
    )
    return performance_metrics


def train_siamese_model(args):
    base_model = make_multimodal_multitask_model(**args.__dict__)
    siamese_model = make_siamese_model(base_model, **args.__dict__)
    generate_train, generate_valid, generate_test = train_valid_test_tensor_generators(
        **args.__dict__, siamese=True
    )
    siamese_model = train_model_from_generators(
        model=siamese_model,
        generate_train=generate_train,
        generate_valid=generate_valid,
        training_steps=args.training_steps,
        validation_steps=args.validation_steps,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate_patience=args.learning_rate_patience,
        learning_rate_reduction=args.learning_rate_reduction,
        output_folder=args.output_folder,
        run_id=args.id,
    )

    data, labels, paths = big_batch_from_minibatch_generator(
        generate_test, args.test_steps,
    )
    prediction = siamese_model.predict(data)
    return plot_roc_per_class(
        prediction,
        labels["output_siamese"],
        {"random_siamese_verification_task": 0},
        args.id,
        os.path.join(args.output_folder, args.id + "/"),
    )


def saliency_maps(args):
    tf.compat.v1.disable_eager_execution()
    _, _, generate_test = train_valid_test_tensor_generators(**args.__dict__)
    model = make_multimodal_multitask_model(**args.__dict__)
    data, labels, paths = big_batch_from_minibatch_generator(
        generate_test, args.test_steps,
    )
    in_tensor = data[args.tensor_maps_in[0].input_name()]
    for tm in args.tensor_maps_out:
        if len(tm.shape) > 1:
            continue
        for channel in tm.channel_map:
            gradients = saliency_map(
                in_tensor, model, tm.output_name(), tm.channel_map[channel],
            )
            plot_saliency_maps(
                in_tensor,
                gradients,
                os.path.join(
                    args.output_folder, f"{args.id}/saliency_maps/{tm.name}_{channel}",
                ),
            )


def _get_common_outputs(models_inputs_outputs, output_prefix):
    """Returns a set of outputs common to all the models so we can compare the models according to those outputs only"""
    all_outputs = []
    for (_, ios) in models_inputs_outputs.items():
        outputs = {k: v for (k, v) in ios.items() if k == output_prefix}
        for (_, output) in outputs.items():
            all_outputs.append(set(output))
    return reduce(set.intersection, all_outputs)


def _get_predictions(
    args, models_inputs_outputs, input_data, outputs, input_prefix, output_prefix,
):
    """Makes multi-modal predictions for a given number of models.

    Returns:
        dict: The nested dictionary of predicted values.

            {
                'tensor_map_1':
                    {
                        'model_1': [[value1, value2], [...]],
                        'model_2': [[value3, value4], [...]]
                    },
                'tensor_map_2':
                    {
                        'model_2': [[value5, value6], [...]],
                        'model_3': [[value7, value8], [...]]
                    }
            }
    """
    predictions = defaultdict(dict)
    for model_file in models_inputs_outputs.keys():
        args.tensor_maps_out = models_inputs_outputs[model_file][output_prefix]
        args.tensor_maps_in = models_inputs_outputs[model_file][input_prefix]
        args.model_file = model_file
        model = make_multimodal_multitask_model(**args.__dict__)
        model_name = os.path.basename(model_file).replace(MODEL_EXT, "_")

        # We can feed 'model.predict()' the entire input data because it knows what subset to use
        y_pred = model.predict(input_data, batch_size=args.batch_size)

        for i, tm in enumerate(args.tensor_maps_out):
            if tm in outputs:
                if len(args.tensor_maps_out) == 1:
                    predictions[tm][model_name] = y_pred
                else:
                    predictions[tm][model_name] = y_pred[i]

    return predictions


def _scalar_predictions_from_generator(
    args, models_inputs_outputs, generator, steps, outputs, input_prefix, output_prefix,
):
    """Makes multi-modal scalar predictions for a given number of models.

    Returns:
        dict: The nested dictionary of predicted values.

            {
                'tensor_map_1':
                    {
                        'model_1': [[value1, value2], [...]],
                        'model_2': [[value3, value4], [...]]
                    },
                'tensor_map_2':
                    {
                        'model_2': [[value5, value6], [...]],
                        'model_3': [[value7, value8], [...]]
                    }
            }
    """
    models = {}
    test_paths = []
    scalar_predictions = {}
    test_labels = {
        tm.output_name(): [] for tm in args.tensor_maps_out if len(tm.shape) == 1
    }

    for model_file in models_inputs_outputs:
        args.model_file = model_file
        args.tensor_maps_in = models_inputs_outputs[model_file][input_prefix]
        args.tensor_maps_out = models_inputs_outputs[model_file][output_prefix]
        model = make_multimodal_multitask_model(**args.__dict__)
        model_name = os.path.basename(model_file).replace(MODEL_EXT, "")
        models[model_name] = model
        scalar_predictions[model_name] = [
            tm
            for tm in models_inputs_outputs[model_file][output_prefix]
            if len(tm.shape) == 1
        ]

    predictions = defaultdict(dict)
    for j in range(steps):
        batch = next(generator)
        input_data, output_data, tensor_paths = (
            batch[BATCH_INPUT_INDEX],
            batch[BATCH_OUTPUT_INDEX],
            batch[BATCH_PATHS_INDEX],
        )
        test_paths.extend(tensor_paths)
        for tl in test_labels:
            test_labels[tl].extend(np.copy(output_data[tl]))

        for model_name, model_file in zip(models, models_inputs_outputs):
            # We can feed 'model.predict()' the entire input data because it knows what subset to use
            y_predictions = models[model_name].predict(input_data)

            for y, tm in zip(
                y_predictions, models_inputs_outputs[model_file][output_prefix],
            ):
                if not isinstance(
                    y_predictions, list,
                ):  # When models have a single output model.predict returns a ndarray otherwise it returns a list
                    y = y_predictions
                if j == 0 and tm in scalar_predictions[model_name]:
                    predictions[tm][model_name] = []
                if tm in scalar_predictions[model_name]:
                    predictions[tm][model_name].extend(np.copy(y))

    for tm in predictions:
        logging.info(f"{tm.output_name()} labels: {len(test_labels[tm.output_name()])}")
        test_labels[tm.output_name()] = np.array(test_labels[tm.output_name()])
        for m in predictions[tm]:
            logging.info(
                f"{tm.output_name()} model: {m} prediction"
                f" length:{len(predictions[tm][m])}",
            )
            predictions[tm][m] = np.array(predictions[tm][m])

    return predictions, test_labels, test_paths


def _calculate_and_plot_prediction_stats(args, predictions, outputs, paths):
    rocs = []
    scatters = []
    for tm in args.tensor_maps_out:
        if tm not in predictions:
            continue
        plot_title = tm.name + "_" + args.id
        plot_folder = os.path.join(args.output_folder, args.id)

        if tm.is_categorical() and tm.static_axes() == 1:
            msg = "For tm '{}' with channel map {}: sum truth = {}; sum pred = {}"
            for m in predictions[tm]:
                logging.info(
                    msg.format(
                        tm.name,
                        tm.channel_map,
                        np.sum(outputs[tm.output_name()], axis=0),
                        np.sum(predictions[tm][m], axis=0),
                    ),
                )
            plot_rocs(
                predictions[tm],
                outputs[tm.output_name()],
                tm.channel_map,
                plot_title,
                plot_folder,
            )
            rocs.append((predictions[tm], outputs[tm.output_name()], tm.channel_map))
        elif tm.is_categorical() and tm.static_axes() == 4:
            for p in predictions[tm]:
                y = predictions[tm][p]
                melt_shape = (
                    y.shape[0] * y.shape[1] * y.shape[2] * y.shape[3],
                    y.shape[4],
                )
                predictions[tm][p] = y.reshape(melt_shape)

            y_truth = outputs[tm.output_name()].reshape(melt_shape)
            plot_rocs(predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder)
            plot_precision_recalls(
                predictions[tm], y_truth, tm.channel_map, plot_title, plot_folder,
            )
            roc_aucs = get_roc_aucs(predictions[tm], y_truth, tm.channel_map)
            precision_recall_aucs = get_precision_recall_aucs(
                predictions[tm], y_truth, tm.channel_map,
            )
            aucs = {"ROC": roc_aucs, "Precision-Recall": precision_recall_aucs}
            log_aucs(**aucs)
        elif tm.is_continuous() and tm.static_axes() == 1:
            scaled_predictions = {
                k: tm.rescale(predictions[tm][k]) for k in predictions[tm]
            }
            plot_scatters(
                scaled_predictions,
                tm.rescale(outputs[tm.output_name()]),
                plot_title,
                plot_folder,
                paths,
            )
            scatters.append(
                (
                    scaled_predictions,
                    tm.rescale(outputs[tm.output_name()]),
                    plot_title,
                    None,
                ),
            )
            coefs = get_pearson_coefficients(
                scaled_predictions, tm.rescale(outputs[tm.output_name()]),
            )
            log_pearson_coefficients(coefs, tm.name)
        else:
            scaled_predictions = {
                k: tm.rescale(predictions[tm][k]) for k in predictions[tm]
            }
            plot_scatters(
                scaled_predictions,
                tm.rescale(outputs[tm.output_name()]),
                plot_title,
                plot_folder,
            )
            coefs = get_pearson_coefficients(
                scaled_predictions, tm.rescale(outputs[tm.output_name()]),
            )
            log_pearson_coefficients(coefs, tm.name)

    if len(rocs) > 1:
        subplot_comparison_rocs(rocs, plot_folder)
    if len(scatters) > 1:
        subplot_comparison_scatters(scatters, plot_folder)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)  # back to the top
