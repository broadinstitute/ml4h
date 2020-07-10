# Imports: standard library
import gc
import os
import logging
import argparse
from timeit import default_timer as timer
from typing import Dict, List
from collections import Counter

# Imports: third party
import numpy as np
import hyperopt
from hyperopt import hp, tpe, fmin
from skimage.filters import threshold_otsu

# Imports: first party
from ml4cvd.plots import plot_metric_history
from ml4cvd.models import train_model_from_generators, make_multimodal_multitask_model
from ml4cvd.defines import IMAGE_EXT, MODEL_EXT, Arguments
from ml4cvd.arguments import parse_args
from ml4cvd.tensor_generators import (
    big_batch_from_minibatch_generator,
    test_train_valid_tensor_generators,
)
from ml4cvd.tensor_maps_by_script import TMAPS

# fmt: off
# need matplotlib -> Agg -> pyplot
import matplotlib                       # isort:skip
matplotlib.use("Agg")                   # isort:skip
from matplotlib import pyplot as plt    # isort:skip
# fmt: on


MAX_LOSS = 9e9


def run(args: argparse.Namespace):
    # Keep track of elapsed execution time
    start_time = timer()
    try:
        if "ecg_architecture" == args.mode:
            optimize_ecg_architecture(args)
        else:
            raise ValueError("Unknown hyperparameter optimization mode:", args.mode)

    except Exception as e:
        logging.exception(e)

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info(
        "Executed the '{}' operation in {:.2f} seconds".format(args.mode, elapsed_time),
    )


def optimize_ecg_architecture(args: argparse.Namespace):
    block_size_sets = [2, 3, 4]
    conv_layers_sets = [[32]]  # Baseline
    conv_normalize_sets = ["", "batch_norm"]
    dense_layers_sets = [
        [256],
        [1000],  # Collin's suggestion
        [16, 64],  # Baseline
    ]
    dense_blocks_sets = [
        [64, 128],  # Collin's suggestion
        [32, 24, 16],  # Baseline
        [48, 36, 24, 16],
    ]
    pool_types = ["max", "average"]

    """
    conv_x_sets = _generate_conv1D_filter_widths(
        num_unique_filters=25,
        list_len_bounds=[1, 4],
        first_filter_width_bounds=[60, 150],
        probability_vary_filter_width=0.75,
        vary_filter_scale_bounds=[1.25, 1.75],
    )
    """
    conv_x_sets = [
        [50],
        [71],
        [100],
        [80, 40, 20, 10],
    ]
    conv_dropout_sets = [0, 0.25, 0.5]
    dropout_sets = [0, 0.25, 0.5]
    learning_rate_sets = [0.0002, 0.0004]

    # Generate weighted loss tmaps for STS death
    weighted_losses = [val for val in range(1, 20, 4)]
    input_tensors_sets = _generate_weighted_loss_tmaps(
        base_tmap_name="sts_death", weighted_losses=weighted_losses,
    )

    space = {
        "block_size": hp.choice("block_size", block_size_sets),
        "conv_layers": hp.choice("conv_layers", conv_layers_sets),
        "conv_normalize": hp.choice("conv_normalize", conv_normalize_sets),
        "conv_x": hp.choice("conv_x", conv_x_sets),
        "conv_dropout": hp.choice("conv_dropout", conv_dropout_sets),
        "dropout": hp.choice("dropout", dropout_sets),
        "dense_blocks": hp.choice("dense_blocks", dense_blocks_sets),
        "dense_layers": hp.choice("dense_layers", dense_layers_sets),
        "input_tensors": hp.choice("input_tensors", input_tensors_sets),
        "learning_rate": hp.choice("learning_rate", learning_rate_sets),
        "dropout": hp.choice("dropout", dropout_sets),
        "pool_type": hp.choice("pool_type", pool_types),
    }

    param_lists = {
        "block_size": block_size_sets,
        "conv_layers": conv_layers_sets,
        "conv_normalize": conv_normalize_sets,
        "conv_x": conv_x_sets,
        "conv_dropout": conv_dropout_sets,
        "dense_blocks": dense_blocks_sets,
        "dense_layers": dense_layers_sets,
        "dropout": dropout_sets,
        "input_tensors": input_tensors_sets,
        "learning_rate": learning_rate_sets,
        "pool_type": pool_types,
    }
    hyperparameter_optimizer(args, space, param_lists)


def hyperparameter_optimizer(
    args: argparse.Namespace,
    space: Dict[str, hyperopt.pyll.base.Apply],
    param_lists: Arguments,
):
    args.keep_paths = False
    args.keep_paths_test = False
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    test_data, test_labels = big_batch_from_minibatch_generator(
        generate_test, args.test_steps,
    )
    generate_test.kill_workers()
    histories = []
    fig_path = os.path.join(args.output_folder, args.id, "plots")
    i = 0

    def loss_from_multimodal_multitask(x: Arguments):
        model = None
        history = None
        nonlocal i
        i += 1
        try:
            set_args_from_x(args, x)
            model = make_multimodal_multitask_model(**args.__dict__)

            if model.count_params() > args.max_parameters:
                logging.info(
                    f"Model too big, max parameters is:{args.max_parameters}, model"
                    f" has:{model.count_params()}. Return max loss.",
                )
                return MAX_LOSS
            generate_train, generate_valid, _ = test_train_valid_tensor_generators(
                **args.__dict__
            )
            model, history = train_model_from_generators(
                model=model,
                generate_train=generate_train,
                generate_valid=generate_valid,
                training_steps=args.training_steps,
                validation_steps=args.validation_steps,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                output_folder=args.output_folder,
                run_id=args.id,
                inspect_model=args.inspect_model,
                inspect_show_labels=args.inspect_show_labels,
                return_history=True,
                plot=False,
            )
            history.history["parameter_count"] = [model.count_params()]
            histories.append(history.history)
            title = (  # refer to loss_by_params.txt to find the params for this trial
                f"trial_{i}"
            )
            plot_metric_history(history, args.training_steps, title, fig_path)
            model.load_weights(
                os.path.join(args.output_folder, args.id, args.id + MODEL_EXT),
            )
            loss_and_metrics = model.evaluate(
                test_data, test_labels, batch_size=args.batch_size,
            )
            logging.info(
                f"Current architecture:\n{_string_from_architecture_dict(x)}\nCurrent"
                f" model size: {model.count_params()}.",
            )
            logging.info(
                f"Iteration {i} out of maximum {args.max_evals}\nTest Loss:"
                f" {loss_and_metrics[0]}",
            )
            generate_train.kill_workers()
            generate_valid.kill_workers()
            return loss_and_metrics[0]

        except ValueError:
            logging.exception(
                "ValueError trying to make a model for hyperparameter optimization."
                " Returning max loss.",
            )
            return MAX_LOSS
        except:
            logging.exception(
                "Error trying hyperparameter optimization. Returning max loss.",
            )
            return MAX_LOSS
        finally:
            del model
            gc.collect()
            if history is None:
                histories.append(
                    {
                        "loss": [MAX_LOSS],
                        "val_loss": [MAX_LOSS],
                        "parameter_count": [0],
                    },
                )

    trials = hyperopt.Trials()

    fmin(
        fn=loss_from_multimodal_multitask,
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
    )
    plot_trials(trials, histories, fig_path, param_lists)
    logging.info("Saved learning plot to:{}".format(fig_path))


def set_args_from_x(args: argparse.Namespace, x: Arguments):
    for k in args.__dict__:
        if k in x:
            logging.info(f"arg: {k}")
            logging.info(f"value from hyperopt: {x[k]}")
            logging.info(f"original value in args: {args.__dict__[k]}")
            if isinstance(args.__dict__[k], int):
                args.__dict__[k] = int(x[k])
            elif isinstance(args.__dict__[k], float):
                v = float(x[k])
                if v == int(v):
                    v = int(v)
                args.__dict__[k] = v
            elif isinstance(args.__dict__[k], list):
                if isinstance(x[k], tuple):
                    args.__dict__[k] = list(x[k])
            else:
                args.__dict__[k] = x[k]
            logging.info(f"value in args is now: {args.__dict__[k]}\n")
    logging.info(f"Set arguments to: {args}")
    args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
    args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]


def _ensure_even_number(num: int) -> int:
    if num % 2 == 1:
        num += 1
    return num


def _generate_conv1D_filter_widths(
    num_unique_filters: int = 50,
    list_len_bounds: List[int] = [1, 4],
    first_filter_width_bounds: List[int] = [60, 150],
    probability_vary_filter_width: float = 0.75,
    vary_filter_scale_bounds: List[float] = [1.25, 2.00],
) -> List[List[int]]:
    """Generate a list of 1D convolutional filter widths that are lists of even ints.

    :param num_unique_filters: number of unique lists of filters to generate,
        e.g. 10 will result in a list of 10 lists of filter widths.

    :param list_len_bounds: bounds of the number of elements in each list of filters;
        the number of elements is a randomly selected integer in these bounds.
        e.g. [1, 4] will choose a random int from among 1, 2, 3, or 4.

    :param first_filter_width_bounds: bounds of the first filter width; randomly
        selected integer in these bounds similar to 'list_len_bounds'.

    :param probability_vary_filter_width: probability of choosing to vary filter size;
        a randomly generated float between 0-1 is compared to this value. If <=, then
        the filter size is varied.

    :param vary_filter_scale_bounds: bounds of the scale factor for decreasing filter
        width in subsequent layers; the scale is a randomly selected float in these
        bounds. The filter width of the next layer equals the filter width of the prior
        layer divided by the filter_scale. This scale factor is applied to all layers:
           ```
           list_len = 4
           first_filter_width = 100
           filter_scale = 1.5
           ```
        These settings would result in the following filter widths: [100, 66, 44, 30]
    """
    list_of_filters = []

    while len(list_of_filters) < num_unique_filters:

        # Generate length of filter sizes
        list_len = np.random.randint(
            low=list_len_bounds[0], high=list_len_bounds[1] + 1, size=1, dtype=int,
        )[0]

        # Generate first filter size
        first_filter_width = np.random.randint(
            low=first_filter_width_bounds[0],
            high=first_filter_width_bounds[1] + 1,
            size=1,
            dtype=int,
        )[0]
        first_filter_width = _ensure_even_number(first_filter_width)

        # Randomly determine if filter size varies or not
        vary_filter_width = probability_vary_filter_width >= np.random.rand()

        # Randomly generate filter scale value by which to divide subsequent filter sizes
        vary_filter_scale = np.random.uniform(
            low=vary_filter_scale_bounds[0], high=vary_filter_scale_bounds[1],
        )

        # Iterate through list of filter sizes
        this_filter = []
        for i in range(list_len):
            this_filter.append(first_filter_width)

            # Check if we want to vary filter size
            if vary_filter_width:
                current_filter_width = first_filter_width
                first_filter_width = int(first_filter_width / vary_filter_scale)
                first_filter_width = _ensure_even_number(first_filter_width)

                # If reducing filter size makes it 0, reset to prior filter size
                if first_filter_width == 0:
                    first_filter_width = current_filter_width

        if this_filter not in list_of_filters:
            list_of_filters.append(this_filter)

    return list_of_filters


def _generate_weighted_loss_tmaps(
    base_tmap_name: str, weighted_losses: List[int],
) -> List[str]:
    new_tmap_names = [
        base_tmap_name + "_weighted_loss_" + str(weight) for weight in weighted_losses
    ]
    return new_tmap_names


def _string_from_architecture_dict(x: Arguments):
    return "\n".join([f"{k} = {x[k]}" for k in x])


def _string_from_trials(trials: hyperopt.Trials, index: int, param_lists: Dict = {}):
    s = ""
    x = trials.trials[index]["misc"]["vals"]
    for k in x:
        s += "\n" + k + " = "
        v = x[k][0]
        if k in param_lists:
            s += str(param_lists[k][int(v)])
        elif k in ["num_layers", "layer_width"]:
            s += str(int(v))
        elif v < 1:
            s += f"{v:.2E}"
        else:
            s += f"{v:.2f}"
    return s


def _model_label_from_losses_and_histories(
    i: int,
    all_losses: np.array,
    histories: List[Dict],
    trials: hyperopt.Trials,
    param_lists: dict,
):
    label = (
        f"Trial {i}: \nTest Loss:{all_losses[i]:.3f}\nTrain"
        f' Loss:{histories[i]["loss"][-1]:.3f}\nValidation'
        f' Loss:{histories[i]["val_loss"][-1]:.3f}'
    )
    label += f'\nModel parameter count: {histories[i]["parameter_count"][-1]}'
    label += f"{_string_from_trials(trials, i, param_lists)}"
    return label


def plot_trials(
    trials: hyperopt.Trials,
    histories: List[Dict],
    figure_path: str,
    param_lists: Dict = {},
):
    all_losses = np.array(trials.losses())  # the losses we will put in the text
    real_losses = all_losses[all_losses != MAX_LOSS]
    cutoff = MAX_LOSS
    try:
        cutoff = threshold_otsu(real_losses)
    except ValueError:
        logging.info("Otsu thresholding failed. Using MAX_LOSS for threshold.")
    lplot = np.clip(all_losses, a_min=-np.inf, a_max=cutoff)  # the losses we will plot
    plt.figure(figsize=(64, 64))
    matplotlib.rcParams.update({"font.size": 9})
    colors = ["r" if x == cutoff else "b" for x in lplot]
    plt.plot(lplot)
    with open(os.path.join(figure_path, "loss_by_params.txt"), "w") as f:
        for i in range(len(trials.trials)):
            label = _model_label_from_losses_and_histories(
                i, all_losses, histories, trials, param_lists,
            )
            plt.text(i, lplot[i], label, color=colors[i])
            f.write(label.replace("\n", ",") + "\n")
    plt.xlabel("Iterations")
    plt.ylabel("Losses")
    plt.ylim(min(lplot) * 0.95, max(lplot) * 1.05)
    plt.title(f"Hyperparameter Optimization\n")
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.axhline(
        cutoff, label=f"Loss display cutoff at {cutoff:.3f}", color="r", linestyle="--",
    )
    loss_path = os.path.join(figure_path, "loss_per_iteration" + IMAGE_EXT)
    plt.legend()
    plt.savefig(loss_path)
    logging.info("Saved loss plot to: {}".format(loss_path))

    fig, [ax1, ax3, ax2] = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(60, 20),
        sharey="all",
        gridspec_kw={"width_ratios": [2, 1, 2]},
    )
    cm = plt.get_cmap("gist_rainbow")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    linestyles = "solid", "dotted", "dashed", "dashdot"
    for i, history in enumerate(histories):
        color = cm(i / len(histories))
        training_loss = np.clip(history["loss"], a_min=-np.inf, a_max=cutoff)
        val_loss = np.clip(history["val_loss"], a_min=-np.inf, a_max=cutoff)
        label = _model_label_from_losses_and_histories(
            i, all_losses, histories, trials, param_lists,
        )
        ax1.plot(training_loss, label=label, linestyle=linestyles[i % 4], color=color)
        ax1.text(len(training_loss) - 1, training_loss[-1], str(i))
        ax2.plot(val_loss, label=label, linestyle=linestyles[i % 4], color=color)
        ax2.text(len(val_loss) - 1, val_loss[-1], str(i))
    ax1.axhline(
        cutoff, label=f"Loss display cutoff at {cutoff:.3f}", color="k", linestyle="--",
    )
    ax1.set_title("Training Loss")
    ax2.axhline(
        cutoff, label=f"Loss display cutoff at {cutoff:.3f}", color="k", linestyle="--",
    )
    ax2.set_title("Validation Loss")
    ax3.legend(
        *ax2.get_legend_handles_labels(),
        loc="upper center",
        fontsize="x-small",
        mode="expand",
        ncol=5,
    )
    ax3.axis("off")
    learning_path = os.path.join(figure_path, "learning_curves" + IMAGE_EXT)
    plt.tight_layout()
    plt.savefig(learning_path)
    logging.info("Saved learning curve plot to: {}".format(learning_path))


if __name__ == "__main__":
    args = parse_args()
    run(args)
