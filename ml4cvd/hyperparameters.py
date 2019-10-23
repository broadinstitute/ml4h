# hyperparameters.py

# Imports
import os
import logging
import numpy as np
from collections import Counter
from timeit import default_timer as timer

import hyperopt
from hyperopt import fmin, tpe, hp

import matplotlib
matplotlib.use('Agg') # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt # First import matplotlib, then use Agg, then import plt

from skimage.filters import threshold_otsu

import tensorflow.keras.backend as K

from ml4cvd.defines import IMAGE_EXT
from ml4cvd.arguments import parse_args
from ml4cvd.plots import plot_metric_history
from ml4cvd.tensor_maps_by_script import TMAPS
from ml4cvd.models import train_model_from_generators, make_multimodal_multitask_model
from ml4cvd.tensor_generators import test_train_valid_tensor_generators, big_batch_from_minibatch_generator

MAX_LOSS = 9e9


def run(args):
    # Keep track of elapsed execution time
    start_time = timer()
    try:
        if 'conv' == args.mode:
            optimize_conv_layers_multimodal_multitask(args)
        elif 'dense_layers' == args.mode:
            optimize_dense_layers_multimodal_multitask(args)
        elif 'lr' == args.mode:
            optimize_lr_multimodal_multitask(args)
        elif 'inputs' == args.mode:
            optimize_input_tensor_maps(args)
        elif 'optimizer' == args.mode:
            optimize_optimizer(args)
        elif 'architecture' == args.mode:
            optimize_architecture(args)
        else:
            raise ValueError('Unknown hyperparameter optimization mode:', args.mode)

    except Exception as e:
        logging.exception(e)

    end_time = timer()
    elapsed_time = end_time - start_time
    logging.info("Executed the '{}' operation in {:.2f} seconds".format(args.mode, elapsed_time))


def hyperparam_optimizer(args, space, param_lists={}):
    stats = Counter()
    args.keep_paths = False
    args.keep_paths_test = False
    generate_train, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    test_data, test_labels = big_batch_from_minibatch_generator(args.tensor_maps_in, args.tensor_maps_out, generate_test, args.test_steps, False)

    histories = []

    fig_path = os.path.join(args.output_folder, args.id, 'plots')
    i = 0

    def loss_from_multimodal_multitask(x):
        nonlocal i
        i += 1
        try:
            set_args_from_x(args, x)
            model = make_multimodal_multitask_model(**args.__dict__)

            if model.count_params() > args.max_parameters:
                logging.info(f"Model too big, max parameters is:{args.max_parameters}, model has:{model.count_params()}. Return max loss.")
                del model
                return MAX_LOSS

            model, history = train_model_from_generators(model, generate_train, generate_test, args.training_steps, args.validation_steps,
                                                         args.batch_size, args.epochs, args.patience, args.output_folder, args.id,
                                                         args.inspect_model, args.inspect_show_labels, True, False)
            histories.append(history.history)
            title = f'trial_{i}'  # refer to loss_by_params.txt to find the params for this trial
            plot_metric_history(history, title, fig_path)
            loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=args.batch_size)
            stats['count'] += 1
            logging.info(f'Current architecture:\n{string_from_arch_dict(x)}')
            logging.info(f"Iteration {stats['count']} out of maximum {args.max_models}\nLoss: {loss_and_metrics[0]}\nCurrent model size: {model.count_params()}.")
            del model
            return loss_and_metrics[0]

        except ValueError as e:
            logging.exception('ValueError trying to make a model for hyperparameter optimization. Returning max loss.')
            return MAX_LOSS
        except:
            logging.exception('Error trying hyperparameter optimization. Returning max loss.')
            return MAX_LOSS

    trials = hyperopt.Trials()
    fmin(loss_from_multimodal_multitask, space=space, algo=tpe.suggest, max_evals=args.max_models, trials=trials)
    plot_trials(trials, histories, fig_path, param_lists)
    logging.info('Saved learning plot to:{}'.format(fig_path))


def optimize_architecture(args):
    dense_blocks_sets = [[16], [32], [48], [32, 16], [32, 32], [32, 24, 16], [48, 32, 24], [48, 48, 48]]
    conv_layers_sets = [[64], [48], [32], [24]]
    dense_layers_sets = [[16, 64], [8, 128], [48], [32], [24], [16]]
    u_connect = [True, False]
    conv_dilate = [True, False]
    activation = ['leaky', 'prelu', 'elu', 'thresh_relu', 'relu']
    conv_bn = [True, False]
    pool_type = ['max', 'average']
    space = {
        'pool_x': hp.quniform('pool_x', 1, 4, 1),
        'conv_layers': hp.choice('conv_layers', conv_layers_sets),
        'dense_blocks': hp.choice('dense_blocks', dense_blocks_sets),
        'dense_layers': hp.choice('dense_layers', dense_layers_sets),
        'u_connect': hp.choice('u_connect', u_connect),
        'conv_dilate': hp.choice('conv_dilate', conv_dilate),
        'activation': hp.choice('activation', activation),
        'conv_bn': hp.choice('conv_bn', conv_bn),
        'pool_type': hp.choice('pool_type', pool_type),
        'dropout': hp.uniform('dropout', 0, .2),
        'conv_dropout': hp.uniform('conv_dropout', 0, .2),
        'conv_width': hp.quniform('conv_width', 2, 128, 1),
        'block_size': hp.quniform('block_size', 1, 4, 1),
    }
    param_lists = {'conv_layers': conv_layers_sets,
                   'dense_blocks': dense_blocks_sets,
                   'dense_layers': dense_layers_sets,
                   'u_connect': u_connect,
                   'conv_dilate': conv_dilate,
                   'activation': activation,
                   'conv_bn': conv_bn,
                   'pool_type': pool_type,
                   }
    hyperparam_optimizer(args, space, param_lists)


def optimize_conv_layers_multimodal_multitask(args):
    dense_blocks_sets = [[16], [32], [48], [32, 16], [32, 32], [32, 24, 16], [48, 32, 24], [48, 48, 48]]
    conv_layers_sets = [[64], [48], [32], [24]]
    dense_layers_sets = [[16, 64], [8, 128], [48], [32], [24], [16]]
    space = {
        'pool_x': hp.choice('pool_x', list(range(1, 5))),
        'conv_layers': hp.choice('conv_layers', conv_layers_sets),
        'dense_blocks': hp.choice('dense_blocks', dense_blocks_sets),
        'dense_layers': hp.choice('dense_layers', dense_layers_sets),
    }
    param_lists = {'conv_layers': conv_layers_sets, 'dense_blocks': dense_blocks_sets, 'dense_layers': dense_layers_sets}
    hyperparam_optimizer(args, space, param_lists)


def optimize_dense_layers_multimodal_multitask(args):
    space = {'num_layers': hp.choice(list(range(2, 42)))}
    hyperparam_optimizer(args, space)


def optimize_lr_multimodal_multitask(args):
    space = {'learning_rate': hp.loguniform('learning_rate', -10, -2)}
    hyperparam_optimizer(args, space)


def optimize_input_tensor_maps(args):
    input_tensor_map_sets = [['categorical-phenotypes-72'], ['mri-slice'], ['sax_inlinevf_zoom'], ['cine_segmented_sax_inlinevf'], ['ekg-leads']]
    space = {'input_tensor_maps': hp.choice('input_tensor_maps', input_tensor_map_sets),}
    param_lists = {'input_tensor_maps': input_tensor_map_sets}
    hyperparam_optimizer(args, space, param_lists)


def optimize_optimizer(args):
    optimizers = [
        'adam',
        'radam',
        'sgd',
    ]
    space = {'learning_rate': hp.loguniform('learning_rate', -10, -2),
             'optimizer': hp.choice('optimizer', optimizers)}
    hyperparam_optimizer(args, space, {'optimizer': optimizers})


def set_args_from_x(args, x):
    for k in args.__dict__:
        if k in x:
            print(k, x[k], args.__dict__[k])
            if isinstance(args.__dict__[k], int):
                args.__dict__[k] = int(x[k])
            elif isinstance(args.__dict__[k], float):
                v = float(x[k])
                if v == int(v):
                    v = int(v)
                args.__dict__[k] = v
            else:
                args.__dict__[k] = x[k]
    logging.info(f"Set arguments to: {args}")
    args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
    args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]


def string_from_arch_dict(x):
    return '\n'.join([f'{k} = {x[k]}' for k in x])


def string_from_trials(trials, index, param_lists={}):
    s = ''
    x = trials.trials[index]['misc']['vals']
    for k in x:
        s += '\n' + k + ' = '
        v = x[k][0]
        if k in param_lists:
            s += str(param_lists[k][int(v)])
        elif k in ['num_layers', 'layer_width']:
            s += str(int(v))
        elif v < 1:
            s += f'{v:.2E}'
        else:
            s += f'{v:.2f}'
    return s


def plot_trials(trials, histories, figure_path, param_lists={}):
    all_losses = np.array(trials.losses())  # the losses we will put in the text
    real_losses = all_losses[all_losses != MAX_LOSS]
    cutoff = threshold_otsu(real_losses)
    lplot = np.clip(all_losses, a_min=-np.inf, a_max=cutoff)  # the losses we will plot
    plt.figure(figsize=(64, 64))
    matplotlib.rcParams.update({'font.size': 9})
    colors = ['r' if x == cutoff else 'b' for x in lplot]
    plt.plot(lplot)
    with open(os.path.join(figure_path, 'loss_by_params.txt'), 'w') as f:
        for i in range(len(trials.trials)):
            text = f'Loss = {all_losses[i]:.3f}{string_from_trials(trials, i, param_lists)}'
            plt.text(i, lplot[i], text, color=colors[i])
            f.write(text.replace('\n', ',') + '\n')
    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.ylim(min(lplot) * .95, max(lplot) * 1.05)
    plt.title(f'Hyperparameter Optimization\n')
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.axhline(cutoff, label=f'Loss display cutoff at {cutoff:.3f}', color='r', linestyle='--')
    loss_path = os.path.join(figure_path, 'loss_per_iteration' + IMAGE_EXT)
    plt.legend()
    plt.savefig(loss_path)
    logging.info('Saved loss plot to: {}'.format(loss_path))

    fig, [ax1, ax3, ax2] = plt.subplots(nrows=1, ncols=3, figsize=(60, 20), sharey='all', gridspec_kw={'width_ratios': [2, 1, 2]})
    cm = plt.get_cmap('gist_rainbow')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    linestyles = 'solid', 'dotted', 'dashed', 'dashdot'
    for i, history in enumerate(histories):
        color = cm(i / len(histories))
        training_loss = np.clip(history['loss'], a_min=-np.inf, a_max=cutoff)
        val_loss = np.clip(history['val_loss'], a_min=-np.inf, a_max=cutoff)
        label = f'Trial {i}:\n{string_from_trials(trials, i, param_lists)}'
        ax1.plot(training_loss, label=label, linestyle=linestyles[i % 4], color=color)
        ax1.text(len(training_loss) - 1, training_loss[-1], str(i))
        ax2.plot(val_loss, label=label, linestyle=linestyles[i % 4], color=color)
        ax2.text(len(val_loss) - 1, val_loss[-1], str(i))
    ax1.axhline(cutoff, label=f'Loss display cutoff at {cutoff:.3f}', color='k', linestyle='--')
    ax1.set_title('Training Loss')
    ax2.axhline(cutoff, label=f'Loss display cutoff at {cutoff:.3f}', color='k', linestyle='--')
    ax2.set_title('Validation Loss')
    ax3.legend(*ax2.get_legend_handles_labels(), loc='upper center', fontsize='x-small', mode='expand', ncol=5)
    ax3.axis('off')
    learning_path = os.path.join(figure_path, 'learning_curves' + IMAGE_EXT)
    plt.tight_layout()
    plt.savefig(learning_path)
    logging.info('Saved learning curve plot to: {}'.format(learning_path))


def limit_mem():
    try:
        K.clear_session()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
    except AttributeError as e:
        logging.exception('Could not clear session. Maybe you are using Theano backend?')


if __name__ == '__main__':
    args = parse_args()
    run(args)  # back to the top
