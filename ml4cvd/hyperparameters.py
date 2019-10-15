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

import keras.backend as K

from ml4cvd.defines import IMAGE_EXT
from ml4cvd.arguments import parse_args
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

    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(40, 20), sharey='all')
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    def loss_from_multimodal_multitask(x):
        try:
            set_args_from_x(args, x)
            model = make_multimodal_multitask_model(**args.__dict__)

            if model.count_params() > args.max_parameters:
                logging.info(f"Model too big, max parameters is:{args.max_parameters}, model has:{model.count_params()}. Return max loss.")
                del model
                return MAX_LOSS

            model, history = train_model_from_generators(model, generate_train, generate_test, args.training_steps, args.validation_steps, 
                                                         args.batch_size, args.epochs, args.patience, args.output_folder, args.id, 
                                                         args.inspect_model, args.inspect_show_labels, plot=False)
            loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=args.batch_size)
            stats['count'] += 1
            logging.info('Current architecture: {}'.format(string_from_arch_dict(x)))
            logging.info('Iteration {} out of maximum {}: Loss: {} Current model size: {}.'.format(stats['count'], args.max_models, loss_and_metrics[0], model.count_params()))
            ax1.plot(history.history['loss'], label=string_from_arch_dict(x))
            ax2.plot(history.history['val_loss'], label=string_from_arch_dict(x))
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
    fig_path = os.path.join(args.output_folder, args.id, 'learning_plot'+IMAGE_EXT)
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    ax1.legend()
    ax2.legend()
    fig.savefig(fig_path)
    fig_path = os.path.join(args.output_folder, args.id, 'loss_per_iteration'+IMAGE_EXT)
    plot_trials(trials, fig_path)
    logging.info('Saved learning plot to:{}'.format(fig_path))

    # Re-train the best model so it's easy to view it at the end of the logs
    args = args_from_best_trials(args, trials, param_lists)
    model = make_multimodal_multitask_model(**args.__dict__)
    model, _ = train_model_from_generators(model, generate_train, generate_test, args.training_steps, args.validation_steps, 
                                           args.batch_size, args.epochs, args.patience, args.output_folder, args.id, 
                                           args.inspect_model, args.inspect_show_labels)


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


def set_args_from_x(args, x):
    for k in args.__dict__:
        if k in x:
            if isinstance(args.__dict__[k], int):
                args.__dict__[k] = int(x[k])
            elif isinstance(args.__dict__[k], float):
                args.__dict__[k] = float(x[k])
            else:
                args.__dict__[k] = x[k]
    logging.info(f"Set arguments to: {args}")
    args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
    args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]


def string_from_arch_dict(x):
    s = ''
    for k in x:
        s += '\n' + k + ' = '
        s += str(x[k])     
    return s


def args_from_best_trials(args, trials, param_lists={}):
    best_trial_idx = np.argmin(trials.losses())
    x = trials.trials[best_trial_idx]['misc']['vals']
    logging.info(f"got best x {x} best model is:{string_from_trials(trials, best_trial_idx, param_lists)}")
    for k in x:
        v = x[k][0]
        if k in param_lists:
            args.__dict__[k] = param_lists[k][int(v)]
        elif k in ['conv_x', 'conv_y', 'conv_z']:
            args.__dict__[k] = int(v)
        else:
            args.__dict__[k] = v
    args.tensor_maps_in = [TMAPS[it] for it in args.input_tensors]
    args.tensor_maps_out = [TMAPS[ot] for ot in args.output_tensors]
    return args   


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
        else:
            s += str(v)
    return s


def plot_trials(trials, figure_path, param_lists={}):
    lmax = max([x for x in trials.losses() if x != MAX_LOSS]) + 1  # add to the max to distinguish real losses from max loss
    lplot = [x if x != MAX_LOSS else lmax for x in trials.losses()]
    best_loss = min(lplot)
    worst_loss = max(lplot)
    std = np.std(lplot)
    plt.figure(figsize=(64, 64))
    matplotlib.rcParams.update({'font.size': 9})
    plt.plot(lplot)
    for i in range(len(trials.trials)):
        if best_loss+std > lplot[i]:
            plt.text(i, lplot[i], string_from_trials(trials, i, param_lists))
        elif worst_loss-std < lplot[i]:
            plt.text(i, lplot[i], string_from_trials(trials, i, param_lists))

    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.title('Hyperparameter Optimization\n')
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    logging.info('Saved loss plot to:{}'.format(figure_path))


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
