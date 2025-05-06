# Training diffusion models built by diffusion_blocks.py
import os
import logging
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import keras
import tensorflow as tf

from ml4h.defines import IMAGE_EXT
from ml4h.explorations import predictions_to_pngs
from ml4h.plots import plot_metric_history, plot_roc
from ml4h.metrics import coefficient_of_determination, calculate_kid, calculate_fid
from ml4h.models.model_factory import make_multimodal_multitask_model
from ml4h.models.diffusion_blocks import DiffusionModel, DiffusionController
from ml4h.tensor_generators import test_train_valid_tensor_generators, big_batch_from_minibatch_generator


def train_diffusion_model(args):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    model = DiffusionModel(args.tensor_maps_in[0], args.batch_size, args.dense_blocks, args.block_size, args.conv_x,
                           args.diffusion_loss, args.sigmoid_beta, args.inspect_model)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=1e-4,
        ),
        loss=keras.losses.MeanAbsoluteError() if args.diffusion_loss == 'mean_absolute_error' else keras.losses.MeanSquaredError(),
    )
    batch = next(iter(generate_train))
    for k in batch[0]:
        logging.info(f"input {k} {batch[0][k].shape}")
        feature_batch = batch[0][k]
    for k in batch[1]:
        logging.info(f"label {k} {batch[1][k].shape}")
    checkpoint_path = f"{args.output_folder}{args.id}/{args.id}.weights.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_i_loss",
        mode="min",
        save_best_only=True,
    )

    callbacks = [checkpoint_callback]

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(feature_batch)
    ##test_preds = model.network.predict(generate_train, steps=1)
    if args.inspect_model:
        model.network.summary(print_fn=logging.info, expand_nested=True)
        tf.keras.utils.plot_model(
            model.network,
            to_file=f"{args.output_folder}/{args.id}/architecture_diffusion_unet.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=args.dpi,
            layer_range=None,
            show_layer_activations=False,
        )
        prefix_value = f'{args.output_folder}{args.id}/learning_generations/'
        if model.tensor_map.axes() == 2:
            plot_partial = partial(model.plot_ecgs, reseed=args.random_seed, prefix=prefix_value)
        else:
            plot_partial = partial(model.plot_images, reseed=args.random_seed, prefix=prefix_value)
        callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=plot_partial))

    batch = next(iter(generate_train))
    images = batch[0][model.tensor_map.input_name()]
    # (2) create a dummy noise_rates tensor of the right shape
    noise_rates = tf.zeros(
        [model.batch_size] + [1] * model.tensor_map.axes(),
        dtype=images.dtype,
    )
    # (3) adapt your normalizer, if you hadn’t already
    model.normalizer.adapt(images)
    # (4) call the model once
    _ = model((images, noise_rates))
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        logging.info(f'Loaded weights from model checkpoint at: {checkpoint_path}')
    else:
        logging.info(f'No checkpoint at: {checkpoint_path}')
    history = model.fit(
        generate_train,
        steps_per_epoch=args.training_steps,
        epochs=args.epochs,
        validation_data=generate_valid,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
    )
    model.load_weights(checkpoint_path)
    plot_metric_history(history, args.training_steps, args.id, os.path.dirname(checkpoint_path))
    if args.inspect_model:
        metrics = model.evaluate(generate_test, batch_size=args.batch_size, steps=args.test_steps, return_dict=True)
        logging.info(f'Test metrics: {metrics}')

        steps = 1 if args.batch_size > 3 else args.test_steps
        data, labels, paths = big_batch_from_minibatch_generator(generate_test, steps)
        sides = int(np.sqrt(steps*args.batch_size))
        preds = model.plot_reconstructions((data, labels), num_rows=sides, num_cols=sides,
                                           prefix=f'{args.output_folder}/{args.id}/reconstructions/')
        image_out = {args.tensor_maps_in[0].output_name(): data[args.tensor_maps_in[0].input_name()]}
        predictions_to_pngs(preds, args.tensor_maps_in, args.tensor_maps_in, data, image_out, paths, f'{args.output_folder}/{args.id}/reconstructions/')
        if model.tensor_map.axes() == 2:
            model.plot_ecgs(num_rows=4, prefix=os.path.dirname(checkpoint_path))
        else:
            model.plot_images(num_cols=sides, num_rows=sides, prefix=os.path.dirname(checkpoint_path))
    return model


def get_eval_model(args, model_file, output_tmap):
    args.tensor_maps_out = [output_tmap]
    eval_model, _, _, _ = make_multimodal_multitask_model(**args.__dict__)
    return eval_model


def regress_on_batch(diffuser, regressor, controls, tm_out, batch_size):
    control_batch = {tm_out.output_name(): controls}

    control_embed = diffuser.control_embed_model(control_batch)
    generated_images = diffuser.generate(
        control_embed,
        num_images=batch_size,
        diffusion_steps=50,
    )
    control_predictions = regressor.predict(generated_images, verbose=0)
    return control_predictions[:, 0] if tm_out.is_continuous() else control_predictions


def regress_on_controlled_generations(diffuser, regressor, tm_out, batches, batch_size, mean, std, prefix):
    preds = []
    all_controls = []
    # controls = np.arange(-8, 8, 1)

    for i in range(batches):
        if tm_out.is_continuous():
            controls = np.random.normal(mean, std, size=batch_size)
        elif tm_out.is_categorical():
            controls = np.eye(tm_out.shape[-1])[np.random.choice(tm_out.shape[-1], batch_size)]
        preds.extend(regress_on_batch(diffuser, regressor, controls, tm_out, batch_size))
        all_controls.extend(controls)
        if i % 4 == 0:
            logging.info(f'Inferred on {i+1} synthetic diffusion batches of {batches}')

    preds = np.array(preds)
    all_controls = np.array(all_controls)
    logging.info(f'Control Predictions was {preds.shape} Control true was {all_controls.shape}')
    if tm_out.is_continuous():
        preds = preds.flatten()
        all_controls = all_controls.flatten()
        pearson = np.corrcoef(preds, all_controls)[1, 0]
        logging.info(f'Pearson correlation {pearson:0.3f} ')
        plt.scatter(preds, all_controls)
        plt.xlabel(f'Predicted {tm_out.name}')
        plt.ylabel(f'Control {tm_out.name}')
        plt.title(f'''Diffusion Phenotype: {tm_out.name} Control vs Predictions
        Pearson correlation {pearson:0.3f}, $R^2$ {coefficient_of_determination(preds, all_controls):0.3f}, N = {len(preds)}''')
        now_string = datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'scatter_{tm_out.name}_r_{pearson:0.3f}_{now_string}{IMAGE_EXT}')
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        plt.savefig(figure_path)
        plt.close()
    elif tm_out.is_categorical():
        plot_roc(preds, all_controls, tm_out.channel_map,
                 f'Diffusion Phenotype: {tm_out.name} Control vs Predictions', prefix)


def interpolate_controlled_generations(diffuser, tensor_maps_out, control_tm, batch_size, prefix):
    control_batch = {}
    if control_tm.is_continuous():
        samples = np.arange(-1, 1, 0.4)
    elif control_tm.is_categorical():
        samples = np.arange(0, len(control_tm.channel_map), 1)
    num_rows = len(samples)
    num_cols = 4
    plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
    for row, pheno_scale in enumerate(samples):
        for cm in tensor_maps_out:
            if cm == control_tm and control_tm.is_continuous():
                control_batch[cm.output_name()] = np.ones((batch_size,) + cm.shape) * pheno_scale
            elif cm == control_tm and control_tm.is_categorical():
                control_batch[cm.output_name()] = np.zeros((batch_size,) + cm.shape)
                control_batch[cm.output_name()][:, pheno_scale] = 1.0
            else:
                control_batch[cm.output_name()] = np.zeros((batch_size,) + cm.shape)

        control_embed = diffuser.control_embed_model(control_batch)
        generated_images = diffuser.generate(
            control_embed,
            num_images=batch_size,
            diffusion_steps=50,
            reseed=12345,  # hold everything constant except for control signal
        )

        for i_col, col in enumerate(range(num_cols)):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            if len(generated_images.shape) == 3:
                for lead in range(generated_images.shape[-1]):
                    plt.plot(generated_images[i_col, :, lead], label=lead)
            elif len(generated_images.shape) == 4:
                plt.imshow(generated_images[i_col], cmap='gray')
            plt.gca().set_title(f'{control_tm.name[:14]}: {pheno_scale:0.1f}')
            plt.axis("off")

    plt.tight_layout()
    now_string = datetime.now().strftime('%Y-%m-%d_%H-%M')
    figure_path = os.path.join(prefix, f'interpolate_synthetic_{control_tm.name}_{now_string}{IMAGE_EXT}')
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path)
    plt.close()


def train_diffusion_control_model(args, supervised=False):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    if supervised:
        supervised_model, _, _, _ = make_multimodal_multitask_model(**args.__dict__)
        model = DiffusionController(
            args.tensor_maps_in[0], args.tensor_maps_out, args.batch_size, args.dense_blocks, args.block_size, args.conv_x,
            args.dense_layers[0], args.attention_window, args.attention_heads, args.attention_modulo, args.diffusion_loss,
            args.sigmoid_beta, args.diffusion_condition_strategy, args.inspect_model, supervised_model, args.supervision_scalar,
        )
    else:
        model = DiffusionController(
            args.tensor_maps_in[0], args.tensor_maps_out, args.batch_size, args.dense_blocks, args.block_size, args.conv_x,
            args.dense_layers[0], args.attention_window, args.attention_heads, args.attention_modulo, args.diffusion_loss,
            args.sigmoid_beta, args.diffusion_condition_strategy, args.inspect_model,
        )

    loss = keras.losses.MeanAbsoluteError() if args.diffusion_loss == 'mean_absolute_error' else keras.losses.MeanSquaredError()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=1e-4,
        ),
        loss=loss,
    )
    batch = next(generate_train)
    for k in batch[0]:
        logging.info(f"input {k} {batch[0][k].shape}")
        feature_batch = batch[0][k]
    for k in batch[1]:
        logging.info(f"label {k} {batch[1][k].shape}")
    checkpoint_path = f"{args.output_folder}{args.id}/{args.id}"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_i_loss",
        mode="min",
        save_best_only=True,
    )
    callbacks = [checkpoint_callback]

    # calculate mean and variance of training dataset for normalization
    model.normalizer.adapt(feature_batch)
    if args.inspect_model:
        model.network.summary(print_fn=logging.info, expand_nested=True)
        tf.keras.utils.plot_model(
            model.network,
            to_file=f"{args.output_folder}/{args.id}/architecture_{args.id}_unet.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=args.dpi,
            layer_range=None,
            show_layer_activations=False,
        )
        tf.keras.utils.plot_model(
            model.control_embed_model,
            to_file=f"{args.output_folder}/{args.id}/architecture_{args.id}_control_embed.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=args.dpi,
            layer_range=None,
            show_layer_activations=True,
        )
        prefix_value = f'{args.output_folder}{args.id}/learning_generations/'
        # Create a partial function with reseed and prefix pre-filled
        if model.input_map.axes() == 2:
            plot_partial = partial(model.plot_ecgs, reseed=args.random_seed, prefix=prefix_value)
        else:
            plot_partial = partial(model.plot_images, reseed=args.random_seed, prefix=prefix_value)
        callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=plot_partial))

    if os.path.exists(checkpoint_path+'.index'):
        model.load_weights(checkpoint_path)
        logging.info(f'Loaded weights from model checkpoint at: {checkpoint_path}')
    else:
        logging.info(f'No checkpoint at: {checkpoint_path}')

    history = model.fit(
        generate_train,
        steps_per_epoch=args.training_steps,
        epochs=args.epochs,
        validation_data=generate_valid,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
    )
    plot_metric_history(history, args.training_steps, args.id, os.path.dirname(checkpoint_path))
    model.load_weights(checkpoint_path)

    if args.inspect_model:
        metrics = model.evaluate(generate_test, batch_size=args.batch_size, steps=args.test_steps, return_dict=True)
        logging.info(f'Test metrics: {metrics}')

        steps = 1 if args.batch_size > 3 else args.test_steps
        data, labels, paths = big_batch_from_minibatch_generator(generate_test, steps)
        sides = int(np.sqrt(steps*args.batch_size))
        preds = model.plot_reconstructions((data, labels), num_rows=sides, num_cols=sides,
                                           prefix=f'{args.output_folder}/{args.id}/reconstructions/')

        image_out = {args.tensor_maps_in[0].output_name(): data[args.tensor_maps_in[0].input_name()]}
        predictions_to_pngs(preds, args.tensor_maps_in, args.tensor_maps_in, data, image_out, paths, f'{args.output_folder}/{args.id}/reconstructions/')
        interpolate_controlled_generations(model, args.tensor_maps_out, args.tensor_maps_out[0], args.batch_size,
                                           f'{args.output_folder}/{args.id}/')
        if model.input_map.axes() == 2:
            model.plot_ecgs(num_rows=2, prefix=os.path.dirname(checkpoint_path))
        else:
            model.plot_images(num_cols=sides, num_rows=sides, prefix=os.path.dirname(checkpoint_path))

        for tm_out, model_file in zip(args.tensor_maps_out, args.model_files):
            args.tensor_maps_out = [tm_out]
            args.model_file = model_file
            eval_model, _, _, _ = make_multimodal_multitask_model(**args.__dict__)
            regress_on_controlled_generations(model, eval_model, tm_out, args.test_steps, args.batch_size,
                                              0.0,2.0,f'{args.output_folder}/{args.id}/')


    return model


def test_diffusion_control_model(args, unconditioned=False, supervised=False):
    _, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    if unconditioned:
        model = DiffusionModel(args.tensor_maps_in[0], args.batch_size, args.dense_blocks, args.block_size, args.conv_x,
                               args.diffusion_loss, args.sigmoid_beta, args.inspect_model)
    elif supervised:
        supervised_model, _, _, _ = make_multimodal_multitask_model(**args.__dict__)
        model = DiffusionController(
            args.tensor_maps_in[0], args.tensor_maps_out, args.batch_size, args.dense_blocks, args.block_size, args.conv_x,
            args.dense_layers[0], args.attention_window, args.attention_heads, args.attention_modulo, args.diffusion_loss,
            args.sigmoid_beta, args.diffusion_condition_strategy, args.inspect_model, supervised_model, args.supervision_scalar,
        )
    else:
        model = DiffusionController(
            args.tensor_maps_in[0], args.tensor_maps_out, args.batch_size, args.dense_blocks, args.block_size, args.conv_x,
            args.dense_layers[0], args.attention_window, args.attention_heads, args.attention_modulo, args.diffusion_loss,
            args.sigmoid_beta, args.diffusion_condition_strategy, args.inspect_model,
        )

    loss = keras.losses.MeanAbsoluteError() if args.diffusion_loss == 'mean_absolute_error' else keras.losses.MeanSquaredError()
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=args.learning_rate, weight_decay=1e-4), loss=loss)
    checkpoint_path = f"{args.output_folder}{args.id}/{args.id}"
    if os.path.exists(checkpoint_path+'.index'):
        model.load_weights(checkpoint_path)
        logging.info(f'Loaded weights from model checkpoint at: {checkpoint_path}')
    else:
        logging.warning(f'Testing a random model?! No checkpoint at: {checkpoint_path}')

    kid_values = []
    fid_values = []
    for i in range(args.test_steps):
        batch = next(generate_test)
        for k in batch[0]:
            feature_batch = batch[0][k]
        model.normalizer.adapt(feature_batch)
        if unconditioned:
            generated_images = model.generate(num_images=args.batch_size, diffusion_steps=50)
        else:
            control_embed = model.control_embed_model(batch[1])
            generated_images = model.generate(control_embed, num_images=args.batch_size, diffusion_steps=50)
        kid_values.append(calculate_kid(feature_batch, generated_images))
        fid_values.append(calculate_fid(feature_batch, generated_images))
    logging.info(f"\nKID:{np.mean(kid_values):.4f} ± {np.std(kid_values, ddof=1):.4f} \nFID:{np.mean(fid_values):.4f} ± {np.std(fid_values, ddof=1):.4f}")