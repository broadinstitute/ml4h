import numpy as np
import tensorflow as tf

# from official.common import flags as tfm_flags
from official.vision.beta.projects.movinet.modeling import movinet, movinet_model

from droid_callbacks import MetricsHistoryCallback, SlackNotifierCallback, run_validation_inference

learning_rate = 0.0001
hidden_units = 256
dropout_rate = 0.5
temperature = 0.05


class DDGenerator:
    def __init__(self, input_dd, output_dd, fill_empty=False):
        self.input_dd = input_dd
        self.output_dd = output_dd
        self.fill_empty = fill_empty

    def __call__(self, sample_ids):
        ret_input = []
        ret_output = []
        for sample_id in sample_ids:
            ret_input.append(
                self.input_dd.get_raw_data(sample_id),
            )
            if self.output_dd is not None:
                ret_output.append(
                    self.output_dd.get_raw_data(sample_id),
                )
            if self.fill_empty:
                ret_output.append(np.NaN)

        if self.output_dd is not None and isinstance(ret_output[0], list):
            ret_output = [np.vstack([ret_output[i][j] for i in range(len(sample_ids))])
                          for j in range(len(ret_output[0]))]
            ret_output = tuple(ret_output)

        if self.output_dd is None and self.fill_empty == False:
            yielded = (ret_input,)
        else:
            yielded = (ret_input, ret_output)
        yield yielded


def create_movinet_classifier(
        n_input_frames,
        batch_size,
        checkpoint_dir,
        num_classes,
        freeze_backbone=False,
):
    backbone = movinet.Movinet(model_id='a2')
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
    model.build([1, 1, 1, 1, 3])
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()

    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes,
    )
    model.build([batch_size, n_input_frames, 224, 224, 3])

    if freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True

    return model, backbone


def create_regressor(encoder, trainable=True, input_shape=(224, 224, 3), n_output_features=1):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape, name='image')
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    outputs = tf.keras.layers.Dense(n_output_features, activation=None, name='echolab')(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="regressor")

    return model


# ---------- Adaptation for regression + classification + survival ---------- #
def create_regressor_classifier(encoder, trainable=True, input_shape=(224, 224, 3), n_output_features=0, categories={},
                                category_order=None, survival_heads=None,
                                add_dense={'regressor': False, 'classifier': False}):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape, name='image')
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)

    outputs = []
    if n_output_features > 0:
        if add_dense['regressor']:
            features_reg = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
            features_reg = tf.keras.layers.Dropout(dropout_rate)(features_reg)
            outputs.append(tf.keras.layers.Dense(n_output_features, activation=None, name='echolab')(features_reg))
        else:
            outputs.append(tf.keras.layers.Dense(n_output_features, activation=None, name='echolab')(features))
    if len(categories.keys()) > 0:
        if add_dense['classifier']:
            features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
            features = tf.keras.layers.Dropout(dropout_rate)(features)
        for category in category_order:
            # added a variable - category_order to make sure the ordering is correct
            # (dictionary items ordering is not necessarily consistent)
            activation = 'softmax'
            n_classes = categories[category]
            outputs.append(tf.keras.layers.Dense(n_classes, name='cls_'+category, activation=activation)(features))

    for task_name, intervals in (survival_heads or {}).items():
        outputs.append(
            tf.keras.layers.Dense(
                intervals,
                name=f'survival_{task_name}',
                activation='sigmoid',
            )(features),
        )

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="regressor_classifier")

    return model
# ---------------------------------------------------------------- #


def train_model(
        model,
        train_loader,
        valid_loader,
        epochs,
        n_train_steps,
        n_valid_steps,
        output_folder,
        es_flags,
        class_weight=None,
        batch_size=None,
        valid_ids=None,
        output_labels=None,
        output_reg_len=0,
        cls_category_map_dicts=None,
        survival_tasks=None,
        run_summary=None,
        run_validation_inference_flag=False,
):
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f'{output_folder}/logs',
        # Log weight histograms/distributions every epoch.
        histogram_freq=1,
        # Scalar (loss/metric) summaries are written once per epoch.
        update_freq='epoch',
        # Model graph for the Graphs tab.
        write_graph=True,
        # Throughput scalar, complements the profiler's utilization traces.
        write_steps_per_second=True,
        # Profiler window that captures GPU/CPU/memory/VRAM utilization (Profile tab).
        profile_batch=[20, 30],
    )
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=es_flags['es_loss2monitor'],
                                                   patience=es_flags['es_patience'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{output_folder}/model/chkp',
        monitor=es_flags['es_loss2monitor'],
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
    metrics_history_callback = MetricsHistoryCallback(output_folder)
    slack_callback = SlackNotifierCallback(output_folder, run_summary=run_summary)

    try:
        model.fit(
            train_loader,
            validation_data=valid_loader,
            callbacks=[tb_callback, es_callback, cp_callback, metrics_history_callback, slack_callback],
            epochs=epochs,
            steps_per_epoch=n_train_steps,
            validation_steps=n_valid_steps,
            workers=1,
            max_queue_size=1,
            use_multiprocessing=False,
            class_weight=class_weight
        )
    except Exception as exc:
        slack_callback.notify_failure(exc)
        raise

    model.load_weights(
        f'{output_folder}/model/chkp'
    )

    validation_summary = None
    if run_validation_inference_flag and batch_size and valid_ids and n_valid_steps > 0:
        validation_summary = run_validation_inference(
            model=model,
            valid_dataset=valid_loader,
            n_valid_steps=n_valid_steps,
            batch_size=batch_size,
            valid_ids=valid_ids,
            output_folder=output_folder,
            output_labels=output_labels or [],
            output_reg_len=output_reg_len,
            cls_category_map_dicts=cls_category_map_dicts,
            survival_tasks=survival_tasks,
        )

    slack_callback.notify_success(validation_summary)

    return model
