import time
import json
from tqdm import tqdm
import logging
import numpy as np
import tracemalloc
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import os
import io
import av
import lmdb
import psutil
import random

# from official.common import flags as tfm_flags
from official.vision.beta.projects.movinet.modeling import movinet, movinet_model

learning_rate = 0.0001
hidden_units = 256
dropout_rate = 0.5
temperature = 0.05
# num_classes = len(category_dictionary)
# num_classes = 10

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def create_encoder(weights=None, input_shape=(224, 224, 3)):
    resnet = tf.keras.applications.ResNet50V2(
        include_top=False, weights=weights, input_shape=input_shape, pooling="avg"
    )

    inputs = tf.keras.Input(shape=input_shape, name='echos')
    outputs = resnet(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet_pretrained")
    return model


def create_video_encoder(
    path='https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3',
    input_shape=(40, 224, 224, 3),
    trainable=True
    ):
    
    inputs = tf.keras.layers.Input(
        shape=input_shape,
        dtype=tf.float32,
        name='image'
    )
    
    movinet = hub.KerasLayer(path, trainable=trainable)
    outputs = movinet({
        'image': inputs
    })

    model = tf.keras.Model(inputs, outputs, name='movinet')
    
    return model


def create_movinet_classifier(
    n_input_frames,
    batch_size,
    checkpoint_dir,
    num_classes,
    freeze_backbone=False
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
        num_classes=num_classes
    )
    model.build([batch_size, n_input_frames, 224, 224, 3])

    if freeze_backbone:
        for layer in model.layers[:-1]:
            layer.trainable = False
        model.layers[-1].trainable = True

    return model, backbone


def create_video_model(
    path='https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3',
    input_shape = (40, 224, 224, 3),
    batch_size = 16,
    n_output_labels = 1,
    trainable_layers_if_finetune = None,
    model_weights_folder = None,
    encoder = False,
    include_top = True
):

    if model_weights_folder is None:
        backbone = movinet.Movinet(model_id='a2')
        model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
        model.build([batch_size] + list(input_shape))
        model_weights = {w.name: w for w in model.weights}

        movinet_hub_model = hub.KerasLayer(path, trainable=True)
        pretrained_weights = {w.name: w for w in movinet_hub_model.weights}
        for name in pretrained_weights:
            model_weights[name].assign(pretrained_weights[name])
    else:
        with open(f'{model_weights_folder}/model_params.json', 'r') as json_file:
            original_model_params = json.load(json_file)
        
        backbone = movinet.Movinet(model_id='a2')
        num_classes = n_output_labels if n_output_labels else 2
        if 'output_labels' in original_model_params:
            num_classes = 2 if ('bottom3_discrepancy' in original_model_params['output_labels']) else len(original_model_params['output_labels'])
        model = movinet_model.MovinetClassifier(
            backbone=backbone, 
            num_classes=num_classes
            )
        model.build([batch_size] + list(input_shape))
        
        model.load_weights(
            f'{model_weights_folder}/model/chkp'
        )

    if not include_top:
        return backbone

    # Wrap the backbone with a new classifier to create a new classifier head
    # with num_classes outputs
    if n_output_labels is not None:
        model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=n_output_labels)
        model.build([batch_size] + list(input_shape))    

    # If no layers given, assume entire model will be trained
    if trainable_layers_if_finetune is not None and len(trainable_layers_if_finetune) != 0:
        # Freeze all layers except for the specified layers
        for layer_idx in range(len(model.layers)):
            layer = model.layers[layer_idx]
            if layer_idx in trainable_layers_if_finetune:
                layer.trainable = True
            else:
                layer.trainable = False

    return model

def create_classifier(encoder, trainable=True, input_shape=(224, 224, 3), categories={'output': 10}):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape, name='image')
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    outputs = []
    for category, n_classes in categories.items():
        activation = 'softmax' if n_classes > 1 else 'sigmoid'
        outputs.append(tf.keras.layers.Dense(n_classes, name=category, activation=activation)(features))
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")
    return model


def create_regressor(encoder, trainable=True, input_shape=(224, 224, 3), n_output_features=1):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape, name='image')
    features = encoder(inputs)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    outputs = tf.keras.layers.Dense(n_output_features, activation=None, name='echolab_sex')(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="regressor")        
    
    return model


def train_model(
    model, 
    train_loader, 
    valid_loader,
    epochs,
    n_train_steps,
    n_valid_steps,
    output_folder,
    class_weight = None
    ):
    
    tb_callback = tf.keras.callbacks.TensorBoard(f'{output_folder}/logs', profile_batch=[160, 170])
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = f'{output_folder}/model/chkp',
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = True,
        mode = 'min'
    )
    model.fit(
        train_loader,
        validation_data = valid_loader,
        callbacks = [tb_callback, es_callback, cp_callback],
        epochs = epochs,
        steps_per_epoch = n_train_steps,
        validation_steps = n_valid_steps,
        workers = 1,
        max_queue_size = 1,
        use_multiprocessing = False,
        class_weight = class_weight
    )

    model.load_weights(
        f'{output_folder}/model/chkp'
    )       

    return model


def train_step_custom(x_batch_train, y_batch_train, model, loss_function, optimizer, train_metrics):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss_value = loss_function(y_batch_train, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    for train_metric in train_metrics:
        train_metric.update_state(y_batch_train, logits)
    return loss_value


def valid_step_custom(x_valid_train, y_valid_train, model, loss_function, valid_metrics):
    logits = model(x_valid_train, training=False)
    loss_value = loss_function(y_valid_train, logits)

    for valid_metric in valid_metrics:
        valid_metric.update_state(y_valid_train, logits)

    return loss_value
    


def train_model_custom(
    model,
    train_loader, 
    valid_loader,
    epochs,
    n_train_steps,
    n_valid_steps,
    output_folder
    ):
    
    tb_callback = tf.keras.callbacks.TensorBoard(f'{output_folder}/logs')

    callbacks = tf.keras.callbacks.CallbackList(
        [tb_callback], add_history=True, model=model
    )

    logs = {}
    callbacks.on_train_begin(logs=logs)

    best_valid_loss_tot = np.inf
    patience_waited = 0

    for epoch in range(epochs):
        logging.info(f'Start of epoch {epoch}')
        callbacks.on_epoch_begin(epoch, logs=logs)
        cur_time = time.time()
        old_time = cur_time

        for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_loader, total=n_train_steps)):

            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_train_batch_begin(step, logs=logs)

            # loss_value = train_step_custom(x_batch_train, y_batch_train, model, loss_function, optimizer, train_metrics)
            logs = model.train_on_batch(x_batch_train, y_batch_train, return_dict=True, reset_metrics=False)

            callbacks.on_train_batch_end(step, logs=logs)
            callbacks.on_batch_end(step, logs=logs)

            if step % 5 == 0:
                cur_time = time.time()
                p = psutil.Process(os.getpid())
                logging.info(
                    f'Step {step}: loss {logs["loss"]:.4f}, pearson: {logs["pearson"]:.4f}, mem: { p.memory_info().rss / 1024 / 1024}, time: {cur_time - old_time}'
                )
                old_time = cur_time

        logging.info(f'Epoch {epoch}: loss {logs["loss"]:.4f}, pearson: {logs["pearson"]:.4f}')
        model.reset_metrics()

        # valid_loss_tot = 0.0
        for step, (x_batch_valid, y_batch_valid) in enumerate(tqdm(valid_loader, total=n_valid_steps)):
            callbacks.on_batch_begin(step, logs=logs)
            callbacks.on_test_batch_begin(step, logs=logs)

            # loss_value = valid_step_custom(x_batch_valid, y_batch_valid, model, loss_function, valid_metrics)
            logs = model.test_on_batch(x_batch_valid, y_batch_valid, return_dict=True, reset_metrics=False)

            callbacks.on_test_batch_end(step, logs=logs)
            callbacks.on_batch_end(step, logs=logs)

        logging.info(f'Epoch {epoch}: val_loss {logs["loss"]:.4f}, val_pearson: {logs["pearson"]:.4f}')

        if logs["loss"] < best_valid_loss_tot:
            best_valid_loss_tot = logs["loss"]
            model.save_weights(
                f'{output_folder}/model/chkp'
            )
            patience_waited = 0
        else:
            patience_waited += 1

        callbacks.on_epoch_end(epoch, logs=logs)

        if patience_waited == 10:
            break

    callbacks.on_train_end(logs=logs)
    model.load_weights(
        f'{output_folder}/model/chkp'
    )
    
    return model

class TripletLoss(tf.keras.losses.Loss):
    def __init__(self, name=None):
        super(TripletLoss, self).__init__(name=name)
    
    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        return tfa.losses.triplet_semihard_loss(tf.squeeze(labels), feature_vectors_normalized, distance_metric="L2")

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None, reduction=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class CorpLoss(tf.keras.losses.Loss):
    def __init__(self, name=None):
        super(CorpLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)

        feature_vectors1 = feature_vectors_normalized[0::2]
        feature_vectors2 = feature_vectors_normalized[1::2]

        feature_vectors = tf.stack([feature_vectors1, feature_vectors2], axis=1)
        
        import pdb; pdb.set_trace()
        loss = tf.keras.losses.categorical_crossentropy(
            labels, feature_vectors, from_logits=True
        )
        
        return loss


class SupervisedContrastiveLossNotfa(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None, reduction=None, simclr=True):
        super(SupervisedContrastiveLossNotfa, self).__init__(name=name)
        self.temperature = temperature
        self.simclr = simclr

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        batch_size = tf.shape(feature_vectors)[0]
        feature_vectors1 = feature_vectors_normalized[:(batch_size // 2)]
        feature_vectors2 = feature_vectors_normalized[(batch_size // 2):]
        contrastive_labels = tf.range(batch_size // 2)
        similarities = (
            tf.matmul(feature_vectors1, feature_vectors2, transpose_b=True)
        )
        if self.simclr:
            similarities_temp = similarities / self.temperature

            # Compute logits
            loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
                contrastive_labels, similarities_temp, from_logits=True
            )
            loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
                contrastive_labels, tf.transpose(similarities_temp), from_logits=True
            )
            return (loss_1_2 + loss_2_1) / 2
        else:
            SMALL_NUM = np.log(1e-45)
            positive_loss = -tf.linalg.diag_part(similarities) / self.temperature
            neg_similarity = tf.concat((tf.matmul(feature_vectors1, feature_vectors1, transpose_b=True), similarities), axis=1) / self.temperature
            neg_mask = tf.tile(tf.eye(batch_size // 2), [1, 2])
            negative_loss = tf.reduce_logsumexp(neg_similarity + neg_mask * SMALL_NUM, axis=1, keepdims=False)
            return (positive_loss + negative_loss) / 2



# Define the contrastive model with model-subclassing
class ContrastiveModel(tf.keras.Model):
    def __init__(self, temperature, encoder, feature_width=2048, projection_width=128):
        super().__init__()

        self.temperature = temperature
        self.encoder = encoder
        # Non-linear MLP as projection head
        self.projection_head = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(feature_width,)),
                tf.keras.layers.Dense(projection_width, activation="relu"),
                tf.keras.layers.Dense(projection_width),
            ],
            name="projection_head",
        )
        # Single dense layer for linear probing
        self.linear_probe = tf.keras.Sequential(
            [tf.layers.Input(shape=(feature_width,)), tf.keras.layers.Dense(1)], name="linear_probe"
        )

        self.encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = tf.keras.losses.logcosh()

        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = tf.keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        # Both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # Labels are only used in evalutation for an on-the-fly logistic regression
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=True
        )
        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            features = self.encoder(preprocessed_images, training=False)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_images, labels = data

        # For testing the components are used with a training=False flag
        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}
        


def add_projection_head(encoder, input_shape=(224, 224, 3), projection_units = 128, name='contrastive_study'):
    inputs = tf.keras.Input(shape=input_shape, name='echos')
    features = encoder(inputs)
    outputs = tf.keras.layers.Dense(projection_units, activation="relu", name=name)(features)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
    )
    return model


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
                self.input_dd.get_raw_data(sample_id)
            )
            if self.output_dd is not None:
                ret_output.append(
                    self.output_dd.get_raw_data(sample_id)                
                )
            if self.fill_empty:
                ret_output.append(np.NaN)
        
        if self.output_dd is None and self.fill_empty == False:
            yielded = (ret_input,)
        else:
            yielded = (ret_input, ret_output)
        yield yielded

class DDGeneratorStart:
    def __init__(self, input_dd, output_dd, fill_empty=False, loading_start=0, wide_df=None):
        self.input_dd = input_dd
        self.output_dd = output_dd
        self.fill_empty = fill_empty
        self.loading_start = loading_start
        self.wide_df = wide_df

    def __call__(self, sample_ids):
        ret_input = []
        ret_output = []
        for sample_id in sample_ids:
            ret_input.append(
                self.input_dd.get_raw_data(sample_id, loading_option={'start': self.loading_start})
            )
            if self.output_dd is not None:
                ret_output.append(
                    self.output_dd.get_raw_data(sample_id)                
                )
            if self.fill_empty:
                ret_output.append(np.NaN)
        
        if self.output_dd is None and self.fill_empty == False:
            yielded = (ret_input,)
        else:
            yielded = (ret_input, ret_output)
        yield yielded



class DDGeneratorDict:
    def __init__(self, input_dd, output_dd_dict, fill_empty=False):
        self.input_dd = input_dd
        self.output_dd_dict = output_dd_dict
        self.fill_empty = fill_empty

    def __call__(self, sample_ids):
        ret_input = []
        ret_output = {k: [] for k in self.output_dd_dict}
        for sample_id in sample_ids:
            ret_input.append(
                self.input_dd.get_raw_data(sample_id)
            )
            for k, output_dd in self.output_dd_dict.items():       
                ret_output[k].append(
                    output_dd.get_raw_data(sample_id)                
                )
        yield (ret_input, ret_output)



class StudyGenerator:     

    def __init__(self, df, selected_views_idx, stitch_view):
        if not stitch_view:
            self.study_indexed_dic = df.groupby('study')['sample_id'].apply(list).to_dict()
        else:
            self.study_indexed_dic = df.groupby(['study', 'view_prediction'])['sample_id'].apply(list).to_dict()
        self.selected_views_idx = selected_views_idx
        self.stitch_view = stitch_view

    def __call__(self, study_ids):
        ret = []        
        for study_id in study_ids:
            if not self.stitch_view:
                ret.append(random.sample(self.study_indexed_dic[study_id], 1)[0])
            else:
                ret.append([])
                for selected_view in self.selected_views_idx:
                    ret[-1].append(random.sample(self.study_indexed_dic[(study_id, selected_view)], 1)[0])        
        yield ret


class StudyDeepSetsGenerator:

    def __init__(self, df, n_samples, input_dd, output_dd):
        self.input_dd = input_dd
        self.output_dd = output_dd
        self.study_indexed_dic = df.groupby('study')['sample_id'].apply(list).to_dict()
        self.n_samples = n_samples

    def __call__(self, study_ids):
        ret_input = {
            f'in_{i}': [] for i in range(self.n_samples)
        }
        ret_output = []       
        for study_id in study_ids:
            samples = random.choices(self.study_indexed_dic[study_id], k=self.n_samples)
            for sample, key in zip(samples, ret_input):
                ret_input[key].append(self.input_dd.get_raw_data(sample))
            ret_output.append(self.output_dd.get_raw_data(sample))
            
        yield ret_input, ret_output
        

class StudyBatchGenerator:
    def __init__(self, df, ds_ids, ds_studies, input_dd, output_dd, sample_id_output=False, n=2, views=2, learning_type='study'):
        self.ds_ids = ds_ids
        self.input_dd = input_dd
        self.output_dd = output_dd
        self.sample_id_output = sample_id_output
        self.learning_type = learning_type

        studies = range(len(ds_studies))
        study_id_to_study = {study_id: ds_studies.index(study_id) for study_id in ds_studies}
        echo_to_study = [ds_studies.index(int(j.split('_')[1])) for j in ds_ids]
        echo_to_study_np = np.array(echo_to_study, dtype=np.int64)
        study_to_echo_dic = {study: np.where(echo_to_study_np==study)[0].tolist() for study in studies}

        self.n = n 
        self.views = views
        self.study_to_view_echo_dic = {}
        self.views_names = set()

        for study in tqdm(studies):
            study_id = ds_studies[study]
            self.study_to_view_echo_dic[study_id] = {}
            for echo in study_to_echo_dic[study]:
                sample_id = self.ds_ids[echo]
                view = df.loc[df['sample_id'] == sample_id]['view_prediction'].values[0]
                self.views_names.add(view)
                if view in self.study_to_view_echo_dic[study_id]:
                    self.study_to_view_echo_dic[study_id][view].append(sample_id)
                else:
                    self.study_to_view_echo_dic[study_id][view] = [sample_id]
        
        self.views_names = list(self.views_names)
        assert(len(self.views_names) == self.views)  
    
    def __call__(self, study_ids):
        ret_input = []
        ret_output = []

        sampled_echoes = {}
        sampled_echoes_views = {}
        for i, study_id in enumerate(study_ids):
            sampled_echoes[study_id] = []
            sampled_echoes_views[study_id] = []
            assert(len(self.study_to_view_echo_dic[study_id]) == self.views)
            for view_idx, view in enumerate(self.views_names):
                sample_ids = random.sample(self.study_to_view_echo_dic[study_id][view], k=self.n)
                sampled_echoes[study_id].extend(sample_ids)
                sampled_echoes_views[study_id].extend([view_idx]*self.n)
         
        for j in range(self.views*self.n):
            for i, study_id in enumerate(study_ids):
                sample_id = sampled_echoes[study_id][j]
                ret_input.append(
                    self.input_dd.get_raw_data(sample_id, loading_option=self.input_dd.get_loading_options(sample_id)[0])
                )
                if self.sample_id_output:
                    ret_output.append(echo)  
                else:
                    if self.learning_type == 'study':
                        ret_output.append(i)    
                    elif self.learning_type == 'view':
                        # arranged by view, then duplicates of the views
                        view = j//self.n
                        assert(view == sampled_echoes_views[study_id][j])
                        ret_output.append(view)

        yield (ret_input, ret_output)


class OrderGenerator:
    def __init__(
        self,
        local_lmdb_dir,
        nframes,
        wide_df
    ):
        self.local_lmdb_dir = local_lmdb_dir
        self.nframes = nframes
        self.wide_df = wide_df

    def __call__(self, sample_ids):
        ret_input = {
            f'in_{i}': [] for i in range(2)
        }
        ret_output = []
        for sample_id in sample_ids:
            try:
                sample_id = sample_id.decode('UTF-8')
            except (UnicodeDecodeError, AttributeError):
                pass
            cine_rate = int(self.wide_df[self.wide_df['sample_id']==sample_id]['CineRate'].values[0])
            cine_rate_factor = 7 / cine_rate
            _, study, view = sample_id.split('_')
            lmdb_folder = os.path.join(self.local_lmdb_dir, f"{study}.lmdb")

            env = lmdb.open(lmdb_folder, readonly=True, lock=False)
            with env.begin(buffers=True) as txn:
                in_mem_bytes_io = io.BytesIO(txn.get(view.encode('utf-8')))
                video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
                view_frames = video_container.streams.video[0].frames

                cine_frames = []
                last_cine_frame = -1
                for i, frame in enumerate(video_container.decode(video=0)):
                    if int(i * cine_rate_factor) > last_cine_frame:
                        cine_frames.append(frame)
                        last_cine_frame = int(i * cine_rate_factor)

                view_frames = len(cine_frames)
                if view_frames < self.nframes*2:
                    for i in range(self.nframes*2-view_frames):
                        cine_frames.append(frame)
                    view_frames = len(cine_frames)

                offset_0 = 0
                offset_1 = 0

                while (np.abs(offset_1 - offset_0) < self.nframes) or ((max(offset_0, offset_1)+self.nframes) > view_frames):
                    offset_0, offset_1 = random.sample(range(view_frames), k=2)

                frames_0 = []
                frames_1 = []                

                for i, frame in enumerate(cine_frames):
                    frame = np.array(frame.to_image())
                    if (i >= offset_0) and (i < offset_0+self.nframes):
                        frames_0.append(frame)

                    if (i >= offset_1) and (i < offset_1+self.nframes):
                        frames_1.append(frame)
                
                frames_0 = np.squeeze(np.array(frames_0, dtype='float32') / 255.)
                frames_1 = np.squeeze(np.array(frames_1, dtype='float32') / 255.)                
            env.close()
            ret_input['in_0'].append(frames_0)
            ret_input['in_1'].append(frames_1)
            ret_output.append(np.zeros(2, dtype='float32'))
            if offset_0 > offset_1:
                ret_output[-1][0] = 1.0
            else:
                ret_output[-1][1] = 1.0
        yield (ret_input, ret_output)


class PatientBatchGenerator:        
    def __init__(self, sample_ids, patients, input_dd, output_dd, sample_id_output=False):
        self.sample_ids = sample_ids
        self.input_dd = input_dd
        self.output_dd = output_dd
        self.sample_id_output = sample_id_output

        self.patients = range(len(patients))
        self.patient_id_to_patient = {patient_id: patients.index(patient_id) for patient_id in patients}
        echo_to_patients = [patients.index(s.split('_')[0]) for s in sample_ids]
        echo_to_patients_np = np.array(echo_to_patients, dtype=np.int64)
        self.patient_to_echo_dic = {patient: np.where(echo_to_patients_np==patient)[0].tolist() for patient in self.patients}
        # self.patient_study_echo_dic = {}
        # for patient in patients:
        #     self.patient_study_echo_dic[patient] = {}
        #     echoes = self.patient_to_echo_dic[patient]
        #     for echo in echoes:
        #         sample_id = self.ds_ids[echo]
        #         mrn, study, view = sample_id.split('_')
        #         study = int(study)
        #         if study not in self.patient_study_echo_dic[patient]:
        #             self.patient_study_echo_dic[patient][study] = []
        #         self.patient_study_echo_dic[patient][study].append(sample_id) 

    def __call__(self, patient_ids):
        ret_input = []
        ret_output = []

        sampled_echoes = {}
        for i, patient_id in enumerate(patient_ids):
            try:
                patient_id = patient_id.decode('UTF-8')
            except:
                pass
            sampled_echoes[patient_id] = []
            patient = self.patient_id_to_patient[patient_id]
            echoes = random.sample(self.patient_to_echo_dic[patient], k=2)
            sampled_echoes[patient_id].extend(echoes)
            # if len(self.patient_study_echo_dic[patient].keys()) == 1:
            #     echoes = random.sample(self.patient_to_echo_dic[patient], k=2)
            #     sampled_echoes[patient_id].extend(echoes)
            # else:
            #     study_ids = random.sample(list(self.patient_study_echo_dic[patient].keys()), k=2)
            #     for study_id in study_ids:
            #         sampled_echoes[patient_id].append(
            #             random.sample(self.patient_study_echo_dic[patient][study_id], k=1)[0])

        for j in range(2):
            for i, patient_id in enumerate(patient_ids):
                try:
                    patient_id = patient_id.decode('UTF-8')
                except:
                    pass
                echo = sampled_echoes[patient_id][j]
                sample_id = self.sample_ids[echo]
                ret_input.append(
                    self.input_dd.get_raw_data(sample_id)
                )
                if self.sample_id_output:
                    ret_output.append(echo)  
                else:
                    ret_output.append(i)      
        yield (ret_input, ret_output)



class MILAttentionLayer(tf.keras.layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:

            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(instance, self.w_weight_params, axes=1)


def create_attention_regressor(encoder, n_views, input_shape, output_labels, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = [tf.keras.layers.Input(input_shape, name=f'in_{i}') for i in range(n_views)]
    embeddings = [encoder(input) for input in inputs]

    alpha = MILAttentionLayer(
        weight_params_dim=256,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        tf.keras.layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    concat = tf.keras.layers.Add()(multiply_layers)

    # Classification output node.
    output = tf.keras.layers.Dense(512, activation='relu')(concat)
    output = tf.keras.layers.Dense(len(output_labels))(output)
    attention_model = tf.keras.Model(inputs, output)
    return attention_model