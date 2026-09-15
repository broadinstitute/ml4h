"""Modular video backbones for DROID.

Every backbone builder returns a Keras encoder that maps a normalized DROID
video clip of shape ``(T, 224, 224, 3)`` to one vector per clip.  The optional
embedding projection gives downstream DROID heads a stable, configurable
interface even when the native backbone feature width changes.
"""

import logging
import os
import threading

import numpy as np
import tensorflow as tf


MOVINET_A2 = 'movinet_a2'
VJEPA2_1_VIT_BASE_384 = 'vjepa2_1_vit_base_384'

BACKBONE_ALIASES = {
    'movinet': MOVINET_A2,
    'vjepa2_1': VJEPA2_1_VIT_BASE_384,
}


def canonical_backbone_name(name):
    """Return the canonical registry key for a backbone name or alias."""
    name = name or MOVINET_A2
    return BACKBONE_ALIASES.get(name, name)


class FrozenVJEPA2Encoder(tf.keras.layers.Layer):
    """Run a frozen V-JEPA 2.1 PyTorch encoder inside a Keras graph.

    DROID currently trains with legacy TensorFlow Keras, while Meta's official
    V-JEPA 2.1 implementation is PyTorch-only. ``tf.py_function`` is therefore
    used as an explicit frozen-feature boundary. Gradients still train the
    optional Keras embedding projection and all downstream heads, but do not
    cross into the PyTorch backbone.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
            self,
            *,
            checkpoint_path,
            n_input_frames,
            repo,
            device='auto',
            model_name=VJEPA2_1_VIT_BASE_384,
            image_size=384,
            output_dim=768,
            microbatch_size=1,
            **kwargs,
    ):
        super().__init__(trainable=False, **kwargs)
        if not checkpoint_path:
            raise ValueError(
                'V-JEPA 2.1 requires --backbone_checkpoint pointing to the '
                'downloaded Meta .pt checkpoint.',
            )
        if n_input_frames <= 0 or n_input_frames % 2:
            raise ValueError('V-JEPA 2.1 requires a positive, even --n_input_frames value.')
        if microbatch_size <= 0:
            raise ValueError('--backbone_microbatch_size must be a positive integer.')

        self.checkpoint_path = checkpoint_path
        self.n_input_frames = n_input_frames
        self.repo = repo
        self.device = device or 'auto'
        self.model_name = model_name
        self.image_size = image_size
        self.output_dim = output_dim
        self.microbatch_size = microbatch_size
        self._torch = None
        self._torch_model = None
        self._device = None
        self._lock = threading.Lock()

    def get_config(self):
        config = super().get_config()
        config.update({
            'checkpoint_path': self.checkpoint_path,
            'n_input_frames': self.n_input_frames,
            'repo': self.repo,
            'device': self.device,
            'model_name': self.model_name,
            'image_size': self.image_size,
            'output_dim': self.output_dim,
            'microbatch_size': self.microbatch_size,
        })
        return config

    @staticmethod
    def _encoder_state_dict(checkpoint):
        if not isinstance(checkpoint, dict):
            raise ValueError('The V-JEPA checkpoint must contain a state-dict mapping.')

        for key in ('ema_encoder', 'target_encoder', 'encoder'):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                break
        else:
            if checkpoint and all(isinstance(key, str) for key in checkpoint):
                state_dict = checkpoint
            else:
                raise ValueError(
                    'Could not find ema_encoder, target_encoder, or encoder weights '
                    'in the V-JEPA checkpoint.',
                )

        return {
            key.replace('module.', '').replace('backbone.', ''): value
            for key, value in state_dict.items()
        }

    def _load_torch_model(self):
        if self._torch_model is not None:
            return

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                'The V-JEPA 2.1 backbone requires PyTorch, timm, and einops. '
                'Use docker/droid/Dockerfile or install those dependencies.',
            ) from exc

        checkpoint_path = os.path.expanduser(self.checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f'V-JEPA checkpoint not found: {checkpoint_path}')

        repo = os.path.expanduser(self.repo)
        source = 'local' if os.path.isdir(repo) else 'github'
        hub_kwargs = {
            'source': source,
            'pretrained': False,
            'num_frames': self.n_input_frames,
        }
        if source == 'github':
            hub_kwargs['trust_repo'] = True
        model_and_predictor = torch.hub.load(repo, self.model_name, **hub_kwargs)
        model = model_and_predictor[0] if isinstance(model_and_predictor, (tuple, list)) else model_and_predictor

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except TypeError:  # PyTorch versions before weights_only was added.
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(self._encoder_state_dict(checkpoint), strict=True)

        device = self.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)
        model.eval()
        model.requires_grad_(False)
        model_dtype = torch.float16 if self._device.type == 'cuda' else torch.float32
        model.to(device=self._device, dtype=model_dtype)

        self._torch = torch
        self._torch_model = model
        logging.info(
            'Loaded frozen %s backbone from %s on %s.',
            self.model_name,
            checkpoint_path,
            self._device,
        )

    def _torch_forward(self, videos):
        with self._lock:
            self._load_torch_model()
            torch = self._torch
            functional = torch.nn.functional

            videos = np.asarray(videos, dtype=np.float32)
            if videos.ndim != 5 or videos.shape[-1] != 3:
                raise ValueError(
                    'V-JEPA expected video input with shape (batch, frames, height, width, 3); '
                    f'got {videos.shape}.',
                )

            # DROID produces RGB clips in [0, 1]. V-JEPA expects B,C,T,H,W,
            # 384-pixel frames, and ImageNet normalization. The internal
            # microbatch keeps the much larger ViT activation footprint
            # independent of the DROID head-training batch size.
            model_dtype = next(self._torch_model.parameters()).dtype
            mean = torch.tensor(
                self._IMAGENET_MEAN,
                dtype=model_dtype,
                device=self._device,
            ).view(1, 3, 1, 1, 1)
            std = torch.tensor(
                self._IMAGENET_STD,
                dtype=model_dtype,
                device=self._device,
            ).view(1, 3, 1, 1, 1)
            pooled_features = []
            for start in range(0, len(videos), self.microbatch_size):
                x = torch.from_numpy(videos[start:start + self.microbatch_size]).permute(0, 4, 1, 2, 3)
                batch, channels, frames, height, width = x.shape
                x = x.to(device=self._device, dtype=model_dtype)
                x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
                x = functional.interpolate(
                    x,
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False,
                )
                x = x.reshape(batch, frames, channels, self.image_size, self.image_size).permute(0, 2, 1, 3, 4)
                x = (x - mean) / std

                with torch.inference_mode():
                    features = self._torch_model(x)
                if isinstance(features, (tuple, list)):
                    features = features[-1]
                if features.ndim != 3 or features.shape[-1] != self.output_dim:
                    raise ValueError(
                        f'Unexpected {self.model_name} output shape {tuple(features.shape)}; '
                        f'expected (batch, tokens, {self.output_dim}).',
                    )
                pooled_features.append(features.mean(dim=1).float().cpu())

            return torch.cat(pooled_features, dim=0).numpy()

    def call(self, inputs):
        features = tf.py_function(self._torch_forward, [inputs], Tout=tf.float32)
        features.set_shape((None, self.output_dim))
        return features


def create_movinet_classifier(
        n_input_frames,
        batch_size,
        checkpoint_dir,
        num_classes,
        freeze_backbone=False,
):
    """Build the historical MoViNet classifier/backbone pair."""
    try:
        from official.projects.movinet.modeling import movinet, movinet_model
    except ImportError:
        from official.vision.beta.projects.movinet.modeling import movinet, movinet_model

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


def _create_movinet_encoder(*, n_input_frames, batch_size, checkpoint_path, freeze_backbone, **kwargs):
    del kwargs
    _, backbone = create_movinet_classifier(
        n_input_frames,
        batch_size,
        num_classes=600,
        checkpoint_dir=checkpoint_path,
        freeze_backbone=freeze_backbone,
    )
    backbone_output = backbone.layers[-1].output[0]
    flatten = tf.keras.layers.Flatten()(backbone_output)
    # Keep the historical anonymous layer/model names when no projection is
    # requested so existing MoViNet DROID weight checkpoints still match.
    return tf.keras.Model(inputs=[backbone.input], outputs=[flatten])


def _create_vjepa2_1_encoder(
        *,
        n_input_frames,
        checkpoint_path,
        backbone_repo,
        backbone_device,
        backbone_microbatch_size,
        **kwargs,
):
    del kwargs
    inputs = tf.keras.Input(shape=(n_input_frames, 224, 224, 3), name='image')
    features = FrozenVJEPA2Encoder(
        checkpoint_path=checkpoint_path,
        n_input_frames=n_input_frames,
        repo=backbone_repo,
        device=backbone_device,
        microbatch_size=backbone_microbatch_size,
        name='vjepa2_1_vit_base_384',
    )(inputs)
    return tf.keras.Model(inputs=inputs, outputs=features, name='vjepa2_1_vit_base_384_encoder')


BACKBONE_BUILDERS = {
    MOVINET_A2: _create_movinet_encoder,
    VJEPA2_1_VIT_BASE_384: _create_vjepa2_1_encoder,
}


def backbone_choices():
    """Names accepted by the DROID command-line recipes."""
    return tuple(sorted(set(BACKBONE_BUILDERS) | set(BACKBONE_ALIASES)))


def backbone_uses_tensorflow(name):
    """Whether a backbone can run natively under TensorFlow distribution."""
    return canonical_backbone_name(name) == MOVINET_A2


def create_video_encoder(
        *,
        backbone_name,
        n_input_frames,
        batch_size,
        checkpoint_path,
        embedding_dim=None,
        freeze_backbone=False,
        backbone_repo=None,
        backbone_device='auto',
        backbone_microbatch_size=1,
):
    """Create a registered DROID video encoder with optional projection."""
    backbone_name = canonical_backbone_name(backbone_name)
    if backbone_name not in BACKBONE_BUILDERS:
        raise ValueError(
            f'Unknown backbone {backbone_name!r}; choose from {sorted(BACKBONE_BUILDERS)}.',
        )
    if embedding_dim is not None and embedding_dim <= 0:
        raise ValueError('--embedding_dim must be a positive integer.')

    backbone_repo = backbone_repo or os.getenv('VJEPA2_REPO', 'facebookresearch/vjepa2')
    encoder = BACKBONE_BUILDERS[backbone_name](
        n_input_frames=n_input_frames,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        freeze_backbone=freeze_backbone,
        backbone_repo=backbone_repo,
        backbone_device=backbone_device,
        backbone_microbatch_size=backbone_microbatch_size,
    )
    if embedding_dim is None:
        return encoder

    features = tf.keras.layers.Dense(
        embedding_dim,
        name='embedding_projection',
    )(encoder.output)
    return tf.keras.Model(
        inputs=encoder.input,
        outputs=features,
        name=f'{backbone_name}_projected_encoder',
    )
