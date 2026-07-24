"""Video data-augmentation transforms for the echo data descriptions.

Contract
--------
Every transform is a callable with the signature ``transform(video, loading_option=None)``
where ``video`` is a float32 clip of shape ``(T, H, W, C)`` with pixel values in
``[0, 1]`` (i.e. already normalized by ``LmdbEchoStudyVideoDataDescription``). Each
transform returns a clip of the same shape/range.

Transforms operate on the *whole clip* rather than a single frame so that
augmentations which must be consistent across time (jitter, rotation, flips, and
optionally the mask sector) can share the same sampled parameters across frames.

Native TensorFlow ops are used wherever they exist (flips, the projective-transform
op behind Keras' rotation/translation layers, the RNG). Augmentations without a
native op (rectangular masking, per-channel brightness/contrast, per-frame noise)
are written with elementwise TF ops.

These run eagerly inside ``get_raw_data`` (the ml4ht generator path), so scalar
Python control flow on tensors is fine.
"""

import numpy as np
import tensorflow as tf

__all__ = [
    'VideoTransform',
    'RandomJitterRotate',
    'RandomSectorMask',
    'RandomFlip',
    'RandomGaussianNoise',
    'RandomBrightnessContrast',
    'AUGMENTATIONS',
]


class VideoTransform:
    """Base class: converts input to a float32 tensor and applies ``self.p`` gating.

    Parameters
    ----------
    p : float
        Probability that the transform is applied at all. With probability
        ``1 - p`` the clip is returned unchanged (identity). Defaults to 1.0.
    """

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, video, loading_option=None):
        video = tf.convert_to_tensor(video, dtype=tf.float32)
        if self.p < 1.0 and tf.random.uniform(()) > self.p:
            return video
        return self._apply(video, loading_option)

    def _apply(self, video, loading_option=None):
        raise NotImplementedError


class RandomJitterRotate(VideoTransform):
    """Shift and rotate every frame by the same randomly sampled amount.

    A single translation ``(tx, ty)`` (up to ``max_shift`` of the frame width/height)
    and rotation angle (uniform in ``[-max_angle_deg, +max_angle_deg]``) are sampled
    once and applied identically to all frames, simulating probe/camera motion that
    is coherent across the clip. Uncovered pixels are filled with 0 (black).

    Implemented with ``tf.raw_ops.ImageProjectiveTransformV3`` — the same op that
    backs ``tf.keras.layers.RandomRotation`` / ``RandomTranslation`` — so a single
    ``[1, 8]`` transform matrix broadcasts to every frame.
    """

    def __init__(self, max_shift: float = 0.2, max_angle_deg: float = 90.0, p: float = 1.0):
        super().__init__(p)
        self.max_shift = max_shift
        self.max_angle = max_angle_deg * np.pi / 180.0

    def _apply(self, video, loading_option=None):
        shape = tf.shape(video)
        h = tf.cast(shape[1], tf.float32)
        w = tf.cast(shape[2], tf.float32)

        angle = tf.random.uniform((), -self.max_angle, self.max_angle)
        tx = tf.random.uniform((), -self.max_shift, self.max_shift) * w
        ty = tf.random.uniform((), -self.max_shift, self.max_shift) * h

        cos = tf.cos(angle)
        sin = tf.sin(angle)
        # Rotation about the frame center, in the op's output->input convention
        # (matches tf.keras get_rotation_matrix).
        x_off = ((w - 1.0) - (cos * (w - 1.0) - sin * (h - 1.0))) / 2.0
        y_off = ((h - 1.0) - (sin * (w - 1.0) + cos * (h - 1.0))) / 2.0

        zero = tf.zeros(())
        one = tf.ones(())
        rotation = tf.stack([
            tf.stack([cos, -sin, x_off]),
            tf.stack([sin, cos, y_off]),
            tf.stack([zero, zero, one]),
        ])
        translation = tf.stack([
            tf.stack([one, zero, -tx]),
            tf.stack([zero, one, -ty]),
            tf.stack([zero, zero, one]),
        ])
        matrix = tf.matmul(rotation, translation)
        # Flatten row-major and drop the trailing [0, 0, 1] -> [a0..b2, c0, c1].
        transform = tf.reshape(matrix, [-1])[:8]

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=video,
            transforms=tf.reshape(transform, [1, 8]),
            output_shape=shape[1:3],
            fill_value=0.0,
            interpolation='BILINEAR',
            fill_mode='CONSTANT',
        )


class RandomSectorMask(VideoTransform):
    """Zero out a random rectangular sector of the clip.

    The rectangle has height uniform in ``[0, max_frac * H]`` and width uniform in
    ``[0, max_frac * W]``, positioned uniformly so it fits inside the frame. With
    probability ``same_across_frames_prob`` the identical sector is masked in every
    frame; otherwise an independent sector is masked per frame.
    """

    def __init__(self, max_frac: float = 0.5, same_across_frames_prob: float = 0.5, p: float = 1.0):
        super().__init__(p)
        self.max_frac = max_frac
        self.same_across_frames_prob = same_across_frames_prob

    def _sample(self, count, hf, wf):
        rect_h = tf.random.uniform((count,), 0.0, self.max_frac) * hf
        rect_w = tf.random.uniform((count,), 0.0, self.max_frac) * wf
        top = tf.random.uniform((count,)) * (hf - rect_h)
        left = tf.random.uniform((count,)) * (wf - rect_w)
        return top, left, rect_h, rect_w

    def _apply(self, video, loading_option=None):
        shape = tf.shape(video)
        t, h, w = shape[0], shape[1], shape[2]
        hf = tf.cast(h, tf.float32)
        wf = tf.cast(w, tf.float32)

        if tf.random.uniform(()) < self.same_across_frames_prob:
            top, left, rect_h, rect_w = self._sample(1, hf, wf)
            top = tf.tile(top, [t])
            left = tf.tile(left, [t])
            rect_h = tf.tile(rect_h, [t])
            rect_w = tf.tile(rect_w, [t])
        else:
            top, left, rect_h, rect_w = self._sample(t, hf, wf)

        rows = tf.cast(tf.range(h), tf.float32)  # (H,)
        cols = tf.cast(tf.range(w), tf.float32)  # (W,)
        row_in = tf.logical_and(
            rows[None, :] >= top[:, None],
            rows[None, :] < (top + rect_h)[:, None],
        )  # (T, H)
        col_in = tf.logical_and(
            cols[None, :] >= left[:, None],
            cols[None, :] < (left + rect_w)[:, None],
        )  # (T, W)
        inside = tf.logical_and(row_in[:, :, None], col_in[:, None, :])  # (T, H, W)
        keep = tf.cast(tf.logical_not(inside), video.dtype)[..., None]  # (T, H, W, 1)
        return video * keep


class RandomFlip(VideoTransform):
    """Randomly flip the clip horizontally and/or vertically.

    Each axis is flipped independently with its own probability, and the flip is
    applied to the whole clip so all frames stay consistent.
    """

    def __init__(self, horizontal_prob: float = 0.5, vertical_prob: float = 0.5, p: float = 1.0):
        super().__init__(p)
        self.horizontal_prob = horizontal_prob
        self.vertical_prob = vertical_prob

    def _apply(self, video, loading_option=None):
        if tf.random.uniform(()) < self.horizontal_prob:
            video = tf.image.flip_left_right(video)
        if tf.random.uniform(()) < self.vertical_prob:
            video = tf.image.flip_up_down(video)
        return video


class RandomGaussianNoise(VideoTransform):
    """Add independent Gaussian noise to each frame.

    For each frame a noise standard deviation is drawn uniformly in
    ``[0, max_fraction * max_pixel_value_of_that_frame]`` and fresh noise is drawn
    per pixel, so every frame gets a different noise field but all channels within
    a frame share the same scale. Output is clipped to ``[0, 1]``.
    """

    def __init__(self, max_fraction: float = 0.2, p: float = 1.0):
        super().__init__(p)
        self.max_fraction = max_fraction

    def _apply(self, video, loading_option=None):
        shape = tf.shape(video)
        frame_max = tf.reduce_max(video, axis=[1, 2, 3], keepdims=True)  # (T, 1, 1, 1)
        std_shape = tf.stack([shape[0], 1, 1, 1])  # (T, 1, 1, 1)
        std = tf.random.uniform(std_shape, 0.0, self.max_fraction) * frame_max
        noise = tf.random.normal(shape) * std
        return tf.clip_by_value(video + noise, 0.0, 1.0)


class RandomBrightnessContrast(VideoTransform):
    """Randomly adjust brightness and/or contrast for the whole clip.

    A brightness delta ``~ N(0, std)`` and/or a contrast adjustment (factor
    ``= 1 + N(0, std)``) are sampled once and applied to every frame, keeping the
    color shift coherent across the clip. With probability ``per_channel_prob`` the
    adjustment is drawn independently per channel; otherwise a single value is
    shared across channels. Contrast uses the standard
    ``(x - channel_mean) * factor + channel_mean`` formulation. Output is clipped
    to ``[0, 1]``.
    """

    def __init__(self, std: float = 0.2, brightness_prob: float = 0.5,
                 contrast_prob: float = 0.5, per_channel_prob: float = 0.5, p: float = 1.0):
        super().__init__(p)
        self.std = std
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.per_channel_prob = per_channel_prob

    def _apply(self, video, loading_option=None):
        # Either draw one value per channel, or a single value shared across channels.
        if tf.random.uniform(()) < self.per_channel_prob:
            n = tf.shape(video)[-1]
        else:
            n = tf.constant(1)
        out = video
        if tf.random.uniform(()) < self.brightness_prob:
            delta = tf.random.normal(shape=tf.stack([n]), mean=0.0, stddev=self.std)
            out = out + tf.reshape(delta, [1, 1, 1, -1])
        if tf.random.uniform(()) < self.contrast_prob:
            factor = 1.0 + tf.random.normal(shape=tf.stack([n]), mean=0.0, stddev=self.std)
            mean = tf.reduce_mean(out, axis=[0, 1, 2], keepdims=True)  # (1, 1, 1, C)
            out = (out - mean) * tf.reshape(factor, [1, 1, 1, -1]) + mean
        return tf.clip_by_value(out, 0.0, 1.0)


# Convenience registry for wiring augmentations from a config/recipe.
AUGMENTATIONS = {
    'jitter_rotate': RandomJitterRotate,
    'sector_mask': RandomSectorMask,
    'flip': RandomFlip,
    'gaussian_noise': RandomGaussianNoise,
    'brightness_contrast': RandomBrightnessContrast,
}
