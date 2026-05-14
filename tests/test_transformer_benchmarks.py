import time
from statistics import median

import numpy as np
import pytest
import tensorflow as tf

from ml4h.models.transformer_blocks_embedding import (
    build_embedding_transformer,
    build_general_embedding_transformer,
)


try:
    tf.config.set_visible_devices([], "GPU")
except (RuntimeError, ValueError):
    # Fine if devices are already initialized or GPU is unavailable.
    pass


def _cpu_device():
    # Keep benchmark placement consistent with GitHub Actions CPU runners.
    return tf.device("/CPU:0")


def _make_general_inputs(num_samples, max_len, feature_dim):
    latent = np.zeros((num_samples, max_len, feature_dim), dtype=np.float32)
    latent[:, :, 0] = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)[:, None]
    latent[:, :, 1] = 1.0
    mask = np.ones((num_samples, max_len), dtype=bool)
    return {"latent": latent, "mask": mask}


def _make_embedding_inputs(num_samples, max_len, feature_dim):
    num = np.zeros((num_samples, max_len, feature_dim), dtype=np.float32)
    num[:, :, 0] = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)[:, None]
    num[:, :, 1] = 1.0
    mask = np.ones((num_samples, max_len), dtype=bool)
    return {"num": num, "mask": mask}


def _make_outputs(signal):
    regression = (1.25 * signal + 0.1).astype(np.float32)
    binary = (signal > 0).astype(np.float32)
    return {
        "regression_task": regression[:, None],
        "binary_task": binary[:, None],
    }


def _make_dataset(inputs, outputs, batch_size=8, repeat=True):
    sample_weights = {
        name: np.ones((values.shape[0], 1), dtype=np.float32)
        for name, values in outputs.items()
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs, sample_weights)).batch(
        batch_size,
    )
    return dataset.repeat() if repeat else dataset


def _measure_call(fn, repeats):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return median(durations)


def _benchmark_model(builder_name, builder_fn, inputs, outputs):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(1234)

    with _cpu_device():
        build_time_s = _measure_call(builder_fn, repeats=2)
        model = builder_fn()

    train_ds = _make_dataset(inputs, outputs, batch_size=8, repeat=True)
    infer_ds = _make_dataset(inputs, outputs, batch_size=8, repeat=False)
    inference_batches = len(list(infer_ds))
    inference_examples = sum(batch_inputs[next(iter(batch_inputs))].shape[0] for batch_inputs, _, _ in infer_ds)

    with _cpu_device():
        model.fit(train_ds, steps_per_epoch=1, epochs=1, verbose=0)

    with _cpu_device():
        train_step_time_s = _measure_call(
            lambda: model.fit(train_ds, steps_per_epoch=2, epochs=1, verbose=0),
            repeats=2,
        ) / 2.0

    with _cpu_device():
        infer_batch_time_s = _measure_call(
            lambda: model.predict(infer_ds, verbose=0),
            repeats=2,
        ) / inference_batches

    return {
        "builder": builder_name,
        "build_time_s": build_time_s,
        "train_step_time_s": train_step_time_s,
        "infer_batch_time_s": infer_batch_time_s,
        "train_examples_per_s": 8.0 / train_step_time_s,
        "infer_examples_per_s": inference_examples / (infer_batch_time_s * inference_batches),
        "trainable_parameter_count": int(
            np.sum([np.prod(v.shape) for v in model.trainable_weights], dtype=np.int64),
        ),
        "non_trainable_parameter_count": int(
            np.sum([np.prod(v.shape) for v in model.non_trainable_weights], dtype=np.int64),
        ),
        "total_parameter_count": model.count_params(),
    }


def _format_metric(value, kind):
    if kind == "time":
        return f"{value:.6f}s"
    if kind == "throughput":
        return f"{value:,.1f}/s"
    if kind == "count":
        return f"{int(value):,}"
    return str(value)


def _comparison_row(label, general_value, embedding_value, kind):
    ratio = embedding_value / general_value if general_value else float("inf")
    return (
        f"{label:<22} | "
        f"{_format_metric(general_value, kind):>14} | "
        f"{_format_metric(embedding_value, kind):>14} | "
        f"{ratio:>8.2f}x"
    )


@pytest.mark.slow
def test_transformer_builder_benchmarks_are_measurable_and_comparable(capsys):
    num_samples = 16
    max_len = 4
    feature_dim = 64
    signal = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)
    outputs = _make_outputs(signal)

    general_inputs = _make_general_inputs(num_samples, max_len, feature_dim)
    embedding_inputs = _make_embedding_inputs(num_samples, max_len, feature_dim)

    general_metrics = _benchmark_model(
        "build_general_embedding_transformer",
        lambda: build_general_embedding_transformer(
            latent_dim=feature_dim,
            numeric_columns=[],
            categorical_columns=[],
            categorical_vocabs={},
            regression_targets=["regression_task"],
            binary_targets=["binary_task"],
            max_len=max_len,
            scalar_embed=8,
            latent_embed=8,
            transformer_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        ),
        general_inputs,
        outputs,
    )

    embedding_metrics = _benchmark_model(
        "build_embedding_transformer",
        lambda: build_embedding_transformer(
            input_numeric_cols=[f"feature_{i}" for i in range(feature_dim)],
            regression_targets=["regression_task"],
            binary_targets=["binary_task"],
            max_len=max_len,
            emb_dim=16,
            token_hidden=12,
            transformer_dim=12,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            view2id=None,
            learning_rate=1e-3,
        ),
        embedding_inputs,
        outputs,
    )

    benchmark_pairs = [
        ("build_time_s", general_metrics["build_time_s"], embedding_metrics["build_time_s"]),
        (
            "train_step_time_s",
            general_metrics["train_step_time_s"],
            embedding_metrics["train_step_time_s"],
        ),
        (
            "infer_batch_time_s",
            general_metrics["infer_batch_time_s"],
            embedding_metrics["infer_batch_time_s"],
        ),
        (
            "train_examples_per_s",
            general_metrics["train_examples_per_s"],
            embedding_metrics["train_examples_per_s"],
        ),
        (
            "infer_examples_per_s",
            general_metrics["infer_examples_per_s"],
            embedding_metrics["infer_examples_per_s"],
        ),
    ]

    for metric_name, general_value, embedding_value in benchmark_pairs:
        assert general_value > 0.0, f"{metric_name} should be measurable for the general builder"
        assert embedding_value > 0.0, f"{metric_name} should be measurable for the embedding builder"
        ratio = max(general_value, embedding_value) / min(general_value, embedding_value)
        assert ratio < 25.0, f"{metric_name} ratio is unexpectedly large: {ratio:.2f}"

    assert general_metrics["trainable_parameter_count"] > 0
    assert embedding_metrics["trainable_parameter_count"] > 0
    assert general_metrics["non_trainable_parameter_count"] >= 0
    assert embedding_metrics["non_trainable_parameter_count"] >= 0
    assert general_metrics["total_parameter_count"] > 0
    assert embedding_metrics["total_parameter_count"] > 0

    report_lines = [
        "",
        "Transformer benchmark comparison",
        "metric                 |        general |      embedding |    ratio",
        "-----------------------+----------------+----------------+---------",
        _comparison_row(
            "build time",
            general_metrics["build_time_s"],
            embedding_metrics["build_time_s"],
            "time",
        ),
        _comparison_row(
            "train step time",
            general_metrics["train_step_time_s"],
            embedding_metrics["train_step_time_s"],
            "time",
        ),
        _comparison_row(
            "infer batch time",
            general_metrics["infer_batch_time_s"],
            embedding_metrics["infer_batch_time_s"],
            "time",
        ),
        _comparison_row(
            "train throughput",
            general_metrics["train_examples_per_s"],
            embedding_metrics["train_examples_per_s"],
            "throughput",
        ),
        _comparison_row(
            "infer throughput",
            general_metrics["infer_examples_per_s"],
            embedding_metrics["infer_examples_per_s"],
            "throughput",
        ),
        _comparison_row(
            "trainable params",
            general_metrics["trainable_parameter_count"],
            embedding_metrics["trainable_parameter_count"],
            "count",
        ),
        _comparison_row(
            "non-trainable params",
            general_metrics["non_trainable_parameter_count"],
            embedding_metrics["non_trainable_parameter_count"],
            "count",
        ),
        _comparison_row(
            "total params",
            general_metrics["total_parameter_count"],
            embedding_metrics["total_parameter_count"],
            "count",
        ),
    ]

    with capsys.disabled():
        print("\n".join(report_lines), flush=True)
