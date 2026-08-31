import numpy as np
import pytest
import tensorflow as tf

from ml4h.models.transformer_blocks_embedding import (
    build_embedding_transformer,
    build_general_embedding_transformer,
    evaluate_multitask_on_dataset,
)


try:
    tf.config.set_visible_devices([], "GPU")
except (RuntimeError, ValueError):
    # It's fine if devices are already initialized or no GPU backend exists.
    pass


def _cpu_device():
    # These tests run in GitHub Actions on CPU-only runners; force placement so
    # local GPU availability does not change behavior or runtime characteristics.
    return tf.device("/CPU:0")


def _make_dataset(inputs, outputs, batch_size=8, repeat=True):
    sample_weights = {
        name: np.ones((values.shape[0], 1), dtype=np.float32)
        for name, values in outputs.items()
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs, sample_weights)).batch(
        batch_size,
    )
    return dataset.repeat() if repeat else dataset


def _metric_value(metrics, task, metric):
    for row in metrics:
        if row["Task"] == task and row["Metric"] == metric:
            return row["Score"]
    raise AssertionError(f"Metric {metric} for task {task} not found in {metrics}")


def _print_metrics(label, metrics):
    lines = [
        "",
        f"{label} metrics",
        "task                 | metric | score",
        "---------------------+--------+----------",
    ]
    for row in metrics:
        lines.append(
            f"{row['Task']:<20} | {row['Metric']:<6} | {row['Score']:.6f}",
        )
    print("\n".join(lines), flush=True)


def _recompile_for_fair_test(model):
    losses = {}
    metrics = {}
    for output_name in model.output_names:
        layer = model.get_layer(output_name)
        activation = getattr(layer, "activation", None)
        is_binary = activation is not None and activation.__name__ == "sigmoid"
        if is_binary:
            losses[output_name] = "binary_crossentropy"
            metrics[output_name] = [
                tf.keras.metrics.AUC(name="auroc"),
                tf.keras.metrics.AUC(name="auprc", curve="PR"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
            ]
        else:
            losses[output_name] = "mse"
            metrics[output_name] = [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.MeanSquaredError(name="mse"),
            ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=losses,
        metrics=metrics,
    )
    return model


class _EchoPredictionModel(tf.keras.Model):
    def call(self, inputs, training=False):
        return {
            "regression_task": inputs["regression_prediction"],
            "binary_task": inputs["binary_prediction"],
        }


def test_evaluate_multitask_performance_data_includes_ci_and_counts():
    regression_truth = np.arange(8, dtype=np.float32)[:, None]
    regression_prediction = (np.arange(8, dtype=np.float32) + 0.1)[:, None]
    binary_truth = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.float32)[:, None]
    binary_prediction = np.array(
        [0.05, 0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.95],
        dtype=np.float32,
    )[:, None]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "regression_prediction": regression_prediction,
                "binary_prediction": binary_prediction,
            },
            {
                "regression_task": regression_truth,
                "binary_task": binary_truth,
            },
            {
                "regression_task": np.array(
                    [0, 1, 1, 1, 1, 1, 1, 1],
                    dtype=np.float32,
                )[:, None],
                "binary_task": np.array(
                    [1, 1, 1, 1, 1, 1, 1, 0],
                    dtype=np.float32,
                )[:, None],
            },
        ),
    ).batch(4)

    metrics = evaluate_multitask_on_dataset(
        "echo_model",
        _EchoPredictionModel(),
        dataset,
        regression_targets=["regression_task"],
        binary_targets=["binary_task"],
        verbose=False,
        n_bootstraps=100,
    )

    assert len(metrics) == 3
    for row in metrics:
        assert "CI_95_lower" in row
        assert "CI_95_upper" in row
        assert row["CI_95_lower"] <= row["CI_95_upper"]
        assert "n" in row

    regression_row = next(row for row in metrics if row["Task"] == "regression_task")
    assert regression_row["n"] == 7

    binary_rows = [row for row in metrics if row["Task"] == "binary_task"]
    assert {row["Metric"] for row in binary_rows} == {"auROC", "auPRC"}
    assert all(row["n"] == 7 for row in binary_rows)
    assert all(row["n_positive"] == 3 for row in binary_rows)


def test_build_embedding_transformer_positional_embedding_toggle():
    params = dict(
        input_numeric_cols=["feature_0", "feature_1"],
        regression_targets=["regression_task"],
        binary_targets=[],
        max_len=3,
        emb_dim=4,
        token_hidden=4,
        transformer_dim=4,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        view2id=None,
    )

    with _cpu_device():
        model_with_pos = build_embedding_transformer(
            **params,
            use_positional_embedding=True,
        )
        model_without_pos = build_embedding_transformer(
            **params,
        )

    with_pos_layer_names = {layer.name for layer in model_with_pos.layers}
    without_pos_layer_names = {layer.name for layer in model_without_pos.layers}

    assert "pos_embedding" in with_pos_layer_names
    assert "add_pos" in with_pos_layer_names
    assert "pos_embedding" not in without_pos_layer_names
    assert "add_pos" not in without_pos_layer_names


def test_build_embedding_transformer_handles_padded_categorical_tokens():
    max_len = 4
    num_features = 3

    num = np.array(
        [
            [[1.0, 0.0, 0.5], [0.5, 0.0, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.5], [0.2, 0.5, 0.1], [0.1, 0.2, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.4, 0.1, 0.3], [0.2, 0.1, 0.1], [0.1, 0.0, 0.1]],
            [[0.5, 0.5, 0.5], [0.3, 0.2, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, True, False, False],
        ],
        dtype=bool,
    )
    view = np.array(
        [
            [1, 2, 0, 0],
            [2, 1, 1, 0],
            [1, 2, 1, 2],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )
    outputs = {"regression_task": np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)}

    with _cpu_device():
        model = build_embedding_transformer(
            input_numeric_cols=[f"feature_{i}" for i in range(num_features)],
            regression_targets=["regression_task"],
            binary_targets=[],
            max_len=max_len,
            emb_dim=4,
            token_hidden=8,
            transformer_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            view2id={"apical": 1, "parasternal": 2},
            learning_rate=1e-3,
        )
        history = model.fit(
            {"num": num, "mask": mask, "view": view},
            outputs,
            epochs=1,
            batch_size=2,
            verbose=0,
        )

    assert np.isfinite(history.history["loss"][-1])


@pytest.mark.parametrize("use_categorical", [False, True])
def test_build_general_embedding_transformer_learns_easy_tasks(use_categorical, capsys):
    tf.keras.utils.set_random_seed(1234)

    num_samples = 32
    max_len = 4
    latent_dim = 1
    numeric_columns = ["age"]
    categorical_columns = ["site"] if use_categorical else []
    categorical_vocabs = {"site": 3} if use_categorical else {}

    latent = np.zeros((num_samples, max_len, latent_dim), dtype=np.float32)

    age = np.tile(
        np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)[:, None],
        (1, max_len),
    )
    mask = np.ones((num_samples, max_len), dtype=bool)

    inputs = {
        "latent": latent,
        "num_age": age,
        "mask": mask,
    }

    regression_target = age[:, 0].astype(np.float32)
    binary_signal = age[:, 0]

    if use_categorical:
        site = (np.arange(num_samples) % 4).astype(np.int32)
        site = np.tile(site[:, None], (1, max_len))
        inputs["cat_site"] = site

    binary_target = (binary_signal > 0).astype(np.float32)

    outputs = {
        "regression_task": regression_target[:, None],
        "binary_task": binary_target[:, None],
    }

    with _cpu_device():
        model = build_general_embedding_transformer(
            latent_dim=latent_dim,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            categorical_vocabs=categorical_vocabs,
            regression_targets=["regression_task"],
            binary_targets=["binary_task"],
            max_len=max_len,
            scalar_embed=4,
            latent_embed=4,
            transformer_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        model = _recompile_for_fair_test(model)

    train_ds = _make_dataset(inputs, outputs, batch_size=8, repeat=True)
    eval_ds = _make_dataset(inputs, outputs, batch_size=8, repeat=False)

    with _cpu_device():
        history = model.fit(train_ds, epochs=20, steps_per_epoch=4, verbose=0)
    assert history.history["loss"][-1] < history.history["loss"][0]

    with _cpu_device():
        metrics = evaluate_multitask_on_dataset(
            "general_embedding_transformer",
            model,
            eval_ds,
            regression_targets=["regression_task"],
            binary_targets=["binary_task"],
            verbose=False,
        )

    with capsys.disabled():
        _print_metrics(
            f"build_general_embedding_transformer(use_categorical={use_categorical})",
            metrics,
        )

    assert _metric_value(metrics, "regression_task", "R^2") > 0.85
    assert _metric_value(metrics, "binary_task", "auROC") > 0.98
    assert _metric_value(metrics, "binary_task", "auPRC") > 0.98


def test_build_embedding_transformer_learns_easy_tasks(capsys):
    tf.keras.utils.set_random_seed(4321)

    num_samples = 32
    max_len = 4
    num_features = 64
    view2id = {"apical": 1, "parasternal": 2}

    num = np.zeros((num_samples, max_len, num_features), dtype=np.float32)
    signal = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)
    num[:, :, 0] = signal[:, None]
    num[:, :, 1] = (signal * 0.5)[:, None]
    num[:, :, 2] = 1.0

    mask = np.ones((num_samples, max_len), dtype=bool)
    view = np.where(signal > 0, 2, 1).astype(np.int32)
    view = np.tile(view[:, None], (1, max_len))

    regression_target = (1.5 * signal + 0.25).astype(np.float32)
    binary_target = (signal > 0).astype(np.float32)

    inputs = {
        "num": num,
        "mask": mask,
        "view": view,
    }
    outputs = {
        "regression_task": regression_target[:, None],
        "binary_task": binary_target[:, None],
    }

    with _cpu_device():
        model = build_embedding_transformer(
            input_numeric_cols=[f"feature_{i}" for i in range(num_features)],
            regression_targets=["regression_task"],
            binary_targets=["binary_task"],
            max_len=max_len,
            emb_dim=8,
            token_hidden=12,
            transformer_dim=12,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            view2id=view2id,
            learning_rate=1e-3,
        )
        model = _recompile_for_fair_test(model)

    train_ds = _make_dataset(inputs, outputs, batch_size=8, repeat=True)
    eval_ds = _make_dataset(inputs, outputs, batch_size=8, repeat=False)

    with _cpu_device():
        history = model.fit(train_ds, epochs=20, steps_per_epoch=4, verbose=0)
    assert history.history["loss"][-1] < history.history["loss"][0]

    with _cpu_device():
        metrics = evaluate_multitask_on_dataset(
            "embedding_transformer",
            model,
            eval_ds,
            regression_targets=["regression_task"],
            binary_targets=["binary_task"],
            verbose=False,
        )

    with capsys.disabled():
        _print_metrics("build_embedding_transformer", metrics)

    assert _metric_value(metrics, "regression_task", "R^2") > 0.9
    assert _metric_value(metrics, "binary_task", "auROC") > 0.98
    assert _metric_value(metrics, "binary_task", "auPRC") > 0.98
