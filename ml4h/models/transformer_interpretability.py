"""
Chefer et al., CVPR 2021 — "Transformer Interpretability Beyond Attention Visualization"
Gradient-weighted attention rollout for encoder-only transformers.

Usage (inference only — model weights are never modified):

    relevancy = compute_relevancy_scores(
        model=trained_model,
        inputs=batch_inputs_dict,
        target_name="my_binary_target",
        num_layers=4,
    )
    # relevancy: np.ndarray of shape (B, T), rows sum to ~1.0
    # Each value is the relevancy of that visit/token to the prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers as klayers
import matplotlib.pyplot as plt

def detect_transformer_builder_type(model):
    input_names = [x.name.split(":")[0] for x in model.inputs]

    if "latent" in input_names:
        return "general_embedding_transformer"

    if "num" in input_names and "mask" in input_names:
        return "embedding_transformer"

    raise ValueError(f"Unknown transformer type. Inputs found: {input_names}")


def extract_build_args_from_transformer_model(model):
    builder_type = detect_transformer_builder_type(model)

    output_names = model.output_names

    regression_targets = []
    binary_targets = []
    categorical_targets = []
    num_classes = None

    for name in output_names:
        layer = model.get_layer(name)
        units = int(layer.units)
        activation = layer.activation.__name__

        if units == 1 and activation == "linear":
            regression_targets.append(name)
        elif units == 1 and activation == "sigmoid":
            binary_targets.append(name)
        elif units > 1 and activation == "softmax":
            categorical_targets.append(name)
            num_classes = units
        else:
            raise ValueError(
                f"Cannot classify output {name}: units={units}, activation={activation}"
            )

    mha_layers = sorted(
        [l for l in model.layers if l.name.startswith("mha_")],
        key=lambda l: int(l.name.split("_")[1])
    )

    if not mha_layers:
        raise ValueError("No mha_0, mha_1, ... layers found.")

    num_layers = len(mha_layers)
    num_heads = int(mha_layers[0].num_heads)
    dropout = float(mha_layers[0].dropout)

    if builder_type == "embedding_transformer":
        inp = {x.name.split(":")[0]: x for x in model.inputs}

        max_len = int(inp["num"].shape[1])
        feat = int(inp["num"].shape[2])

        token_hidden = int(model.get_layer("token_proj").units)

        # In your builder, MHA key_dim uses transformer_dim // num_heads.
        # So recover transformer_dim from key_dim * num_heads.
        transformer_dim = int(mha_layers[0].key_dim * num_heads)

        has_view = "view" in inp

        view2id = None
        emb_dim = None

        if has_view:
            view_emb = model.get_layer("view_embedding")
            emb_dim = int(view_emb.output_dim)

            # We cannot recover original category strings, only IDs.
            # This dummy mapping preserves max id / vocab size.
            max_id = int(view_emb.input_dim - 1)
            view2id = {str(i): i for i in range(1, max_id + 1)}
        else:
            emb_dim = None

        use_positional_embedding = any(l.name == "pos_embedding" for l in model.layers)

        return {
            "builder_type": builder_type,
            "build_args": {
                "input_numeric_cols": [f"feature_{i}" for i in range(feat)],
                "regression_targets": regression_targets,
                "binary_targets": binary_targets,
                "max_len": max_len,
                "emb_dim": emb_dim,
                "token_hidden": token_hidden,
                "transformer_dim": transformer_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dropout": dropout,
                "view2id": view2id,
                "learning_rate": 0.00005,
                "binary_class_prevalences": None,
                "categorical_targets": categorical_targets,
                "num_classes": num_classes,
                "label_weights": None,
                "use_positional_embedding": use_positional_embedding,
            },
        }

    if builder_type == "general_embedding_transformer":
        inp = {x.name.split(":")[0]: x for x in model.inputs}

        max_len = int(inp["latent"].shape[1])
        latent_dim = int(inp["latent"].shape[2])

        numeric_columns = [
            name.replace("num_", "", 1)
            for name in inp
            if name.startswith("num_")
        ]

        categorical_columns = [
            name.replace("cat_", "", 1)
            for name in inp
            if name.startswith("cat_")
        ]

        latent_embed = int(model.get_layer("latent_emb").units)
        transformer_dim = int(model.get_layer("emb_projection").units)

        if numeric_columns:
            scalar_embed = int(model.get_layer(f"num_{numeric_columns[0]}_emb").units)
        elif categorical_columns:
            scalar_embed = int(model.get_layer(f"cat_{categorical_columns[0]}_emb").output_dim)
        else:
            raise ValueError("Cannot infer scalar_embed.")

        categorical_vocabs = {}
        for col in categorical_columns:
            emb = model.get_layer(f"cat_{col}_emb")
            categorical_vocabs[col] = int(emb.input_dim - 1)

        return {
            "builder_type": builder_type,
            "build_args": {
                "latent_dim": latent_dim,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "categorical_vocabs": categorical_vocabs,
                "regression_targets": regression_targets,
                "binary_targets": binary_targets,
                "max_len": max_len,
                "scalar_embed": scalar_embed,
                "latent_embed": latent_embed,
                "transformer_dim": transformer_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dropout": dropout,
                "categorical_targets": categorical_targets,
                "num_classes": num_classes,
                "label_weights": None,
            },
        }

def chefer_rollout_keras(
    model_explain,
    batch_inputs,
    target_name,
    class_index=None,
    residual_weight=0.5,
    attention_weight=0.5,
    use_abs_grad=True,
    positive_only=False,
    pool_mode="attention",  # "attention" or "uniform"
):
    """
    Returns token/timestamp relevance: shape (B, T)

    pool_mode:
        "attention" = collapse rollout using model attention-pooling weights
        "uniform"   = collapse rollout by averaging over valid query tokens
    """

    with tf.GradientTape() as tape:
        outputs = model_explain(batch_inputs, training=False)

        attn_keys = sorted(
            [
                k for k in outputs.keys()
                if k.startswith("mha_") and k.endswith("_scores")
            ],
            key=lambda x: int(x.split("_")[1]),
        )
        print(f"Found {len(attn_keys)} attention score outputs: {attn_keys}")
        attn_scores = [outputs[k] for k in attn_keys]
        tape.watch(attn_scores)

        y = outputs[target_name]

        if class_index is not None:
            target_score = y[:, class_index]
        else:
            target_score = tf.reshape(y, [tf.shape(y)[0], -1])[:, 0]

    grads = tape.gradient(target_score, attn_scores)

    B = tf.shape(attn_scores[0])[0]
    T = tf.shape(attn_scores[0])[-1]

    mask = tf.cast(batch_inputs["mask"], tf.float32)
    pair_mask = mask[:, :, None] * mask[:, None, :]

    I = tf.eye(T, batch_shape=[B]) * mask[:, :, None]
    R = I

    for A, G in zip(attn_scores, grads):
        if use_abs_grad:
            cam = A * tf.abs(G)
        else:
            cam = A * G

        if positive_only:
            cam = tf.nn.relu(cam)

        cam = tf.reduce_mean(cam, axis=1)
        cam = cam * pair_mask

        cam = cam / (tf.reduce_sum(cam, axis=-1, keepdims=True) + 1e-8)

        cam = residual_weight * I + attention_weight * cam
        cam = cam * pair_mask
        cam = cam / (tf.reduce_sum(cam, axis=-1, keepdims=True) + 1e-8)

        R = tf.matmul(cam, R)

    if pool_mode == "attention":
        query_wts = outputs["attn_wts"]

    elif pool_mode == "uniform":
        valid_count = tf.reduce_sum(mask, axis=-1, keepdims=True)
        query_wts = mask / (valid_count + 1e-8)

    else:
        raise ValueError(f"Unknown pool_mode: {pool_mode}")

    relevance = tf.reduce_sum(R * query_wts[:, :, None], axis=1)

    relevance = relevance * mask
    relevance = relevance / (
        tf.reduce_sum(relevance, axis=-1, keepdims=True) + 1e-8
    )

    return relevance.numpy(), outputs


def normalize_rows(A, eps=1e-8):
    row_sum = A.sum(axis=-1, keepdims=True)
    return A / (row_sum + eps)


def _ordered_attn_score_keys(outputs, n_layers=None):
    """
    Return attention-score output keys in layer order.

    Supports both naming conventions:
        notebook style: attn_scores_0, attn_scores_1, ...
        ml4h builder:   mha_0_scores, mha_1_scores, ...
    """
    keys = sorted(
        [k for k in outputs if k.startswith("attn_scores_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not keys:
        keys = sorted(
            [k for k in outputs if k.startswith("mha_") and k.endswith("_scores")],
            key=lambda x: int(x.split("_")[1]),
        )
    if not keys:
        raise ValueError(
            "No attention-score outputs found "
            "(expected attn_scores_* or mha_*_scores)."
        )
    if n_layers is not None:
        keys = keys[:n_layers]
    return keys


def grad_weighted_attention_rollout(
    explainable_model,
    batch_inputs,
    actual_len,
    n_layers=None,
    target_key="prediction",
    residual_weight=0.5,
    attention_weight=0.5,
    use_abs_grad=True,
    positive_only=False,
):
    """
    Single-sample gradient-weighted attention rollout (Chefer et al., CVPR 2021).

    This is a direct port of the reference notebook implementation
    (grad_weighted_attention_rollout in visTrans.ipynb). It runs per layer in
    NumPy, mixes a residual/identity path with the gradient-weighted attention
    path, rolls the per-layer matrices together, and collapses the rollout to a
    per-token relevance vector using a *uniform* mean over query tokens.

    Inputs
    ------
    explainable_model:
        Keras model whose outputs include `target_key` and per-layer attention
        scores named either `attn_scores_{i}` or `mha_{i}_scores`.
    batch_inputs:
        dict of model inputs for a single sample, e.g.
        {"num": [1, T, F], "mask": [1, T] bool} (plus "view": [1, T] if the
        model was built with a categorical/view input). Passed to the model
        as-is, so every required input must be present.
    actual_len:
        number of real (non-padded) tokens for this sample
    n_layers:
        number of transformer layers to use; if None, all attention-score
        outputs are used in order.
    target_key:
        model output to explain (scalar head, e.g. "prediction").

    use_abs_grad:
        True:  cam = A * |dy/dA|
        False + positive_only=True:  cam = ReLU(A * dy/dA)
        False + positive_only=False: cam = A * dy/dA (raw signed)

    Returns
    -------
    relevance_real:   [actual_len] normalized token relevance
    rollout_real:     [actual_len, actual_len] rollout matrix
    outputs_np:       dict of model outputs (numpy)
    layer_debug_df:   per-layer diagnostics
    layer_mats:       list of per-layer mixed matrices
    """

    mask_input = np.asarray(batch_inputs["mask"])
    valid = mask_input[0].astype(bool)
    valid_idx = np.where(valid)[0]

    with tf.GradientTape(persistent=True) as tape:
        outputs = explainable_model(batch_inputs, training=False)

        target = outputs[target_key][0, 0]

        attn_keys = _ordered_attn_score_keys(outputs, n_layers)
        attn_scores_list = [outputs[k] for k in attn_keys]

    layer_mats = []
    layer_debug = []

    for i, attn_scores in enumerate(attn_scores_list):
        grads = tape.gradient(target, attn_scores)

        if grads is None:
            raise ValueError(f"Gradient is None for {attn_keys[i]}")

        A = attn_scores[0]  # [heads, T, T]
        G = grads[0]        # [heads, T, T]

        if use_abs_grad:
            cam = A * tf.abs(G)
        else:
            cam = A * G
            if positive_only:
                cam = tf.nn.relu(cam)

        # Average across heads
        cam = tf.reduce_mean(cam, axis=0).numpy()  # [T, T]

        # Remove padded positions
        cam[~valid, :] = 0.0
        cam[:, ~valid] = 0.0

        # Normalize gradient-weighted attention
        cam = normalize_rows(cam)

        # Residual matrix only on real tokens
        I = np.zeros_like(cam)
        I[valid_idx, valid_idx] = 1.0

        # Mix residual path and gradient-weighted attention path
        M = residual_weight * I + attention_weight * cam

        # Remove padding again
        M[~valid, :] = 0.0
        M[:, ~valid] = 0.0

        # Normalize rows
        M = normalize_rows(M)

        M_real = M[:actual_len, :actual_len]
        offdiag = M_real.copy()
        np.fill_diagonal(offdiag, 0.0)

        layer_debug.append({
            "layer": i,
            "attn_scores_min": float(A.numpy()[:, :actual_len, :actual_len].min()),
            "attn_scores_max": float(A.numpy()[:, :actual_len, :actual_len].max()),
            "grad_min": float(G.numpy()[:, :actual_len, :actual_len].min()),
            "grad_max": float(G.numpy()[:, :actual_len, :actual_len].max()),
            "cam_sum_real": float(cam[:actual_len, :actual_len].sum()),
            "M_diag_mean": float(np.diag(M_real).mean()),
            "M_diag_min": float(np.diag(M_real).min()),
            "M_diag_max": float(np.diag(M_real).max()),
            "M_offdiag_sum": float(offdiag.sum()),
            "M_offdiag_max": float(offdiag.max()),
        })

        layer_mats.append(M)

    del tape

    # Start rollout from identity over valid tokens
    rollout = np.zeros_like(layer_mats[0])
    rollout[valid_idx, valid_idx] = 1.0

    for M in layer_mats:
        rollout = M @ rollout

    rollout_real = rollout[:actual_len, :actual_len]

    # Since the model later pools all tokens, summarize source-token relevance
    relevance_real = rollout_real.mean(axis=0)
    relevance_real = relevance_real / (relevance_real.sum() + 1e-12)

    outputs_np = {k: v.numpy() for k, v in outputs.items()}
    layer_debug_df = pd.DataFrame(layer_debug)

    return relevance_real, rollout_real, outputs_np, layer_debug_df, layer_mats


def plot_average_rollout_importance_from_results_binned(
    results,
    target,
    output_folder,
    run_id,
    n_position_bins=10,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    rel_col = f"{target}_relevance"
    rows = []

    for sample_id, rel, n_rows in zip(
        results["mrn"],
        results[rel_col],
        results["n_rows"],
    ):
        rel = np.asarray(rel, dtype=float)[: int(n_rows)]
        n = len(rel)

        if n == 0:
            continue
        if n < 2:
                continue

        uniform = 1.0 / n

        for pos, score in enumerate(rel):
            relative_position = pos / (n-1)

            rows.append({
                "sample_id": sample_id,
                "position_index": pos,
                "n_rows": n,
                "relative_position": relative_position,
                "rollout": float(score),
                "rollout_ratio_uniform": float(score / uniform),
            })

    global_rollout_df = pd.DataFrame(rows)

    overall_position_summary = (
        global_rollout_df
        .assign(
            position_bin=lambda x: pd.cut(
                x["relative_position"],
                bins=np.linspace(0, 1, n_position_bins + 1),
                include_lowest=True,
            )
        )
        .groupby("position_bin", observed=False)
        .agg(
            mean_rollout_ratio=("rollout_ratio_uniform", "mean"),
            median_rollout_ratio=("rollout_ratio_uniform", "median"),
            sem_rollout_ratio=(
                "rollout_ratio_uniform",
                lambda x: x.std() / np.sqrt(len(x)),
            ),
            n_ecgs=("rollout_ratio_uniform", "count"),
            n_patients=("sample_id", "nunique"),
        )
        .reset_index()
    )

    overall_position_summary["position_mid"] = (
        overall_position_summary["position_bin"]
        .apply(lambda x: x.mid)
        .astype(float)
    )

    os.makedirs(output_folder, exist_ok=True)

    summary_path = os.path.join(
        output_folder,
        f"overall_position_rollout_summary_{target}_{run_id}.csv",
    )
    overall_position_summary.to_csv(summary_path, index=False)

    fig = plt.figure(figsize=(9, 5))

    x = overall_position_summary["position_mid"].astype(float).values
    y = overall_position_summary["mean_rollout_ratio"].astype(float).values
    sem = overall_position_summary["sem_rollout_ratio"].astype(float).values

    plt.plot(x, y, marker="o", label="Mean rollout / uniform")
    plt.fill_between(x, y - sem, y + sem, alpha=0.2)

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="Uniform baseline",
    )

    plt.xlabel(f"Relative {target}(Token) position within patient sequence")
    plt.ylabel("Rollout relevance / uniform")
    plt.title(f"Average rollout importance across all {target} Token-count groups")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(
        output_folder,
        f"overall_mean_rollout_ratio_by_position_{target}_{run_id}.png",
    )

    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return global_rollout_df, overall_position_summary, summary_path, plot_path