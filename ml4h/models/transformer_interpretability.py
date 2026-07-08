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
import tensorflow as tf
import keras
from keras import layers as klayers
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Step 1 — Layer traversal helpers
# ---------------------------------------------------------------------------

def _run_pretransformer_embedding(model, inputs, training=False):
    """
    Runs the pre-transformer layers for build_embedding_transformer.

    Returns:
        x        (B, T, token_hidden)  — ready for transformer blocks
        mask     (B, T, T) bool        — 2-D attention mask for MHA
        inp_mask (B, T) bool           — 1-D mask needed for pooling
    """
    inp_mask = tf.cast(inputs['mask'], dtype='bool')

    if 'view' in inputs:
        view_emb = model.get_layer('view_embedding')(inputs['view'], training=training)
        x = model.get_layer('token_concat')([view_emb, inputs['num']])
    else:
        x = inputs['num']

    x = model.get_layer('token_proj')(x, training=training)

    # Learnable positional embedding (optional — not all models have it)
    try:
        pos_idx = model.get_layer('pos_idx')(x)
        pos_emb = model.get_layer('pos_embedding')(pos_idx, training=training)
        x = model.get_layer('add_pos')([x, pos_emb])
    except ValueError:
        pass

    # Build 2-D (B, T, T) attention mask
    m_q = model.get_layer('mask_q')(inp_mask)
    m_k = model.get_layer('mask_k')(inp_mask)
    mask_2d = model.get_layer('mask_qk')([m_q, m_k])

    return x, mask_2d, inp_mask


def _run_pretransformer_general(model, inputs, training=False):
    """
    Runs the pre-transformer layers for build_general_embedding_transformer.

    Returns:
        x        (B, T, transformer_dim)  — ready for transformer blocks
        mask     (B, 1, T) bool           — 1-D expanded attention mask for MHA
        inp_mask (B, T) bool              — 1-D mask needed for pooling
    """
    inp_mask = tf.cast(inputs['mask'], dtype='bool')

    # Numeric columns
    num_keys = sorted(k for k in inputs if k.startswith('num_'))
    num_embs = []
    for key in num_keys:
        col = key[len('num_'):]
        xc = model.get_layer(f'num_{col}_expand')(inputs[key])
        xc = model.get_layer(f'num_{col}_emb')(xc, training=training)
        num_embs.append(xc)

    if len(num_embs) > 1:
        num_emb = model.get_layer('num_emb_sum')(num_embs)
    elif len(num_embs) == 1:
        num_emb = num_embs[0]
    else:
        num_emb = None

    # Categorical columns
    cat_keys = sorted(k for k in inputs if k.startswith('cat_'))
    cat_embs = []
    for key in cat_keys:
        col = key[len('cat_'):]
        xc = model.get_layer(f'cat_{col}_emb')(inputs[key], training=training)
        cat_embs.append(xc)

    if len(cat_embs) > 1:
        cat_emb = model.get_layer('cat_emb_sum')(cat_embs)
    elif len(cat_embs) == 1:
        cat_emb = cat_embs[0]
    else:
        cat_emb = None

    # Latent + concat + projection
    latent_emb = model.get_layer('latent_emb')(inputs['latent'], training=training)
    to_concat = [latent_emb]
    if num_emb is not None:
        to_concat.append(num_emb)
    if cat_emb is not None:
        to_concat.append(cat_emb)

    x = model.get_layer('total_emb')(to_concat)
    x = model.get_layer('emb_projection')(x, training=training)

    # Positional embedding
    pos_idx = model.get_layer('pos_indices')(x)
    pos_emb = model.get_layer('pos_embedding')(pos_idx, training=training)
    x = model.get_layer('add_pos')([x, pos_emb])

    # Attention mask: (B, T) → (B, 1, T)
    mask_1d = model.get_layer('attn_mask')(inp_mask)

    return x, mask_1d, inp_mask


def _run_block_remainder_embedding(model, i, x_pre, mha_out, training=False):
    """
    Runs everything after the MHA call for block i in build_embedding_transformer.
    Unnamed Dropout layers are skipped (no-op at inference).

      x_pre + mha_out → ln1_{i} → ff1_{i} → ff2_{i} → ln2_{i}
    """
    x = model.get_layer(f'ln1_{i}')(x_pre + mha_out)
    ff = model.get_layer(f'ff1_{i}')(x, training=training)
    ff = model.get_layer(f'ff2_{i}')(ff, training=training)
    x = model.get_layer(f'ln2_{i}')(x + ff)
    return x


def _run_block_remainder_general(model, i, x_pre, mha_out, training=False):
    """
    Runs everything after the MHA call for block i in build_general_embedding_transformer.

      attn_dropout_{i} → attn_residual_{i} → attn_norm_{i}
      → ffn_dense_1_{i} → ffn_dropout_{i} → ffn_dense_2_{i} → ffn_dropout_{i}
      → ffn_residual_{i} → ffn_norm_{i}
    """
    attn = model.get_layer(f'attn_dropout_{i}')(mha_out, training=training)
    x = model.get_layer(f'attn_residual_{i}')([x_pre, attn])
    x = model.get_layer(f'attn_norm_{i}')(x)

    ff = model.get_layer(f'ffn_dense_1_{i}')(x, training=training)
    ff = model.get_layer(f'ffn_dropout_1_{i}')(ff, training=training)
    ff = model.get_layer(f'ffn_dense_2_{i}')(ff, training=training)
    ff = model.get_layer(f'ffn_dropout_2_{i}')(ff, training=training)
    x = model.get_layer(f'ffn_residual_{i}')([x, ff])
    x = model.get_layer(f'ffn_norm_{i}')(x)
    return x


def _run_pooling(model, x, inp_mask):
    """
    Runs the attention pooling shared by both builders.

    Returns:
        attn_wts  (B, T)  — softmax pooling weights (used for Chefer weighting)
        ctx       (B, D)  — pooled context vector
    """
    score_h = model.get_layer('attn_h')(x)
    score = model.get_layer('attn_score')(score_h)
    score = model.get_layer('attn_score_squeeze')(score)
    mask_f = model.get_layer('mask_cast')(inp_mask)
    very_neg = model.get_layer('veryneg')(mask_f)
    score_m = model.get_layer('score_masked')([score, very_neg])
    attn_wts = model.get_layer('attn_wts')(score_m)
    wts_e = model.get_layer('wts_e')(attn_wts)
    ctx = model.get_layer('apply_wts')([x, wts_e])
    ctx = model.get_layer('pool')(ctx)
    return attn_wts, ctx


def _find_head_layers(model, target_name):
    """
    Walks back from the named target output layer to find the two unnamed
    shared-tower layers: Dense(128, relu) and Dropout.

    Both builders share the same structure:
        pool → Dense(128, relu) → Dropout → Dense(target)

    Returns:
        dense_128  — the Dense(128) layer
        dropout    — the Dropout layer immediately before the target head
    """
    target_layer = model.get_layer(target_name)
    dropout_layer = target_layer._inbound_nodes[0].input_tensors[0]._keras_history.operation
    dense_128_layer = dropout_layer._inbound_nodes[0].input_tensors[0]._keras_history.operation
    return dense_128_layer, dropout_layer


# ---------------------------------------------------------------------------
# Step 2 — GradientTape forward pass
# Step 3 — Chefer rollout + final relevancy
# ---------------------------------------------------------------------------

def compute_relevancy_scores(
    model: keras.Model,
    inputs: dict,
    target_name: str,
    num_layers: int,
) -> np.ndarray:
    """
    Compute per-token relevancy scores using Chefer et al., CVPR 2021.

    Args:
        model        — trained keras.Model from build_embedding_transformer or
                       build_general_embedding_transformer
        inputs       — same input dict you would pass to model.predict()
        target_name  — name of the output head to explain (key in model.outputs)
        num_layers   — number of transformer blocks the model was built with

    Returns:
        np.ndarray of shape (B, T) — per-token relevancy scores.
        Each row sums to ~1.0. Higher = more relevant to the prediction.
    """
    # Detect model variant
    is_general = 'latent' in inputs

    # Convert all float/int inputs to tensors (mask stays bool)
    tensor_inputs = {
        k: tf.cast(v, tf.float32) if k != 'mask' else tf.cast(v, tf.bool)
        for k, v in inputs.items()
    }

    # Retrieve head layers before entering the tape
    dense_128, dropout_layer = _find_head_layers(model, target_name)

    # ------------------------------------------------------------------
    # Step 2: Manual forward pass inside GradientTape
    # tape.watch(A_i) is called right after each A_i is produced,
    # before it flows into block remainder → ensures gradient path exists.
    # ------------------------------------------------------------------
    attn_maps = []

    with tf.GradientTape(persistent=True) as tape:

        # Pre-transformer
        if is_general:
            x, mask, inp_mask = _run_pretransformer_general(model, tensor_inputs)
        else:
            x, mask, inp_mask = _run_pretransformer_embedding(model, tensor_inputs)

        # Transformer blocks
        for i in range(num_layers):
            mha_layer = model.get_layer(f'mha_{i}')
            mha_out, A_i = mha_layer(x, x, attention_mask=mask,
                                     return_attention_scores=True, training=False)
            tape.watch(A_i)           # watch BEFORE A_i is used downstream
            attn_maps.append(A_i)

            if is_general:
                x = _run_block_remainder_general(model, i, x, mha_out)
            else:
                x = _run_block_remainder_embedding(model, i, x, mha_out)

        # Pooling
        attn_wts, ctx = _run_pooling(model, x, inp_mask)

        # Shared tower + target head
        h = dense_128(ctx, training=False)
        h = dropout_layer(h, training=False)
        pred = model.get_layer(target_name)(h, training=False)
        pred_scalar = tf.reduce_sum(pred)

    # ------------------------------------------------------------------
    # Step 3: Gradient computation + Chefer rollout
    # ------------------------------------------------------------------

    # Gradient of scalar prediction w.r.t. each attention map
    grads = [tape.gradient(pred_scalar, A) for A in attn_maps]
    del tape  # release persistent tape

    # Chefer per-layer relevancy matrices
    B = tf.shape(attn_maps[0])[0].numpy()
    T = tf.shape(attn_maps[0])[2].numpy()
    I = np.eye(T)[np.newaxis]          # (1, T, T) identity

    R = np.broadcast_to(I, (B, T, T)).copy()   # (B, T, T), start as identity

    for A, G in zip(attn_maps, grads):
        A_np = A.numpy()               # (B, h, T, T)
        G_np = G.numpy()               # (B, h, T, T)

        # Element-wise grad × attn, relu, mean over heads → (B, T, T)
        cam = np.mean(np.maximum(G_np * A_np, 0), axis=1)

        # Add identity for residual connection, normalize rows
        cam = cam + I
        row_sums = cam.sum(axis=-1, keepdims=True)
        cam = cam / np.where(row_sums == 0, 1.0, row_sums)

        R = cam @ R                    # (B, T, T) rollout

    # Weight rollout by learned attention pooling weights
    w = attn_wts.numpy()               # (B, T)
    relevancy = np.einsum('bt,btj->bj', w, R)   # (B, T)

    return relevancy


def plot_relevancy_by_position(
    relevancy: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 10,
    ax=None,
    title: str = "Average rollout importance across sequence",
    xlabel: str = "Relative position within sequence",
    ylabel: str = "Rollout relevance / uniform",
    ci_alpha: float = 0.2,
) -> plt.Axes:
    """
    Plot mean rollout relevance (normalized by uniform baseline) vs relative
    position within the valid sequence.

    Args:
        relevancy   — (B, T) float array from compute_relevancy_scores
        mask        — (B, T) bool array, True = valid token (same mask passed to model)
        n_bins      — number of equal-width bins along the relative-position axis
        ax          — existing matplotlib Axes to draw on; creates a new figure if None
        title       — plot title
        xlabel      — x-axis label
        ylabel      — y-axis label
        ci_alpha    — opacity of the ±1-std confidence band

    Returns:
        ax — the matplotlib Axes object
    """
    relevancy = np.asarray(relevancy, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Collect normalized relevance values per bin across all sequences
    bin_values = [[] for _ in range(n_bins)]

    for b in range(relevancy.shape[0]):
        valid_idx = np.where(mask[b])[0]
        T_valid = len(valid_idx)
        if T_valid == 0:
            continue

        uniform = 1.0 / T_valid
        rel_positions = np.arange(T_valid) / T_valid  # [0, 1)

        for rank, t in enumerate(valid_idx):
            rel_pos = rel_positions[rank]
            norm_rel = relevancy[b, t] / uniform
            bin_idx = min(int(rel_pos * n_bins), n_bins - 1)
            bin_values[bin_idx].append(norm_rel)

    means = np.array([np.mean(v) if v else np.nan for v in bin_values])
    stds  = np.array([np.std(v)  if v else np.nan for v in bin_values])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(bin_centers, means - stds, means + stds,
                    alpha=ci_alpha, color="steelblue")
    ax.plot(bin_centers, means, marker="o", color="steelblue",
            label="Mean rollout / uniform")
    ax.axhline(1.0, linestyle="--", color="steelblue", alpha=0.6,
               label="Uniform baseline")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return ax
