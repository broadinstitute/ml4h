# TensorMap Reference

A `TensorMap` is the central schema object in ml4h. It describes a single input or output signal—its shape, semantics, normalization, loss function, and how to read it from an HDF5 file. Every model in the toolkit is assembled from a list of `TensorMap` instances.

---

## Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | **required** | Unique identifier; also determines the HDF5 key path used by the default reader. |
| `interpretation` | `Interpretation` | `CONTINUOUS` | Semantic type (see below). Controls automatic inference of `loss`, `activation`, and `metrics`. |
| `shape` | `tuple[int, ...]` | inferred | Tensor shape **excluding** the batch dimension. For categorical maps, inferred from `channel_map` if omitted. |
| `channel_map` | `dict[str, int]` | `None` | Maps label strings to channel indices. Required for `CATEGORICAL` maps; also used by `TIME_TO_EVENT`. |
| `normalization` | `Normalizer` or `dict` | `None` | Applied after reading the raw array. Pass a `Normalizer` subclass (e.g. `Standardize(mean, std)`) or a legacy dict `{'mean': m, 'std': s}`. |
| `loss` | `str` or `Callable` | inferred | Keras loss. Inferred from `interpretation` when omitted (e.g. `'mse'` for continuous, `'categorical_crossentropy'` for categorical). |
| `loss_weight` | `float` | `1.0` | Relative weight of this output's loss in multitask objectives. |
| `activation` | `str` or `Callable` | inferred | Output activation. Inferred from `interpretation` (`'linear'`, `'softmax'`, `'sigmoid'`). |
| `metrics` | `list` | inferred | Keras metrics. Inferred from `interpretation`; pass an explicit list to override. |
| `tensor_from_file` | `Callable(tm, hd5, dependents) -> np.ndarray` | `None` | Custom reader. When `None`, the default reader uses `name` and `path_prefix` to locate the dataset in the HDF5 file. |
| `path_prefix` | `str` | `None` | HDF5 group prefix prepended to `name` when building the lookup key. |
| `sentinel` | `float` | `None` | A value that signals "missing data". When set on a `CONTINUOUS` map, the loss is automatically replaced with a sentinel-masked logcosh. |
| `validator` | `Callable(tm, tensor, hd5) -> None` | no-op | Called on each tensor; raise `ValueError` to skip a sample. See `make_range_validator` and `no_nans` helpers. |
| `augmentations` | `list[Callable]` | `None` | Shape-preserving transforms applied only at training time. |
| `cacheable` | `bool` | `True` | Whether tensors built by this map may be stored in the worker cache. Disable when `tensor_from_file` contains randomness. |
| `annotation_units` | `int` | `32` | Embedding dimension for unstructured / text maps. |
| `model` | `keras.Model` | `None` | Encoder model used when `interpretation=EMBEDDING`. |
| `parents` | `list[TensorMap]` | `None` | Upstream maps whose tensors must be constructed before this one. Used by `EMBEDDING` maps. |
| `dependent_map` | `TensorMap` | `None` | A map that is determined by this one (e.g. the next-character target for a language map). |
| `days_window` | `int` | `1825` | Maximum follow-up window in days for `SURVIVAL_CURVE` maps. |
| `storage_type` | `StorageType` | `None` | Hint for how categorical values are stored on disk (`CATEGORICAL_INDEX`, `CATEGORICAL_FLAG`, etc.). |
| `time_series_limit` | `int` | `None` | Maximum number of tensors to draw from a time-series group. Activates dynamic shaping (prepends a `None` dimension). |
| `time_series_order` | `TimeSeriesOrder` | `NEWEST` | Which end of the time series to prefer (`NEWEST`, `OLDEST`, `RANDOM`). |
| `time_series_lookup` | `dict[int, tuple]` | `None` | Maps integer offsets to date-range tuples for filtering a time series. |
| `discretization_bounds` | `list[float]` | `None` | Bin edges. When set, the map reads a continuous value and one-hot encodes it into `len(bounds)+1` bins. |

---

## Interpretation Enum

| Value | Typical use | Auto loss | Auto activation |
|---|---|---|---|
| `CONTINUOUS` | Scalars, raw waveforms, pixel arrays | `mse` (or sentinel logcosh) | `linear` |
| `CATEGORICAL` | Disease labels, segmentation masks | `categorical_crossentropy` / `binary_crossentropy` | `softmax` |
| `DISCRETIZED` | Continuous value binned at runtime | same as `CATEGORICAL` | `softmax` |
| `TIME_TO_EVENT` | Cox regression targets | `cox_hazard_loss` | `sigmoid` |
| `SURVIVAL_CURVE` | Discrete survival curve | `survival_likelihood_loss` | `sigmoid` |
| `LANGUAGE` | Character-level sequences | `categorical_crossentropy` | `softmax` |
| `EMBEDDING` | Latent vector from a sub-model | `mse` | `linear` |
| `MESH` | 3-D mesh data | `mse` | `linear` |
| `TEXT` | Compressed text blobs | (custom) | (custom) |

---

## Normalization

Normalization is applied inside `postprocess_tensor` after `tensor_from_file` returns and before the tensor is placed into the batch array. The inverse (`rescale`) is used when logging human-readable stats.

```python
from ml4h.normalizer import Standardize, ZeroMeanStd1

# Normalize ECG voltage using UK Biobank population statistics
ecg_map = TensorMap(
    'ecg_rest_raw',
    shape=(5000, 12),
    normalization=Standardize(mean=0.0, std=1000.0),  # raw values in µV
)
```

> **Note on Issue #609 (small output values):** If a model trained on UK Biobank data (ECG in µV, σ ≈ 1000) is applied to a cohort where ECG amplitudes are in mV (σ ≈ 1), the `Standardize` parameters in the TensorMap will produce outputs scaled down by ~1000. Always match `mean`/`std` to the units of the target cohort.

---

## HDF5 Key Resolution

The default `tensor_from_file` looks up tensors using:

```
/{path_prefix}/{name}/   if path_prefix is set
/{name}/                 otherwise
```

The first dataset found inside that group is returned. Override `tensor_from_file` for non-standard layouts:

```python
def my_reader(tm, hd5, dependents={}):
    return np.array(hd5['ecg/lead_i/voltage'])

ecg_lead_i = TensorMap('lead_i', shape=(5000,), tensor_from_file=my_reader)
```

---

## Common Patterns

### Binary classification output
```python
from ml4h.TensorMap import TensorMap, Interpretation

af_label = TensorMap(
    'atrial_fibrillation',
    interpretation=Interpretation.CATEGORICAL,
    channel_map={'no_af': 0, 'af': 1},
    # shape is inferred as (2,); loss as 'binary_crossentropy'
)
```

### Continuous output with sentinel for missing values
```python
lvm = TensorMap(
    'lv_mass',
    shape=(1,),
    sentinel=-1.0,
    normalization=Standardize(mean=150.0, std=40.0),
    # loss is auto-set to sentinel_logcosh_loss(-1.0)
)
```

### Time-series input (dynamic batch)
```python
ecg_series = TensorMap(
    'ecg_rest',
    shape=(5000, 12),
    path_prefix='ecg',
    time_series_limit=3,
    time_series_order=TimeSeriesOrder.NEWEST,
    # effective shape: (None, 5000, 12)
)
```

### Custom loss weight in multitask training
```python
# Upweight a rare-event output relative to the reconstruction loss
rare_event = TensorMap(
    'rare_phenotype',
    interpretation=Interpretation.CATEGORICAL,
    channel_map={'no': 0, 'yes': 1},
    loss_weight=5.0,
)
```

---

## Validator Helpers

```python
from ml4h.TensorMap import make_range_validator, no_nans

# Reject samples where heart rate is outside a plausible range
hr_map = TensorMap(
    'heart_rate',
    shape=(1,),
    validator=make_range_validator(20, 300),
)

# Reject samples containing NaN
ecg_map = TensorMap('ecg', shape=(5000, 12), validator=no_nans)
```
