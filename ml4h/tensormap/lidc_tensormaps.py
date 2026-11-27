"""Lazy CT readers and ml4h TensorMap factories for lidc_to_hd5.py output.

ml4h is imported only in the factory; the readers also work independently.
Volumes use DHW; image/volume TensorMaps add a trailing singleton channel.
"""
from __future__ import annotations

from functools import partial
import json
import warnings
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import zoom


def _string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def select_series(hd5: h5py.File, selection: str = "primary") -> h5py.Group:
    if not bool(hd5.attrs.get("conversion_complete", False)):
        raise ValueError(f"Incomplete conversion in {hd5.filename}; inspect conversion_report.json")
    series = hd5["series"]
    keys = sorted(series.keys())
    if not keys:
        raise ValueError(f"No CT series in {hd5.filename}")
    if selection == "primary":
        uid = _string(hd5.attrs["primary_series_uid"])
    elif selection == "random":
        uid = keys[int(np.random.randint(len(keys)))]
    else:
        uid = selection
    if uid not in series:
        raise KeyError(f"Series {uid!r} not found in {hd5.filename}")
    return series[uid]


def load_json(dataset: h5py.Dataset) -> Any:
    return json.loads(dataset[:].tobytes().decode("utf-8"))


def read_xml_metadata(hd5: h5py.File, series_selection: str = "primary",
                      document_id: str | None = None, normalized: bool = True) -> Any:
    group = select_series(hd5, series_selection)
    if document_id is None:
        if bool(group.attrs.get("preferred_xml_ambiguous", False)):
            raise ValueError("Distinct XML variants exist; choose a document_id explicitly")
        if "preferred_xml_document" not in group.attrs:
            raise KeyError("This CT series has no XML annotations")
        document_id = _string(group.attrs["preferred_xml_document"])
    name = "annotations_json" if normalized else "parsed_json"
    return load_json(group["xml"][document_id][name])


def read_native_volume(hd5: h5py.File, series_selection: str = "primary") -> np.ndarray:
    """Return the complete native DHW array in HU, without resizing/windowing."""
    return np.asarray(select_series(hd5, series_selection)["hu"][()], dtype=np.float32)


def _resize(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if array.shape == shape:
        return array.astype(np.float32, copy=False)
    # This is interpolation in voxel-index coordinates, NOT physical-mm resampling.
    factors = tuple(target / current for target, current in zip(shape, array.shape))
    resized = zoom(array, factors, order=1, mode="nearest", prefilter=False)
    if resized.shape != shape:
        raise ValueError(f"Unexpected resized shape {resized.shape}; expected {shape}")
    return resized.astype(np.float32, copy=False)


def _window(array: np.ndarray, window: tuple[float, float] | None,
            output_range: tuple[float, float]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if window is None:
        return array
    low, high = window
    if not low < high or not output_range[0] < output_range[1]:
        raise ValueError("Window and output_range must each have increasing bounds")
    array = (np.clip(array, low, high) - low) / (high - low)
    return (array * (output_range[1] - output_range[0]) + output_range[0]).astype(np.float32)


def ct_random_slice_from_hd5(tm, hd5: h5py.File, dependents=None, *,
                            series_selection: str = "primary", slice_selection: str = "random",
                            window: tuple[float, float] | None = (-1000.0, 400.0),
                            output_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Load only one HDF5 slice chunk; return HWC with one channel."""
    if len(tm.shape) != 3 or tm.shape[-1] != 1:
        raise ValueError("A slice TensorMap must have shape (height, width, 1)")
    dataset = select_series(hd5, series_selection)["hu"]
    if slice_selection == "random":
        index = int(np.random.randint(dataset.shape[0]))
    elif slice_selection == "middle":
        index = dataset.shape[0] // 2
    else:
        raise ValueError("slice_selection must be 'random' or 'middle'")
    image = np.asarray(dataset[index], dtype=np.float32)
    image = _resize(image, tuple(tm.shape[:2]))
    return _window(image, window, output_range)[..., np.newaxis]


def ct_volume_from_hd5(tm, hd5: h5py.File, dependents=None, *,
                       series_selection: str = "primary",
                       window: tuple[float, float] | None = (-1000.0, 400.0),
                       output_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Read the whole scan and resize the entire DHW field to a fixed DHWC shape."""
    if len(tm.shape) != 4 or tm.shape[-1] != 1:
        raise ValueError("A volume TensorMap must have shape (depth, height, width, 1)")
    volume = read_native_volume(hd5, series_selection)
    volume = _resize(volume, tuple(tm.shape[:3]))
    return _window(volume, window, output_range)[..., np.newaxis]


def make_lidc_tensormap(name: str, *, mode: str = "slice", shape: tuple[int, ...] | None = None,
                        series_selection: str = "primary", slice_selection: str = "random",
                        window: tuple[float, float] | None = (-1000.0, 400.0),
                        output_range: tuple[float, float] = (0.0, 1.0)):
    from ml4h.TensorMap import Interpretation, TensorMap

    if mode not in ("slice", "volume"):
        raise ValueError("mode must be 'slice' or 'volume'")
    shape = shape or ((256, 256, 1) if mode == "slice" else (128, 128, 128, 1))
    expected_rank = 3 if mode == "slice" else 4
    if (len(shape) != expected_rank or shape[-1] != 1
            or any(not isinstance(n, int) or n < 1 for n in shape)):
        raise ValueError("Provide a fixed positive HWC / DHWC shape with one channel")
    kwargs = dict(series_selection=series_selection, window=window, output_range=output_range)
    if mode == "slice":
        reader = partial(ct_random_slice_from_hd5, slice_selection=slice_selection, **kwargs)
    else:
        reader = partial(ct_volume_from_hd5, **kwargs)
    # Caching would freeze random selection; volumes can also overwhelm a tensor cache.
    return TensorMap(name, interpretation=Interpretation.CONTINUOUS, shape=shape,
                     tensor_from_file=reader, cacheable=False)


def _mask_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    shape = tuple(shape)
    if (len(shape) != 4 or shape[-1] != 1
            or any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_))
                   or n < 1 for n in shape)):
        raise ValueError("A malignancy mask must have fixed positive shape (depth, height, width, 1)")
    return tuple(int(n) for n in shape)


def _rated_contours_by_slice(annotations: list[dict], native_shape: tuple[int, ...],
                             skip_unmatched: bool) -> dict:
    """Index each rated reader/nodule's contours by the native SOP-linked slice."""
    depth, height, width = native_shape
    indexed: dict[int, list] = {}
    unmatched = 0
    for annotation in annotations:
        if annotation.get("kind") != "nodule":
            continue
        rating = annotation.get("characteristics", {}).get("malignancy")
        if rating is None or str(rating).strip() == "":
            continue  # Unrated small nodules do not have a malignancy contribution.
        context = f"annotation {annotation.get('annotation_id')!r}"
        try:
            score = float(rating)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid malignancy {rating!r} in {context}") from error
        if score not in (1, 2, 3, 4, 5):
            raise ValueError(f"Malignancy must be an integer from 1 to 5 in {context}")
        reader = annotation.get("session_index")
        if not isinstance(reader, int) or isinstance(reader, bool) or reader < 0:
            raise ValueError(f"Missing/invalid session_index in {context}")
        per_slice: dict[int, list] = {}
        for roi in annotation.get("rois", []):
            z = roi.get("slice_index")
            if z is None:
                if skip_unmatched:
                    unmatched += 1
                    continue
                raise ValueError(
                    f"Unmatched ROI SOP UID {roi.get('image_sop_uid')!r} in {context}; "
                    "no native slice_index. Fix the XML link or explicitly set skip_unmatched=True."
                )
            if not isinstance(z, int) or isinstance(z, bool) or not 0 <= z < depth:
                raise ValueError(f"Invalid native slice_index {z!r} in {context}")
            inclusion = roi.get("inclusion")
            if not isinstance(inclusion, bool):
                raise ValueError(f"Missing/invalid ROI inclusion flag in {context}")
            points = np.asarray(roi.get("points_xy", []), dtype=np.float64)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
                    or not np.isfinite(points).all()):
                raise ValueError(f"Invalid or empty points_xy in {context}")
            rounded = np.rint(points)
            if not np.allclose(points, rounded, atol=1e-6, rtol=0):
                raise ValueError(f"Expected native integer pixel coordinates in {context}")
            if (np.any(rounded[:, 0] < 0) or np.any(rounded[:, 0] >= width)
                    or np.any(rounded[:, 1] < 0) or np.any(rounded[:, 1] >= height)):
                raise ValueError(f"Contour coordinates outside native image bounds in {context}")
            xy = [tuple(map(int, point)) for point in rounded]
            per_slice.setdefault(z, []).append((inclusion, xy))
        for z, contours in per_slice.items():
            indexed.setdefault(z, []).append((reader, score, contours))
    if unmatched:
        warnings.warn(f"Skipped {unmatched} rated ROIs without a SOP-linked slice_index",
                      RuntimeWarning, stacklevel=3)
    return indexed


def _reader_summed_malignancy_slice(lesions: list, shape: tuple[int, int]) -> np.ndarray:
    # Pillow is imported only for masks; importing the CT readers still needs no TensorFlow.
    from PIL import Image, ImageDraw

    height, width = shape
    reader_masks = {}
    for reader, score, contours in lesions:
        included = Image.new("1", (width, height), 0)
        excluded = Image.new("1", (width, height), 0)
        include_draw, exclude_draw = ImageDraw.Draw(included), ImageDraw.Draw(excluded)
        for inclusion, xy in contours:
            draw = include_draw if inclusion else exclude_draw
            # Native XML also contains degenerate one/two-point end contours.
            if len(xy) == 1:
                draw.point(xy, fill=1)
            elif len(xy) == 2:
                draw.line(xy, fill=1, width=1)
            else:
                draw.polygon(xy, fill=1)
        lesion_mask = np.asarray(included, dtype=bool) & ~np.asarray(excluded, dtype=bool)
        if reader not in reader_masks:
            reader_masks[reader] = np.zeros(shape, dtype=np.float32)
        # One contribution per reader/voxel, even when that reader's nodules overlap.
        np.maximum(reader_masks[reader], score, out=reader_masks[reader], where=lesion_mask)
    summed = np.zeros(shape, dtype=np.float32)
    for reader_mask in reader_masks.values():
        summed += reader_mask
    return summed


def ct_malignancy_mask_from_hd5(tm, hd5: h5py.File, dependents=None, *,
                               series_selection: str = "primary",
                               document_id: str | None = None,
                               skip_unmatched: bool = False) -> np.ndarray:
    """Return an unnormalized float32 DHWC mask summed over reader sessions.

    At each native voxel, each reader contributes the highest malignancy of
    their rated nodule contours covering that voxel. Different readers add;
    five readers assigning 4 to the same voxel therefore yield 20.

    Inclusion contours are filled, including their boundary pixels; exclusion
    contours are subtracted within the same nodule before reader aggregation.
    Non-nodules and nodules with missing malignancy are omitted. No consensus
    threshold, averaging, z interpolation, or cross-reader nodule matching is used.

    The output uses the same voxel-index grid as ct_volume_from_hd5, but nearest
    interpolation preserves the integer-valued scores. Only the native slices
    needed by that grid are rasterized; neither CT pixels nor a full native
    mask volume are loaded. Use the same fixed series selection for image/mask
    TensorMaps. Unmatched scored ROIs raise unless explicitly opted out.
    """
    output_shape = _mask_shape(tm.shape)
    group = select_series(hd5, series_selection)
    native_shape = tuple(group["hu"].shape)
    if len(native_shape) != 3 or min(native_shape) < 1:
        raise ValueError("Expected a nonempty native DHW CT volume")
    # Resolve a random series at most once, so metadata and shape refer to the same scan.
    selected_uid = group.name.rsplit("/", 1)[-1]
    annotations = read_xml_metadata(hd5, series_selection=selected_uid, document_id=document_id)
    indexed = _rated_contours_by_slice(annotations, native_shape, skip_unmatched)
    output = np.zeros(output_shape, dtype=np.float32)
    if not indexed:
        return output

    # This is exactly the depth sampling used by a 3-D order=0 scipy zoom with
    # the same factors as the image reader (default grid_mode=False).
    source_z = zoom(np.arange(native_shape[0], dtype=np.int32),
                    (output_shape[0] / native_shape[0],),
                    order=0, mode="nearest", prefilter=False)
    if source_z.shape != (output_shape[0],):
        raise ValueError("Unexpected nearest-neighbor depth sampling shape")
    plane_shape = output_shape[1:3]
    for z in np.unique(source_z):
        if int(z) not in indexed:
            continue
        plane = _reader_summed_malignancy_slice(indexed[int(z)], native_shape[1:])
        if plane.shape != plane_shape:
            plane = zoom(plane, tuple(target / current for target, current
                                     in zip(plane_shape, plane.shape)),
                         order=0, mode="nearest", prefilter=False)
        if plane.shape != plane_shape:
            raise ValueError(f"Unexpected resized mask shape {plane.shape}; expected {plane_shape}")
        output[source_z == z, :, :, 0] = plane
    return output


def make_lidc_malignancy_mask_tensormap(name: str, *,
                                       shape: tuple[int, ...] = (128, 128, 128, 1),
                                       series_selection: str = "primary",
                                       document_id: str | None = None,
                                       skip_unmatched: bool = False):
    """Create a continuous volume TensorMap for the unnormalized reader sum.

    Set shape and fixed series_selection to match the CT volume TensorMap.
    document_id chooses one XML variant, never multiple copies of the readers.
    """
    from ml4h.TensorMap import Interpretation, TensorMap

    shape = _mask_shape(shape)
    reader = partial(ct_malignancy_mask_from_hd5, series_selection=series_selection,
                     document_id=document_id, skip_unmatched=skip_unmatched)
    return TensorMap(name, interpretation=Interpretation.CONTINUOUS, shape=shape,
                     tensor_from_file=reader, cacheable=False)


lidc_random_slice_256 = make_lidc_tensormap(
    "lidc_random_slice",
    mode="slice",
    shape=(256, 256, 1),
    series_selection="primary",   # Use "random" to also randomize the CT series.
    slice_selection="random",
    window=(-1000.0, 400.0),       # HU bounds, mapped to [0, 1].
    output_range=(0.0, 1.0),
)