"""Convert plot_predictions prediction PNGs into label-map mask PNGs for tensorize_pngs.

The '_prediction_' PNGs written by plot_predictions are plasma-colormapped visualizations
of the argmax over the model's output channels, normalized per image. This script inverts
the colormap: each pixel color is matched to the plasma colormap, the per-image
normalization factor is recovered, and the resulting channel indices are remapped to the
raw label values expected by the segmentation tensormap (which drops kidney=6 and merges
body=1 into background=0 on load). Masks are written with label values in the red channel
as <output_folder>/<sample_id>_t1map.png.mask.png, plus a manifest.tsv, ready for
./scripts/tensorize.sh -m tensorize_pngs

Example:

    ./scripts/tf.sh -c model_zoo/t1_time_from_segmented_regions/convert_prediction_pngs_to_masks.py \
        --predictions $HOME/demo/outputs/plot_predictions/public_model_infer/prediction_pngs/ \
        --output_folder $HOME/demo/pseudo_gt/
"""
import os
import csv
import glob
import argparse

import imageio
import numpy as np
from matplotlib import cm

# Model output channel -> raw label value expected by the segmentation tensormap
CHANNEL_TO_RAW_LABEL = np.array([0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13], dtype=np.uint8)
NUM_CHANNELS = len(CHANNEL_TO_RAW_LABEL)
COLORMAP = cm.get_cmap('plasma')  # must match the cmap used in predictions_to_pngs


def invert_colormap(png_path):
    """Recover integer channel indices from a colormapped prediction PNG."""
    img = imageio.imread(png_path)
    rgb = img[..., :3].astype(np.float64) / np.iinfo(img.dtype).max
    lut = COLORMAP(np.linspace(0.0, 1.0, 256))[:, :3]

    flat = rgb.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    if len(unique_colors) == 1:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    # nearest colormap entry for each distinct color -> normalized values in [0, 1]
    distances = ((unique_colors[:, None, :] - lut[None, :, :]) ** 2).sum(axis=-1)
    normalized = np.argmin(distances, axis=1) / 255.0

    # plt.imsave scaled channel indices by the per-image max: find the integer scale
    # that maps every distinct color back onto a distinct integer channel index
    best_k, best_error = None, np.inf
    for k in range(1, NUM_CHANNELS):
        candidates = normalized * k
        rounded = np.round(candidates)
        if len(np.unique(rounded)) < len(rounded):
            continue
        error = np.abs(candidates - rounded).max()
        if error < best_error:
            best_k, best_error = k, error
    if best_k is None or best_error > 0.25:
        raise ValueError(f'Could not invert colormap for {png_path}: unreliable normalization (error {best_error:.3f})')

    channels = np.round(normalized * best_k).astype(np.uint8)
    return channels[inverse].reshape(img.shape[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, help="Folder of '_prediction_' PNGs from plot_predictions.")
    parser.add_argument('--output_folder', required=True, help='Folder to write mask PNGs.')
    parser.add_argument('--instance_number', default='1', help='Annotation instance suffix for the hd5 key.')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    prediction_pngs = sorted(glob.glob(os.path.join(args.predictions, '*_prediction_*.png')))
    if not prediction_pngs:
        raise ValueError(f'No prediction PNGs found in {args.predictions}')

    for png_path in prediction_pngs:
        sample_id = os.path.basename(png_path).split('_')[0]
        channels = invert_colormap(png_path)
        label_map = CHANNEL_TO_RAW_LABEL[channels]

        mask = np.zeros(label_map.shape + (3,), dtype=np.uint8)
        mask[..., 0] = label_map
        dicom_file = f'{sample_id}_t1map'
        mask_path = os.path.join(args.output_folder, f'{dicom_file}.png.mask.png')
        imageio.imwrite(mask_path, mask)
        print(f'Wrote {mask_path} with raw labels {sorted(np.unique(label_map).tolist())}')


if __name__ == '__main__':
    main()
