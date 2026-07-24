"""Visual + coverage test for the echo video augmentations.

Generates a few structured random clips, applies every augmentation (forcing each
discrete branch so *all permutations* are exercised), prints a coverage report,
and renders a self-contained HTML file with animated players showing the original
clip next to each labeled augmentation.

Run from the DROID directory:

    python test_transforms.py                       # -> transforms_preview.html
    python test_transforms.py --output /tmp/aug.html --videos 3 --frames 16

Requires tensorflow (the transforms use native TF ops) and numpy.
"""

import argparse
import base64
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_descriptions.transforms import (  # noqa: E402
    AUGMENTATIONS,
    RandomBrightnessContrast,
    RandomFlip,
    RandomGaussianNoise,
    RandomJitterRotate,
    RandomSectorMask,
)


def make_random_video(rng, n_frames, height, width, channels=3):
    """A structured-random clip so augmentations are visually obvious.

    Contains: a channel-dependent gradient background (so per-channel color ops
    show up), a fixed bright square in the top-left corner (an orientation marker
    for flips/rotation), and a few moving colored blobs (temporal structure so
    per-frame effects like noise and per-frame masking are visible).
    """
    video = np.zeros((n_frames, height, width, channels), np.float32)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    for c in range(channels):
        video[:, :, :, c] = 0.12 + 0.30 * (xx / width) * (c + 1) / channels + 0.15 * (yy / height)

    # Orientation marker: bright square in the top-left corner (asymmetric).
    video[:, 2:height // 4, 2:width // 4, :] = 1.0

    # Moving blobs.
    for _ in range(int(rng.integers(2, 4))):
        color = rng.uniform(0.4, 1.0, size=channels).astype(np.float32)
        radius = int(rng.integers(max(4, height // 10), max(6, height // 5)))
        x0 = rng.uniform(radius, width - radius)
        y0 = rng.uniform(radius, height - radius)
        vx = rng.uniform(-width * 0.03, width * 0.03)
        vy = rng.uniform(-height * 0.03, height * 0.03)
        for t in range(n_frames):
            cx, cy = x0 + vx * t, y0 + vy * t
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            video[t][mask] = color

    return np.clip(video, 0.0, 1.0)


def build_cases():
    """All augmentation permutations, each a fresh callable (or list) + description.

    Probabilities are forced to 0/1 so every discrete branch is deterministically
    exercised. Each case is tagged with the augmentation key(s) it covers so the
    coverage check below can confirm nothing is missed.
    """
    return [
        {
            'label': 'Jitter + rotation',
            'desc': 'Same X/Y shift (<=20%) and rotation (+/-90 deg) applied to every frame; black fill.',
            'covers': ['jitter_rotate'],
            'build': lambda: RandomJitterRotate(p=1.0),
        },
        {
            'label': 'Sector mask (same each frame)',
            'desc': 'One random rectangle (<=0.25x by 0.25y) zeroed in the SAME place in every frame (static).',
            'covers': ['sector_mask'],
            'build': lambda: RandomSectorMask(same_across_frames_prob=1.0, p=1.0),
        },
        {
            'label': 'Sector mask (different each frame)',
            'desc': 'A fresh random rectangle zeroed per frame (mask flickers across the clip).',
            'covers': ['sector_mask'],
            'build': lambda: RandomSectorMask(same_across_frames_prob=0.0, p=1.0),
        },
        {
            'label': 'Flip horizontal',
            'desc': 'Left-right flip applied to the whole clip (corner marker moves to the right).',
            'covers': ['flip'],
            'build': lambda: RandomFlip(horizontal_prob=1.0, vertical_prob=0.0),
        },
        {
            'label': 'Flip vertical',
            'desc': 'Up-down flip applied to the whole clip (corner marker moves to the bottom).',
            'covers': ['flip'],
            'build': lambda: RandomFlip(horizontal_prob=0.0, vertical_prob=1.0),
        },
        {
            'label': 'Flip both',
            'desc': 'Horizontal + vertical flip (corner marker moves to the bottom-right).',
            'covers': ['flip'],
            'build': lambda: RandomFlip(horizontal_prob=1.0, vertical_prob=1.0),
        },
        {
            'label': 'Gaussian noise',
            'desc': 'Per-frame noise, scale <=20% of frame max, same scale for all channels (fresh noise each frame).',
            'covers': ['gaussian_noise'],
            'build': lambda: RandomGaussianNoise(p=1.0),
        },
        {
            'label': 'Brightness (shared channels)',
            'desc': 'One brightness delta ~ N(0, 0.2) added to all channels; constant across frames.',
            'covers': ['brightness_contrast'],
            'build': lambda: RandomBrightnessContrast(
                brightness_prob=1.0, contrast_prob=0.0, per_channel_prob=0.0),
        },
        {
            'label': 'Brightness (per channel)',
            'desc': 'Independent brightness delta per channel (color shift); constant across frames.',
            'covers': ['brightness_contrast'],
            'build': lambda: RandomBrightnessContrast(
                brightness_prob=1.0, contrast_prob=0.0, per_channel_prob=1.0),
        },
        {
            'label': 'Contrast (shared channels)',
            'desc': 'One contrast factor (1 + N(0, 0.2)) for all channels; constant across frames.',
            'covers': ['brightness_contrast'],
            'build': lambda: RandomBrightnessContrast(
                brightness_prob=0.0, contrast_prob=1.0, per_channel_prob=0.0),
        },
        {
            'label': 'Contrast (per channel)',
            'desc': 'Independent contrast factor per channel; constant across frames.',
            'covers': ['brightness_contrast'],
            'build': lambda: RandomBrightnessContrast(
                brightness_prob=0.0, contrast_prob=1.0, per_channel_prob=1.0),
        },
        {
            'label': 'Brightness + contrast (per channel)',
            'desc': 'Both brightness and contrast adjusted independently per channel.',
            'covers': ['brightness_contrast'],
            'build': lambda: RandomBrightnessContrast(
                brightness_prob=1.0, contrast_prob=1.0, per_channel_prob=1.0),
        },
        {
            'label': 'All combined (random)',
            'desc': 'Jitter+rotate -> flip -> sector mask -> noise -> brightness/contrast, each with its own randomness.',
            'covers': ['jitter_rotate', 'flip', 'sector_mask', 'gaussian_noise', 'brightness_contrast'],
            'build': lambda: [
                RandomJitterRotate(p=1.0),
                RandomFlip(),
                RandomSectorMask(p=1.0),
                RandomGaussianNoise(p=1.0),
                RandomBrightnessContrast(brightness_prob=1.0, contrast_prob=1.0),
            ],
        },
    ]


def apply_case(video, transform):
    """Apply a single transform or a list of transforms to a clip."""
    transforms = transform if isinstance(transform, list) else [transform]
    out = video
    for t in transforms:
        out = t(out)
    return np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)


def frames_to_datauris(clip):
    """Encode an (T, H, W, C) float clip in [0, 1] as a list of base64 PNG data URIs."""
    u8 = (clip * 255.0).round().astype(np.uint8)
    uris = []
    for frame in u8:
        png = tf.io.encode_png(frame).numpy()
        uris.append('data:image/png;base64,' + base64.b64encode(png).decode('ascii'))
    return uris


def check_coverage(cases):
    """Confirm every augmentation in the registry is exercised; print a report."""
    covered = set()
    for case in cases:
        covered.update(case['covers'])
    expected = set(AUGMENTATIONS)
    missing = expected - covered
    print('Coverage report')
    print('-' * 60)
    for key in sorted(expected):
        n = sum(1 for c in cases if key in c['covers'])
        print(f'  {key:20s} covered by {n} case(s)')
    print('-' * 60)
    if missing:
        raise AssertionError(f'Augmentations NOT covered: {sorted(missing)}')
    print(f'All {len(expected)} augmentations covered across {len(cases)} cases.\n')


CARD_TEMPLATE = """
      <div class="card{extra}">
        <div class="label">{label}</div>
        <img id="{pid}" class="player" alt="{label}">
        <div class="desc">{desc}</div>
      </div>"""


def render_html(sections, players, fps):
    interval = int(1000 / fps)
    players_json = json.dumps(players)
    cards_html = []
    for section in sections:
        cards = ''.join(section['cards'])
        cards_html.append(
            f'    <section>\n      <h2>{section["title"]}</h2>\n'
            f'      <div class="row">{cards}\n      </div>\n    </section>'
        )
    body = '\n'.join(cards_html)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Echo video augmentation preview</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #111; color: #eee; margin: 24px; }}
  h1 {{ font-weight: 600; }}
  h2 {{ font-weight: 500; color: #9cf; border-bottom: 1px solid #333; padding-bottom: 6px; margin-top: 32px; }}
  .row {{ display: flex; flex-wrap: wrap; gap: 16px; }}
  .card {{ width: 160px; background: #1c1c1c; border: 1px solid #2c2c2c; border-radius: 8px; padding: 8px; }}
  .card.original {{ border-color: #4c8; }}
  .label {{ font-size: 13px; font-weight: 600; margin-bottom: 6px; min-height: 32px; }}
  .card.original .label {{ color: #6e6; }}
  .player {{ width: 144px; height: 144px; image-rendering: pixelated; background: #000; border-radius: 4px; }}
  .desc {{ font-size: 11px; color: #999; margin-top: 6px; line-height: 1.35; }}
</style>
</head>
<body>
  <h1>Echo video augmentation preview</h1>
  <p>Each clip plays as an animation ({fps} fps). The green-bordered card is the
  unaugmented original; every other card is the labeled augmentation applied to
  that same clip.</p>
{body}
<script>
  const players = {players_json};
  for (const [id, frames] of Object.entries(players)) {{
    const img = document.getElementById(id);
    let i = 0;
    img.src = frames[0];
    setInterval(() => {{ i = (i + 1) % frames.length; img.src = frames[i]; }}, {interval});
  }}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='transforms_preview.html')
    parser.add_argument('--videos', type=int, default=2, help='number of base clips')
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--width', type=int, default=120)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cases = build_cases()
    check_coverage(cases)

    players = {}
    sections = []
    for v in range(args.videos):
        base = make_random_video(rng, args.frames, args.height, args.width)

        cards = []
        pid = f'v{v}_orig'
        players[pid] = frames_to_datauris(base)
        cards.append(CARD_TEMPLATE.format(
            extra=' original', label='Original', pid=pid,
            desc='Unaugmented synthetic clip.'))

        for j, case in enumerate(cases):
            out = apply_case(base, case['build']())
            pid = f'v{v}_c{j}'
            players[pid] = frames_to_datauris(out)
            cards.append(CARD_TEMPLATE.format(
                extra='', label=case['label'], pid=pid, desc=case['desc']))
            print(f'  clip {v}: applied "{case["label"]}"')

        sections.append({'title': f'Base clip #{v}', 'cards': cards})

    html = render_html(sections, players, args.fps)
    out_path = os.path.abspath(args.output)
    with open(out_path, 'w') as f:
        f.write(html)

    total_players = len(players)
    print(f'\nWrote {out_path} ({total_players} animated clips, '
          f'{args.videos} base + {args.videos * len(cases)} augmented).')


if __name__ == '__main__':
    main()
