from itertools import product

from ml4cvd.TensorMap import TensorMap, Interpretation


CONTINUOUS_TMAPS = [
    TensorMap(f'{n}d_cont', shape=tuple(range(1, n + 1)), interpretation=Interpretation.CONTINUOUS)
    for n in range(1, 6)
]
CATEGORICAL_TMAPS = [
    TensorMap(
        f'{n}d_cat', shape=tuple(range(1, n + 1)),
        interpretation=Interpretation.CATEGORICAL,
        channel_map={f'c_{i}': i for i in range(n)},
    )
    for n in range(1, 6)
]
TMAPS_UP_TO_4D = CONTINUOUS_TMAPS[:-1] + CATEGORICAL_TMAPS[:-1]
TMAPS_5D = CONTINUOUS_TMAPS[-1:] + CATEGORICAL_TMAPS[-1:]
MULTIMODAL_UP_TO_4D = [list(x) for x in product(CONTINUOUS_TMAPS[:-1], CATEGORICAL_TMAPS[:-1])]


TMAPS = {
    tmap.name: tmap
    for tmap in CONTINUOUS_TMAPS + CATEGORICAL_TMAPS
}
