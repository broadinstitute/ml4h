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
SEGMENT_IN = TensorMap(f'2d_for_segment_in', shape=(32, 32, 1), interpretation=Interpretation.CONTINUOUS, metrics=['mse'])
SEGMENT_OUT = TensorMap(f'2d_for_segment_out', shape=(32, 32, 2), interpretation=Interpretation.CATEGORICAL, channel_map={'yes': 0, 'no': 1})


TMAPS = {
    tmap.name: tmap
    for tmap in CONTINUOUS_TMAPS + CATEGORICAL_TMAPS
}
PARENT_TMAPS = [
    TensorMap(f'parent_test_{i}', shape=(1,), interpretation=Interpretation.CONTINUOUS)
    for i in range(3)
]
for i in range(len(PARENT_TMAPS)):
    PARENT_TMAPS[i].parents = PARENT_TMAPS[:i]
CYCLE_PARENTS = [
    TensorMap(f'parent_test_cycle_{i}', shape=(1,), interpretation=Interpretation.CONTINUOUS)
    for i in range(3)
]
for i in range(len(CYCLE_PARENTS)):
    CYCLE_PARENTS[i].parents = [CYCLE_PARENTS[i - 1]]  # 0th tmap will be child of last
