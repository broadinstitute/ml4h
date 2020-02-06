import pytest

from ml4cvd.models import make_multimodal_multitask_model
from ml4cvd.TensorMap import TensorMap, Interpretation


tmap_continuous_1d = TensorMap('1d', shape=(2,), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_2d = TensorMap('2d', shape=(2, 3), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_3d = TensorMap('3d', shape=(2, 3, 4), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_4d = TensorMap('4d', shape=(2, 3, 4, 5), interpretation=Interpretation.CONTINUOUS)

tmap_categorical_1d = TensorMap('1d_cat', shape=(2,), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_2d = TensorMap('2d_cat', shape=(2, 3), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_3d = TensorMap('3d_cat', shape=(2, 3, 4), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_4d = TensorMap('4d_cat', shape=(2, 3, 4, 5), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_5d = TensorMap('5d_cat', shape=(2, 3, 4, 5, 6), interpretation=Interpretation.CATEGORICAL)

CONTINUOUS_TMAPS = [
    tmap_continuous_1d,
    tmap_continuous_2d,
    tmap_continuous_3d,
    tmap_continuous_4d,
]
CATEGORICAL_TMAPS = [
    tmap_continuous_1d,
    tmap_continuous_2d,
    tmap_continuous_3d,
    tmap_continuous_4d,
]


class MakeMultimodalMultitaskModelTest:

    @pytest.mark.parametrize(
        'input_tmap',
        CATEGORICAL_TMAPS + CONTINUOUS_TMAPS,
    )
    @pytest.mark.parametrize(
        'output_tmap',
        CATEGORICAL_TMAPS + CONTINUOUS_TMAPS,
        )
    def unimodal_unitask_test(self, input_tmap: TensorMap, output_tmap: TensorMap):
        m = make_multimodal_multitask_model(
            [input_tmap],
            [output_tmap],
        )
        assert m.input.shape == input_tmap.shape
        assert m.output_shape == output_tmap.shape
        assert m.input.name == input_tmap.input_name()
        assert m.output.name == input_tmap.input_name()
