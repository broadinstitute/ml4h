# Imports: third party
from tensorflow.keras.losses import logcosh

# Imports: first party
from ml4cvd.TensorMap import TensorMap


class TestTensorMaps:
    def test_tensor_map_equality(self):
        tensor_map_1a = TensorMap(
            name="tm", loss="logcosh", channel_map={"c1": 1, "c2": 2}, metrics=[],
        )
        tensor_map_1b = TensorMap(
            name="tm", loss="logcosh", channel_map={"c1": 1, "c2": 2}, metrics=[],
        )
        tensor_map_2a = TensorMap(
            name="tm", loss=logcosh, channel_map={"c1": 1, "c2": 2}, metrics=[],
        )
        tensor_map_2b = TensorMap(
            name="tm", loss=logcosh, channel_map={"c2": 2, "c1": 1}, metrics=[],
        )
        tensor_map_3 = TensorMap(
            name="tm", loss=logcosh, channel_map={"c1": 1, "c2": 3}, metrics=[],
        )
        tensor_map_4 = TensorMap(
            name="tm", loss=logcosh, channel_map={"c1": 1, "c2": 3}, metrics=[all],
        )
        tensor_map_5a = TensorMap(
            name="tm", loss=logcosh, channel_map={"c1": 1, "c2": 3}, metrics=[all, any],
        )
        tensor_map_5b = TensorMap(
            name="tm", loss=logcosh, channel_map={"c1": 1, "c2": 3}, metrics=[any, all],
        )
        tensor_map_6a = TensorMap(
            name="tm",
            loss=logcosh,
            channel_map={"c1": 1, "c2": 3},
            dependent_map=tensor_map_1a,
        )
        tensor_map_6b = TensorMap(
            name="tm",
            loss=logcosh,
            channel_map={"c1": 1, "c2": 3},
            dependent_map=tensor_map_1b,
        )

        assert tensor_map_1a == tensor_map_1b
        assert tensor_map_2a == tensor_map_2b
        assert tensor_map_1a == tensor_map_2a
        assert tensor_map_5a == tensor_map_5b
        assert tensor_map_6a == tensor_map_6b

        assert tensor_map_2a != tensor_map_3
        assert tensor_map_3 != tensor_map_4
        assert tensor_map_3 != tensor_map_5a
        assert tensor_map_4 != tensor_map_5a
