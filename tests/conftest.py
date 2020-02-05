from ml4cvd.TensorMap import TensorMap, Interpretation


tmap_continuous_1d = TensorMap('1d', shape=(2,), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_2d = TensorMap('2d', shape=(2, 3), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_3d = TensorMap('3d', shape=(2, 3, 4), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_4d = TensorMap('4d', shape=(2, 3, 4, 5), interpretation=Interpretation.CONTINUOUS)

tmap_categorical_1d = TensorMap('1d_cat', shape=(2,), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_2d = TensorMap('2d_cat', shape=(2, 3), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_3d = TensorMap('3d_cat', shape=(2, 3, 4), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_4d = TensorMap('4d_cat', shape=(2, 3, 4, 5), interpretation=Interpretation.CATEGORICAL)

