import numpy as np
import pytest
import tensorflow as tf

from ml4h.metrics import dice, per_class_dice

SIMPLE_LABELS = {'background': 0, 'foreground': 1}
THREE_LABELS  = {'background': 0, 'class_a': 1, 'class_b': 2}
MULTI_LABELS  = {'background': 0, 'class_a': 1, 'class_b': 2, 'class_c': 3}

H, W = 4, 4

def make_onehot(C, class_idx, batch=1):
    if not isinstance(class_idx, list):
        class_idx = [class_idx]
    arr = np.zeros((batch, H, W, C), dtype=np.float32)
    for c in class_idx:
        arr[:, :, :, c] = 1.0
    return tf.constant(arr)

def make_hard_coded_ys():
    y_true =  [[[[0.37363327, 0.21187197, 0.4144948 ],
                 [0.39462468, 0.05329282, 0.5520825 ],
                 [0.4543256,  0.46920314, 0.07647125],
                 [0.2576654,  0.21213321, 0.53020144]],
            
                [[0.33709493, 0.43075603, 0.23214906],
                 [0.2687178,  0.6558161,  0.07546615],
                 [0.37324676, 0.28486928, 0.34188393],
                 [0.1598155,  0.43757752, 0.40260693]],
            
                [[0.5406405,  0.13518988, 0.3241696 ],
                 [0.04971248, 0.1751017,  0.7751858 ],
                 [0.3654203,  0.47471222, 0.15986742],
                 [0.3598489,  0.45610666, 0.18404447]],
            
                [[0.15605517, 0.5713928,  0.27255207],
                 [0.3453283,  0.22537738, 0.42929432],
                 [0.3795698,  0.16931435, 0.45111582],
                 [0.5435501,  0.2617088,  0.19474119]]]]

    y_pred =  [[[[0.30859458, 0.29316968, 0.3982358 ],
                 [0.0356649,  0.41285014, 0.551485  ],
                 [0.3359447,  0.23898377, 0.42507148],
                 [0.7678544,  0.11798254, 0.11416306]],
            
                [[0.87619734, 0.05819954, 0.06560314],
                 [0.2443302,  0.16162986, 0.5940399 ],
                 [0.7188649,  0.1548999,  0.12623519],
                 [0.25660914, 0.36728388, 0.376107  ]],
            
                [[0.32421544, 0.36816892, 0.30761567],
                 [0.23573558, 0.5330063,  0.23125812],
                 [0.3389197, 0.41501033, 0.24606998],
                 [0.39741543, 0.2885476,  0.314037  ]],
            
                [[0.1856378,  0.2387597,  0.5756026 ],
                 [0.11451269, 0.49046806, 0.39501923],
                 [0.22394544, 0.38586223, 0.39019236],
                 [0.31319028, 0.11418176, 0.57262796]]]]
    return tf.constant(y_true), tf.constant(y_pred)

class TestDice:
    def test_perfect_prediction_loss(self):
        y = make_onehot(len(SIMPLE_LABELS), class_idx=0)
        assert float(dice(y, y)) == pytest.approx(-1.0, abs=1e-4)

    def test_complete_mismatch_loss(self):
        y_true = make_onehot(len(SIMPLE_LABELS), class_idx=0)
        y_pred = make_onehot(len(SIMPLE_LABELS), class_idx=1)
        assert float(dice(y_true, y_pred)) == pytest.approx(0.0, abs=1e-3)

    def test_soft_probabilities_loss(self):        
        y_true, y_pred = make_hard_coded_ys()
        assert float(dice(y_true, y_pred)) == pytest.approx(-0.7527249455451965, abs=1e-3)

class TestPerClassDice:
    def test_returns_one_function_per_label(self):
        fns = per_class_dice(MULTI_LABELS)
        assert len(fns) == len(MULTI_LABELS)

    def test_function_names_derived_from_label_keys(self):
        # Hyphens and spaces in label names should become underscores
        labels = {'my-label': 0, 'other label': 1}
        fns = per_class_dice(labels)
        assert fns[0].__name__ == 'my_label_dice'
        assert fns[1].__name__ == 'other_label_dice'

    def test_perfect_prediction_coefficients(self):
        y = make_onehot(len(MULTI_LABELS), class_idx=0)
        fns = per_class_dice(MULTI_LABELS)
        assert len(fns) == len(MULTI_LABELS)
        for fn in fns:
            assert float(fn(y, y)) == pytest.approx(1.0, abs=1e-4)

    def test_complete_mismatch_coefficients(self):
        y_true = make_onehot(len(SIMPLE_LABELS), class_idx=0)
        y_pred = make_onehot(len(SIMPLE_LABELS), class_idx=1)
        fns = per_class_dice(SIMPLE_LABELS)
        assert len(fns) == len(SIMPLE_LABELS)
        for fn in fns:
            assert float(fn(y_true, y_pred)) == pytest.approx(0.0, abs=1e-4)

    def test_different_situations_coefficients(self):
        # Each function must return the dice for its own class.
        #
        # Setup:
        #   background (idx 0): present in y_true, absent from y_pred → coefficient ≈ 0
        #   class_a    (idx 1): absent from y_true, present in y_pred → coefficient ≈ 0
        #   class_b    (idx 2): absent from both                      → coefficient ≈ 1
        #   class_c    (idx 3): present in both                       → coefficient ≈ 1
        y_true = make_onehot(len(MULTI_LABELS), class_idx=[0,3])
        y_pred = make_onehot(len(MULTI_LABELS), class_idx=[1,3])
        fns = per_class_dice(MULTI_LABELS)
        assert len(fns) == len(MULTI_LABELS)

        coeff_background = float(fns[0](y_true, y_pred))
        coeff_class_a    = float(fns[1](y_true, y_pred))
        coeff_class_b    = float(fns[2](y_true, y_pred))
        coeff_class_c    = float(fns[3](y_true, y_pred))

        assert coeff_background == pytest.approx(0.0, abs=1e-3)
        assert coeff_class_a    == pytest.approx(0.0, abs=1e-3)
        assert coeff_class_b    == pytest.approx(1.0, abs=1e-3)
        assert coeff_class_c    == pytest.approx(1.0, abs=1e-3)

    def test_soft_probabilities_coefficients(self):
        y_true, y_pred = make_hard_coded_ys()
        fns = per_class_dice(THREE_LABELS)
        assert len(fns) == len(THREE_LABELS)

        coeff_background = float(fns[0](y_true, y_pred))
        coeff_class_a    = float(fns[1](y_true, y_pred))
        coeff_class_b    = float(fns[2](y_true, y_pred))

        assert coeff_background == pytest.approx(0.7871747612953186, abs=1e-3)
        assert coeff_class_a    == pytest.approx(0.7200393676757812, abs=1e-3)
        assert coeff_class_b    == pytest.approx(0.7509607672691345, abs=1e-3)