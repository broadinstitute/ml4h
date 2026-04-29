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
    y_true = \
        [[[[0.37363327, 0.21187197, 0.4144948 ],
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
           [0.5435501,  0.2617088,  0.19474119]]],


         [[[0.6677017,  0.1367232,  0.19557498],
           [0.00504563, 0.5393079,  0.45564649],
           [0.3625883,  0.4014423,  0.23596944],
           [0.69100124, 0.1698486,  0.13915014]],

          [[0.3920861,  0.27634567, 0.33156824],
           [0.39164504, 0.32494724, 0.28340778],
           [0.62552875, 0.3399984,  0.03447281],
           [0.41206333, 0.2024707,  0.38546592]],

          [[0.74491036, 0.20419887, 0.05089074],
           [0.2274923,  0.2373637,  0.53514403],
           [0.27778772, 0.3909241,  0.33128813],
           [0.29291674, 0.58673203, 0.12035124]],

          [[0.0271962,  0.10782634, 0.8649774 ],
           [0.41085196, 0.1434555,  0.4456925 ],
           [0.11763471, 0.53778684, 0.34457842],
           [0.29023203, 0.22966821, 0.48009974]]]]

    y_pred = \
        [[[[0.30859458, 0.29316968, 0.3982358 ],
           [0.0356649,  0.41285014, 0.551485  ],
           [0.3359447,  0.23898377, 0.42507148],
           [0.7678544,  0.11798254, 0.11416306]],

          [[0.87619734, 0.05819954, 0.06560314],
           [0.2443302,  0.16162986, 0.5940399 ],
           [0.7188649,  0.1548999,  0.12623519],
           [0.25660914, 0.36728388, 0.376107  ]],

          [[0.32421544, 0.36816892, 0.30761567],
           [0.23573558, 0.5330063,  0.23125812],
           [0.3389197,  0.41501033, 0.24606998],
           [0.39741543, 0.2885476,  0.314037  ]],

          [[0.1856378,  0.2387597,  0.5756026 ],
           [0.11451269, 0.49046806, 0.39501923],
           [0.22394544, 0.38586223, 0.39019236],
           [0.31319028, 0.11418176, 0.57262796]]],


         [[[0.7387094,  0.19665748, 0.0646331 ],
           [0.47842818, 0.3872994,  0.13427243],
           [0.05401381, 0.52096915, 0.42501703],
           [0.31050298, 0.21586749, 0.47362953]],

          [[0.3956212,  0.41374108, 0.19063772],
           [0.2862886,  0.33314437, 0.38056704],
           [0.41920838, 0.52964497, 0.05114669],
           [0.50704116, 0.32072362, 0.17223527]],

          [[0.7440144,  0.11934569, 0.13663992],
           [0.36042786, 0.29690582, 0.34266633],
           [0.26202253, 0.42427298, 0.31370452],
           [0.35303944, 0.3541221,  0.2928384 ]],

          [[0.03276591, 0.24196127, 0.72527283],
           [0.26715282, 0.06793231, 0.6649149 ],
           [0.17861432, 0.11889059, 0.7024951 ],
           [0.26242286, 0.36045143, 0.3771257 ]]]]
    
    return tf.constant(y_true), tf.constant(y_pred)

class TestDice:
    def test_perfect_prediction_loss(self):
        y = make_onehot(len(SIMPLE_LABELS), class_idx=0)
        assert float(dice(y, y)) == pytest.approx(-1.0, abs=1e-5)

    def test_complete_mismatch_loss(self):
        y_true = make_onehot(len(SIMPLE_LABELS), class_idx=0)
        y_pred = make_onehot(len(SIMPLE_LABELS), class_idx=1)
        assert float(dice(y_true, y_pred)) == pytest.approx(0.0, abs=1e-5)

    def test_soft_probabilities_loss(self):        
        y_true, y_pred = make_hard_coded_ys()
        assert float(dice(y_true, y_pred)) == pytest.approx(-0.8196584582328796, abs=1e-5)

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
            assert float(fn(y, y)) == pytest.approx(1.0, abs=1e-5)

    def test_complete_mismatch_coefficients(self):
        y_true = make_onehot(len(SIMPLE_LABELS), class_idx=0)
        y_pred = make_onehot(len(SIMPLE_LABELS), class_idx=1)
        fns = per_class_dice(SIMPLE_LABELS)
        assert len(fns) == len(SIMPLE_LABELS)
        for fn in fns:
            assert float(fn(y_true, y_pred)) == pytest.approx(0.0, abs=1e-5)

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
        
        assert coeff_background == pytest.approx(0.0, abs=1e-5)
        assert coeff_class_a    == pytest.approx(0.0, abs=1e-5)
        assert coeff_class_b    == pytest.approx(1.0, abs=1e-5)
        assert coeff_class_c    == pytest.approx(1.0, abs=1e-5)

    def test_soft_probabilities_coefficients(self):
        y_true, y_pred = make_hard_coded_ys()
        fns = per_class_dice(THREE_LABELS)
        assert len(fns) == len(THREE_LABELS)

        coeff_background = float(fns[0](y_true, y_pred))
        coeff_class_a    = float(fns[1](y_true, y_pred))
        coeff_class_b    = float(fns[2](y_true, y_pred))

        assert coeff_background == pytest.approx(0.8415175080299377, abs=1e-5)
        assert coeff_class_a    == pytest.approx(0.8051831722259521, abs=1e-5)
        assert coeff_class_b    == pytest.approx(0.8122745752334595, abs=1e-5)