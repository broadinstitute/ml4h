# MIT License
#
# Copyright (c) 2020 Marcus D. R. Klarqvist, PhD, MSc
# https://github.com/mklarqvist/tf-computer-vision
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow_addons.utils.types import TensorLike, Number
from typeguard import typechecked
import sys  # If debugging

class ShrinkageLoss(LossFunctionWrapper):
    """For learning convolutional regression networks, the input search area
    has to contain a large body of background surrounding target objects.
    This background information increases the number of easy samples. Recent
    work has that adding a modulating factor to the entropy loss helps alleviate
    the data imbalance issue using a loss called the focal loss. In regression
    learning, this amounts to re-weighting the square loss using an exponential
    form of the absolute difference term `L` as follows:

    ```latex
    L_F = L^2 L^{\gamma} = L^{2+\gamma}
    ```

    Hence, the focal loss for regression learning is equal to the L3-norm diff.
    This norm penalizes not only easy samples but also difficult samples. Addressing
    this, the authors propose a shrinkage loss where the modulating function is shaped
    like a Sigmoid-like function with two hyperparameters `a` and `c` controlling
    the shrinkage speed and the localization, respectively. The proposed shrinkage
    loss only penalizes the importance of easy samples.

    ```latex
    L_S = \frac{L^2}{1 + \exp(a \times (c - L^1))}
    ```

    Standalone usage:

    ```python
    >>> y_true = tf.ones((2,256,256,10), dtype = tf.float32)
    >>> y_pred = tf.random.uniform((2,256,256,10))
    >>> ShrinkageLoss(a=5.0,c=0.2)(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.24997455>
    ```

    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile(opt='sgd', loss=ShrinkageLoss(a=5.0,c=0.2))
    ```

    Args:
        a: Shrinkage speed (squishification of the Sigmoid-like modulating functon).
        c: Localization of the shrinkage (lateral shift of Sigmoid-like modulating function)
        reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. Defaults to `SUM_OVER_BATCH_SIZE`.
            See `tf.keras.losses.Reduction` for more information.
        name: Optional name for the op.


    References:
        Deep Regression Tracking with Shrinkage Loss
        (https://openaccess.thecvf.com/content_ECCV_2018/html/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.html).

    """

    @typechecked
    def __init__(
        self,
        a: Number = 5.0,
        c: Number = 0.2,
        reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "shrinkage_loss",
    ):
        super().__init__(
            shrinkage_loss,
            reduction=reduction,
            name=name,
            a=a,
            c=c,
        )


@tf.function
def shrinkage_loss(
    y_true: TensorLike, y_pred: TensorLike, a: Number = 5.0, c: Number = 0.2,
):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    a = tf.convert_to_tensor(a, dtype=K.floatx())
    c = tf.convert_to_tensor(c, dtype=K.floatx())
    l1 = tf.math.abs(y_true - y_pred)
    l2 = tf.math.square(l1)
    shrinkage = tf.math.divide_no_nan(l2, 1 + tf.math.exp(a * (c - l1)))
    return shrinkage
