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

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, decay_function):
        self.decay_function = decay_function
        self.async_safe = True

    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.lr.append(self.decay_function(len(self.losses)))


class BatchMetricsLogger(tf.keras.callbacks.Callback):
    """Callback function used during `model.evaluate` calls with batch size set to 1.
    This approach stores `tf.keras.metrics` callback results for each test example.
    Only data generated by `tf.keras.metrics` functions will produce correct results!

    Example use:
    ```python
    metrics_dict = {m.name: m for m in model.metrics}
    logger = BatchMetricsLogger(metrics = metrics_dict)
    eval = model.evaluate(test_ds, callbacks=[logger], verbose=1)
    eval_batch = pd.DataFrame(logger.storage, index = test_data.index)
    ```
    """

    def __init__(self, metrics):
        super(BatchMetricsLogger, self).__init__()
        self.metrics = metrics
        self.storage = []
        self.async_safe = True

    #
    def on_test_batch_end(self, batch, logs=None):
        self.storage.append(logs)
