import os
import matplotlib.pyplot as plt
from ml4cvd.arguments import parse_args
from ml4cvd.plots import _plot_reconstruction
from ml4cvd.tensor_generators import test_train_valid_tensor_generators, big_batch_from_minibatch_generator
from ml4cvd.models import make_multimodal_multitask_model


args = parse_args()
_, _, generate_test = test_train_valid_tensor_generators(**args.__dict__)
generate_test.augment = True
model = make_multimodal_multitask_model(**args.__dict__)
test_data, test_labels, paths = big_batch_from_minibatch_generator(generate_test, args.test_steps)
generate_test.kill_workers()
tm = args.tensor_maps_in[0]
y_pred = model.predict(test_data)
y_per = test_data[tm.input_name()]
y_orig = test_labels[tm.output_name()]

SUBPLOT_SIZE = 5
alpha = .7
for i in range(5):
    title = f'ECG reconstruction'
    y = y_orig[i].reshape(tm.shape)
    yp = y_pred[i].reshape(tm.shape)
    ya = y_per[i].reshape(tm.shape)
    fig = plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE * tm.shape[1]))
    for j in range(tm.shape[1]):
        plt.subplot(tm.shape[1], 1, j + 1)
        plt.plot(y[:, j], label='original', linewidth=3, alpha=alpha)
        plt.plot(ya[:, j], label='augmented', linewidth=3, alpha=alpha)
        plt.plot(yp[:, j], label='reconstruction', linewidth=3, alpha=alpha)
        plt.axis('off')
        if j == 0:
            plt.title(title)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, f'reconstruction_{i}.png'), dpi=200)
    plt.clf()

