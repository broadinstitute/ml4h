# Command line recipes introduction
All of the functionalilty of the `ml4h` package is available from the command line through [recipes](ml4h/recipes.py).
Which `recipe` you use is specified by `--mode`.
E.g.
```bash
python ml4h/recipes.py --mode train ...
```
or
```bash
python ml4h/recipes.py --mode explore ...
```

The command line arguments are specified in [arguments](ml4h/arguments.py).
Almost all of the arguments are optional and depend on the `recipe` you want to run.

# Index
Pipelines:
* [Tensorization](#tensorization)
* [Modeling](#modeling)

Recipes modes:
* [explore](#explore)
* [train](#train)
* [train_transformer_for_survival](#train_transformer_for_survival)
* [test](#test)
* [infer](#infer)

# Examples

## train_transformer_for_survival

Train one phecode survival transformer from participant-day parquet files. The
train/validation file must contain `split=train/valid`, and the held-out file
must contain `split=test`. Required columns are `person_id`, `date`, `split`,
`last_encounter_date`, `<phecode>__start_date`, and the requested numeric/latent
features. Use a string phecode, including any decimal point. Features must already
be numeric, finite, and preprocessed; fit preprocessing on training data only.

```bash
python -m ml4h.recipes \
  --mode train_transformer_for_survival \
  --id phecode_250_2_survival --output_folder ./runs \
  --survival_phecode 250.2 \
  --transformer_input_file /data/250.2__trainval__noblank.pq \
  --transformer_test_file /data/250.2__test.pq \
  --latent_dimensions 14 --input_numeric_columns delta_norm \
  --transformer_max_size 1000 --transformer_positional_embedding \
  --transformer_scalar_embed 16 --transformer_latent_embed 64 \
  --transformer_size 64 --attention_heads 4 --transformer_layers 2 \
  --transformer_dropout_rate 0.1 \
  --survival_intervals 20 --survival_days_window 3650 \
  --survival_test_screen_days 1000 --survival_num_curves 64 \
  --batch_size 4 --epochs 50 --learning_rate 0.0001 --patience 8 \
  --training_steps 0 --validation_steps 0 --test_steps -1 \
  --dpi 150 --plot_width 9 --plot_height 7 --inspect_model
```

Training samples only the start position, uniformly over each participant's full
chronologically sorted history, then takes consecutive recorded days up to the
context size. Windows near the end may be shorter; missing calendar days are not
inserted. Follow-up starts at the sampled first date and ends at diagnosis or last
encounter, capped at the prediction horizon. Diagnoses on/before the sampled start
are encoded as time-zero events. Training and validation retain prevalent and
postdiagnosis data, including days labelled `12`; day/participant class labels are
not used to construct the date-based survival target. Both trainval parquet variants
therefore give the same survival inputs/targets when their dates and features match.

Validation uses each participant's first context window. Testing first excludes
participants with **any date strictly after diagnosis among their first 1,000
recorded days** (configurable with `--survival_test_screen_days`), then uses the first
context window within that period. The supplied participant splits are preserved
and checked for leakage. This task uses observations after the survival origin;
it is not a forecast conditioned only on observations available at that origin.

Positive `--training_steps` trains on a repeating stream for that many batches per
epoch; zero or negative uses one participant pass. Positive `--validation_steps`
caps validation batches; zero or negative evaluates all validation participants.
Positive `--test_steps` caps test batches, zero skips testing, and negative evaluates
all eligible test participants. Existing transformer defaults remain unchanged, so
specify the 14 latent features and 1,000-day context as shown above.

The run directory contains `<id>.keras` (best validation epoch, or last epoch with
`--save_last_model`), loss history, population/individual/risk-stratified survival
plots, horizon calibration, and `metrics_<id>.json`. Metrics use the training
recipe's `Model`, `Task`, `Metric`, `Score`, `CI_95_lower`, `CI_95_upper`, `n` and
`n_positive` columns. Confidence intervals are not estimated here; they and
undefined scores are JSON `null`. Calibration/Brier scores use participants with
known horizon outcomes and are not censoring-adjusted population estimates.

`--inspect_model` additionally logs a real sampled training batch and writes
`survival_inspection_{summary,feature_means,targets}.tsv` plus an architecture plot.
Inspection reports window dates, origin-relative status, postdiagnosis input days,
feature means excluding padding, and both survival-target masks. Its independent
seed does not advance the training sampling RNG. `--model_file` loads a compatible
saved model for further training with the requested optimizer settings.

## Tensorization
TODO

## Modeling
For all of our modeling examples, we will use MNIST data, which requires you to have MNIST data in `hd5` format.
To set that up, run the [MNIST demo](notebooks/mnist_demo.ipynb) at least through **Tensorization**.

You also should have docker set up following the instructions in the [readme](README.md).
You can run the recipes from the docker image
```bash
cd [path_to_repo]/ml  # navigate to repo
docker run -it --rm --ipc=host -v $PWD:$PWD gcr.io/broad-ml4cvd/deeplearning:tf2-latest-cpu  # enter cpu docker image
cd [path_to_repo]/ml  # navigate to repo in docker
pip install .  # install ml4h package
```
To run recipes with the gpu, use
```bash
docker run --gpus -it --rm --ipc=host -v $PWD:$PWD gcr.io/broad-ml4cvd/deeplearning:tf2-latest-gpu
```

### explore
The first step of modeling is to explore your dataset.
```bash
python ml4h/recipes.py --mode explore --input_tensors mnist.mnist_image mnist.mnist_label --tensors notebooks/mnist_hd5s --output_folder ./explorations --id mnist_explore
```
We can look at the summary stats of the exploration in `explorations/mnist_explore/summary_stats_categorical_mnist_label_intersect.csv`.
All of the labels appear in `explorations/mnist_explore/tensors_all_union.csv`.

### train
Now lets train a couple models.
First, we'll use the default model architecture settings.
```bash
python ml4h/recipes.py --mode train --input_tensors mnist.mnist_image --output_tensors mnist.mnist_label --tensors notebooks/mnist_hd5s --output_folder train_runs --id mnist_train_default --batch_size 256 --epochs 5 --training_steps 130 --validation_steps 20 --test_steps 1
```
Now we can see how precision and recall change for each digit in the training and validation sets at
`train_runs/mnist_train_default/metric_history_mnist_train_default.png`.

Let's try adding dropout, using the ["swish"](https://arxiv.org/abs/1710.05941) activation, switching to the `Adam` optimizer, and decreasing the learning rate.
```
python ml4h/recipes.py --mode train --input_tensors mnist.mnist_image --output_tensors mnist.mnist_label --tensors notebooks/mnist_hd5s --output_folder train_runs --id mnist_train_swish_dropout --batch_size 256 --epochs 5 --training_steps 130 --validation_steps 20 --test_steps 1 \
--activation swish --conv_normalize batch_norm --dense_regularize dropout --dense_regularize_rate .01 --optimizer adam --learning_rate 1e-4
```
Now we can see the metrics changing over epochs of training in 
`/train_runs/mnist_train_swish_dropout/metric_history_mnist_train_swish_dropout.png`.

### test
Now let's look at the test set performance of the second model we trained.

:warning: `ml4h` sets a seed so the test data will be the same in this example, but it's best to specify test data using `--test_csv` :warning:
```bash
python ml4h/recipes.py --mode test  --input_tensors mnist.mnist_image --output_tensors mnist.mnist_label --tensors notebooks/mnist_hd5s \
--output_folder test_runs --model_file train_runs/mnist_train_swish_dropout/mnist_train_swish_dropout.h5 --test_steps 5 --batch_size 256 \
--id test_swish_dropout
```
The number of samples evaluated will be `test_steps * batch_size = 5 * 256 = 1,280`.
Lots of figures are automatically produced, including `test_runs/test_swish_dropout/calibrations_mnist_label.png` which shows the calibration of our classifier.

### compare
The `compare` recipe allows us to compare and plot the results of our two models.
```bash
python ml4h/recipes.py --mode compare  --input_tensors mnist.mnist_image --output_tensors mnist.mnist_label --tensors notebooks/mnist_hd5s --output_folder test_runs \
--model_files train_runs/mnist_train_default/mnist_train_default.h5 train_runs/mnist_train_swish_dropout/mnist_train_swish_dropout.h5 --test_steps 5 --batch_size 256 \
--id compare_default_swish_dropout
```
The results of the comparison can be seen in files like `test_runs/compare_default_swish_dropout/per_class_roc_mnist_label_compare_default_swish_dropout.png`.

### infer
We can also get a model's predictions on the entire dataset in a tab delimited file (`tsv`).
```bash
python ml4h/recipes.py --mode infer  --input_tensors mnist.mnist_image --output_tensors mnist.mnist_label --tensors notebooks/mnist_hd5s \
--output_folder test_runs --model_file train_runs/mnist_train_swish_dropout/mnist_train_swish_dropout.h5 --id test_swish_dropout
```
The results will appear in `test_runs/test_swish_dropout/inference_test_swish_dropout.tsv`.
