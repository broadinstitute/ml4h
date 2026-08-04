# DROID-RV synthetic test pipeline

A small synthetic dataset checked into this repo, plus scripts to run the DROID-RV
training and inference pipelines against it. Use it to verify an environment,
container, or code change end to end without needing access to real
echocardiograms.

**The videos are random noise.** Predictions and training metrics from this dataset
are not meaningful measurements. What it verifies is that the data loads, the
checkpoints restore, and forward and backward passes run with the expected shapes.

## Setup

The dataset and model checkpoints are stored in git-lfs and are not present after a
plain `git clone`. Pull them:

```bash
git clone https://github.com/broadinstitute/ml4h.git
cd ml4h
git lfs pull --include model_zoo/DROID-RV/synthetic_data/data/*
git lfs pull --include model_zoo/DROID-RV/movinet_a2_base/*
git lfs pull --include model_zoo/DROID-RV/droid_rv_checkpoint/*
git lfs pull --include model_zoo/DROID-RV/droid_rvef_checkpoint/*
```

Confirm the pull worked — if lfs did not run, these files will be ~130 byte
pointer stubs rather than real data:

```bash
du -sh model_zoo/DROID-RV/synthetic_data/data     # expect ~37M
```

Then start the DROID container, which has TensorFlow, `ml4ht`, and the video
dependencies. It is not compatible with Apple Silicon.

```bash
docker pull alalusim/droid:latest
docker run -it -v {PATH TO CLONED ML4H DIRECTORY}:/ml4h/ alalusim/droid:latest
cd /ml4h/model_zoo/DROID-RV/synthetic_data/
```

## Run everything

```bash
./run_smoke_test.sh
```

That runs three stages in order and exits non-zero if any fails:

| Stage | Script | What it checks |
| --- | --- | --- |
| validate | `validate_synthetic_data.py` | Dataset matches the format the recipes require |
| inference | `run_inference_smoke_test.py` | Published DROID-RV and DROID-RVEF checkpoints load and predict |
| training | `echo_supervised_training_recipe.py` | One epoch trains with regression and classification heads |

Variants and stage selection:

```bash
./run_smoke_test.sh rvef                          # training uses the DROID-RVEF output heads
STAGES=validate ./run_smoke_test.sh               # format checks only, no TensorFlow needed
STAGES=validate,inference ./run_smoke_test.sh     # skip training
DATA_DIR=/path/to/other/data ./run_smoke_test.sh  # point at a different dataset
```

## Run stages individually

### Validate the dataset

Checks columns, dtypes, id formats, split disjointness, LMDB manifests, and that
every sampled video decodes to the expected shape. Needs only `lmdb`, `av`,
`pandas`, `pyarrow`, and `numpy` — no TensorFlow, no checkpoints. Exits non-zero on
failure, and runs all checks before reporting rather than stopping at the first.

```bash
python validate_synthetic_data.py --data_dir ./data
```

```
--data_dir ./data
--n_frames 16          expected frames per stored video
--n_input_frames 16    frames the model consumes
--skip_modulo 1
--batch_size 4         batch size to check split sizes against
--sample_studies 3     studies whose videos are fully decoded
```

If `ml4ht` is importable it additionally pushes one sample through the real
`LmdbEchoStudyVideoDataDescription` and asserts the tensor is
`(n_input_frames, 224, 224, 3)` float32 in `[0, 1]`; otherwise that check reports
`SKIP`.

### Inference with the published checkpoints

Loads videos through the same loader the DROID recipes use, runs both published
checkpoints, rescales the regression heads into physical units, and writes
predictions to parquet.

```bash
python run_inference_smoke_test.py --data_dir ./data --split test
```

```
--data_dir ./data
--split test           train | valid | test | all
--models both          rv | rvef | both
--n_samples 0          0 runs every sample in the split
--batch_size 4
--n_input_frames 16
--skip_modulo 1
--movinet_chkp_dir ../movinet_a2_base
--output_dir ./inference_smoke_test_output
```

Writes `predictions_droid_rv.pq` and `predictions_droid_rvef.pq`, each with one row
per video: `sample_id`, the rescaled regression outputs, the argmax class per
classification head, and the per-class softmax probabilities.

Beyond shape and finiteness, it checks each regression output falls inside a
generous physiologic window (for example Age in `[0, 120]`). A checkpoint that
failed to restore produces values far outside these, so this catches a silently
broken weight load — the most likely failure mode, since Keras will happily build
the head architecture and leave it randomly initialized.

On CPU expect a few minutes: MoViNet-A2 over 16×224×224×3 is not fast. Use
`--n_samples 8` for a quicker check.

### Training

```bash
cd ../../DROID
python echo_supervised_training_recipe.py \
    --n_input_frames 16 \
    --output_labels age \
    --output_labels rvedd \
    --output_labels rv_size \
    --output_labels rv_function \
    --output_labels sex \
    --output_labels_types rrccc \
    --wide_file ../DROID-RV/synthetic_data/data/wide_file.parquet \
    --splits_file ../DROID-RV/synthetic_data/data/splits.json \
    --lmdb_folder ../DROID-RV/synthetic_data/data/lmdb \
    --selected_views A4C \
    --selected_views RV_focused \
    --selected_doppler standard \
    --selected_quality good \
    --selected_canonical on_axis \
    --n_train_patients all \
    --batch_size 4 \
    --epochs 1 \
    --es_patience 5 \
    --scale_outputs \
    --skip_modulo 1 \
    --adam 1e-4 \
    --movinet_chkp_dir ../DROID-RV/movinet_a2_base \
    --output_dir ./smoke_test_output
```

The recipe must be run from the `DROID` directory: it imports `echo_defines`,
`data_descriptions`, and `model_descriptions` as top-level modules.

`--batch_size 4` matters. Batches are formed with `drop_remainder=True` and the
validation split holds 7 videos, so a larger batch size would leave the validation
dataset empty; the recipe now raises a clear error in that case rather than failing
obscurely later.

Training draws only from `patient_train`. Earlier versions of the recipe built the
training dataset from every filtered row regardless of split, which leaked
validation and test patients into training — see [Recipe change](#recipe-change)
below.

To fine-tune from a published checkpoint rather than training the heads from
scratch, add `--pretrained_chkp_dir ../DROID-RV/droid_rv_checkpoint/chkp`. Note that
this path also requires `model_params.json` and
`classification_class_label_mapping_per_output.json` in the checkpoint directory's
*parent* (that is, `model_zoo/DROID-RV/`), which the published checkpoints do not
ship — so fine-tuning needs those files supplied by hand.

## What is in the dataset

100 datapoints — one row per video: 20 patients × 1 study each × 5 videos. Videos
are 16 frames of 224×224×3 noise. About 37 MB.

```
data/
├── wide_file.parquet          100 rows, one per video
├── splits.json                train=12, valid=2, test=6 patients
└── lmdb/
    ├── 170682.lmdb/           one LMDB per study
    │   ├── data.mdb           keys = video ids, values = MJPG-encoded avi bytes
    │   └── log_170682.pq      manifest: view (= video id), study, log, stored
    └── ...
```

Wide file columns:

| Column | Notes |
| --- | --- |
| `sample_id` | `{patient_id}_{study_id}_{video_id}`, e.g. `170682_170682_3a7f1c9e` |
| `patient_id`, `study_id`, `video_id` | Convenience columns; not read by the recipes |
| `view_prediction`, `doppler_prediction`, `quality_prediction`, `canonical_prediction` | Integer codes from [`echo_defines.py`](../../DROID/echo_defines.py) |
| `age` (years), `rvedd` (mm), `rvef` (%), `rvedv` (mL), `rvesv` (mL) | Regression outcomes (`r`) |
| `sex`, `rv_size`, `rv_function` | Classification outcomes (`c`) |

Classification values match the readme's documented output ordering, because the
training recipe alphabetically sorts each column's unique values to assign class
indices: `Female`/`Male`, `Dilated`/`Not Dilated`, `Hypokinetic`/`Not Hypokinetic`.
Outcomes are constant per patient and internally consistent — `rvedd` is larger when
`rv_size` is `Dilated`, `rvef` lower when `rv_function` is `Hypokinetic` — and
regression values sit in the same physical ranges the published checkpoints predict.

70 of the 100 rows carry `A4C`/`RV_focused` + `standard` + `good` + `on_axis`
metadata and so survive the selection filter used throughout this README; the other
30 deliberately fail at least one criterion, so filtering is exercised rather than
being a no-op. Every patient keeps at least 3 passing videos. After filtering and
splitting: 42 train, 7 valid, 21 test videos.

The splits file carries all three keys the recipes read. The 70/30 train/test split
is refined to 60/10/30 by carving validation out of the training pool, since the
training recipe builds a validation dataset from `patient_valid`.

## Format constraints

Derived from the consuming code rather than the docs, and worth knowing when
building a real dataset in this format:

- **`sample_id` must split into exactly three `_`-delimited parts.**
  `LmdbEchoStudyVideoDataDescription.get_raw_data` does
  `_, study, view = sample_id.split('_')`, so none of the three ids may contain an
  underscore. Video ids are hex strings for that reason.
- **The patient prefix must parse as an int.** The recipe partitions with
  `int(sample_id.split('_')[0]) in patient_train`, so patient ids in the splits
  file must be JSON ints, not strings.
- **LMDB values must be encoded video bytes, not arrays.** The loader hands them to
  `av.open`. These were written with the same `cv2.VideoWriter` + `MJPG` + `.avi`
  path as [`echo_to_lmdb.py`](../../DROID/echo_to_lmdb.py).
- **The `view` column in `log_{study}.pq` holds video ids, not echo view names.**
  That is the convention `echo_to_lmdb.py` established; `get_loading_options` reads
  it to confirm a video was stored.
- **Frames per video must be at least `n_input_frames * skip_modulo`.** The loader
  wraps with `itertools.cycle`, so a short video silently repeats from frame 0
  instead of erroring. These videos have 16 frames, hence `--skip_modulo 1`
  throughout. The readme's `--skip_modulo 4` would need 64-frame videos.
- **`lock.mdb` is not committed.** LMDB recreates it on open; it is gitignored.

## Recipe change

Building this test pipeline surfaced a bug in
[`echo_supervised_training_recipe.py`](../../DROID/echo_supervised_training_recipe.py),
fixed alongside it: the training dataset and step count were built from
`working_ids` — every row surviving the view filter — instead of `train_ids`. Since
`train_ids` was computed but only used for output scaling, validation and test
patients were being trained on, and `val_loss` was measured on data the model had
already seen. Any early stopping or model selection driven by that number was
compromised.

The recipe now batches `train_ids` and sets `n_train_steps` from it, and raises up
front if either split is smaller than one batch. Runs before this fix saw more
training data per epoch than they should have, so their validation metrics are
optimistic and not comparable to runs after it.

## Regenerating

`generate_synthetic_data.py` produced the checked-in dataset and can rebuild it or
make a larger one. `--seed` makes it reproducible.

```bash
python generate_synthetic_data.py --output_dir ./data --n_patients 20 --n_frames 16
python generate_synthetic_data.py --help
```

Regenerating changes every id, so the checked-in parquet and all 20 LMDBs will show
as modified. Only commit a regenerated dataset if you intend to replace it, and
check `git lfs status` before pushing.
