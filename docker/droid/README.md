# DROID Docker

This image layers the DROID model zoo code on top of the official `ml4h`
TensorFlow 2.19 image instead of using the legacy `alalusim/droid` container.

## Build

From the repository root:

```commandline
docker build \
  -f docker/droid/Dockerfile \
  -t ml4h-droid:tf2.19 \
  .
```

To build from a different official `ml4h` base tag, override `BASE_IMAGE`:

```commandline
docker build \
  -f docker/droid/Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/broadinstitute/ml4h:tf2.19-latest-cpu \
  -t ml4h-droid:tf2.19-cpu \
  .
```

## Run

The image contains the `ml4h` checkout at `/ml4h` and sets
`TF_USE_LEGACY_KERAS=1` so DROID continues to run against the legacy
`tf.keras` stack required by the MoViNet dependencies in TensorFlow 2.19.

```commandline
docker run -it --rm -v {PATH_TO_CLONED_ML4H}:/ml4h ml4h-droid:tf2.19 /bin/bash
```

Example inference:

```commandline
cd /ml4h/model_zoo/DROID-MVP
python droid_mvp_inference.py
```
