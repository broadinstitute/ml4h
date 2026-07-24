ARG BASE_IMAGE=ghcr.io/broadinstitute/ml4h:tf2.19-latest-gpu
FROM ${BASE_IMAGE}

LABEL maintainer="Broad Institute ML4H"
LABEL org.opencontainers.image.description="DROID runtime layered on top of the official ml4h TensorFlow 2.19 image"

# TensorFlow 2.19 defaults to Keras 3. DROID relies on tf.keras-era behavior
# through tf-models-official, so keep the runtime on the legacy tf.keras stack.
ENV TF_USE_LEGACY_KERAS=1
ENV PYTHONPATH=/ml4h
WORKDIR /ml4h

# The base ml4h image already carries the TensorFlow 2.19 stack and DROID
# runtime dependencies. Copy the repo in so the image can run DROID recipes
# directly without an extra install step.
COPY . /ml4h

RUN pip install --upgrade nvidia-cublas-cu11 nvidia-cudnn-cu11 scikit-survival
