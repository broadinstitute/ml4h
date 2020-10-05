# The suggested base images are:
#   - tensorflow/tensorflow:2.1.0-gpu-py3 for GPU-enabled machines
#   - tensorflow/tensorflow:2.1.0-py3 for CPU-only (non-GPU-enabled) machines
#   BASE_IMAGE can be specified at build time by adding the following argument:
#   --build_arg BASE_IMAGE="some_other_image"

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="James Pirruccello <jamesp@broadinstitute.org>"

# Setup time zone (or else docker build hangs)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./config/* /app/
WORKDIR /app

# Note that some layers are kept separate to encourage layer re-use and to try
# to minimize full recompilation where possible.

# Basic setup
RUN ./ubuntu.sh

# Requirements for the tensorflow project
RUN pip3 install --upgrade pip
RUN pip3 install -r pre_requirements.txt
RUN pip3 install -r tensorflow-requirements.txt \
  # Configure notebook extensions.
  && jupyter nbextension install --user --py ipycanvas \
  && jupyter nbextension enable --user --py ipycanvas
