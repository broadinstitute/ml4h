# The suggested base images are:
#   - ufoym/deepo:all-py36-jupyter for GPU-enabled machines
#   - ufoym/deepo:all-py36-jupyter-cpu for CPU-only (non-GPU-enabled) machines
# BASE_IMAGE can be specified at build time by adding the following argument:
#   --build_arg BASE_IMAGE="some_other_image"

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="Sam Freesun Friedman <sam@broadinstitute.org>"

# Setup time zone (or else docker build hangs)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./config/* /app/
WORKDIR /app

# Note that some layers are kept separate to encourage layer re-use and to try
# to minimize full recompilation where possible.

# Basic setup
#RUN rm /etc/apt/sources.list.d/cuda.list
##RUN rm /etc/apt/sources.list.d/nvidia-ml.list
#RUN apt-key del 7fa2af80
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
#RUN ./ubuntu.sh

# FastAI. See the Developer Install under https://github.com/fastai/fastai/ to
# understand this odd sequence of installing then uninstalling fastai before
# installing it from github. (Basically, to get its deps.)
# RUN pip3 install -r fastai-requirements.txt
# RUN pip3 uninstall -y fastai
# RUN ./fastai.sh

RUN apt-get update
RUN apt-get upgrade
RUN apt-get install python3 python3-pip python3-tk libgl1-mesa-glx libxt-dev -y
RUN apt-get install -y wget unzip curl python3-pydot python3-pydot-ng graphviz git ffmpeg

# Requirements for the tensorflow project
RUN pip3 install --upgrade pip
#RUN pip3 install -r pre_requirements.txt
RUN pip3 install -r tensorflow-requirements.txt
