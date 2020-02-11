FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter
MAINTAINER M. D. R. Klarqvist <mklarqvi@broadinstitute.org>

# Setup time zone (or else docker build hangs)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Temporary solution.
# Circumvent grabbing latest codebase from GitHub.
RUN apt-get update && apt-get install -y python3-tk libxt-dev libgl1-mesa-glx libsm6 python-pydot python-pydot-ng graphviz
RUN pip install --upgrade pip
# Copy current directory to container.
COPY . ml
# Install the codebase.
RUN cd ml; pip install .