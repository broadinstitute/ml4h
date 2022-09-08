FROM us.gcr.io/broad-dsp-gcr-public/terra-jupyter-gatk:2.2.8
# https://github.com/DataBiosphere/terra-docker/blob/master/terra-jupyter-gatk/CHANGELOG.md

USER root
ENV PIP_USER=false

WORKDIR /

RUN pip3 install --upgrade git+https://github.com/broadinstitute/ml4h.git

ENV USER jupyter
USER $USER
ENV PIP_USER=true

RUN  jupyter nbextension install --user  --py vega \
  && jupyter nbextension enable --user  --py vega \
  && jupyter nbextension install --user  --py ipycanvas \
  && jupyter nbextension enable --user  --py ipycanvas
