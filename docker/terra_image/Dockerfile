FROM us.gcr.io/broad-dsp-gcr-public/terra-jupyter-gatk:2.0.1
# https://github.com/DataBiosphere/terra-docker/blob/master/terra-jupyter-gatk/CHANGELOG.md

USER root
ENV PIP_USER=false

RUN pip3 install -U git+https://github.com/broadinstitute/ml4h.git

# Remove the following two installs after https://github.com/DataBiosphere/terra-docker/issues/244 is addressed.
RUN pip3 install -U tensorflow==2.4.3
RUN pip3 install tensorflow-probability==0.12.2

# Remove this after https://broadworkbench.atlassian.net/browse/CA-1240
# As of release [google-cloud-bigquery 1.26.0 (2020-07-20)](https://github.com/googleapis/python-bigquery/blob/master/CHANGELOG.md#1260-2020-07-20)
# the BigQuery Python client uses the BigQuery Storage client by default.
# This currently causes an error on Terra Cloud Runtimes `the user does not have 'bigquery.readsessions.create'
# permission for '<Terra billing project id>'`. To work-around this uninstall the dependency so that flag
# `--use_rest_api` can be used with `%%bigquery` to use the older, slower mechanism for data transfer.
RUN pip3 uninstall -y google-cloud-bigquery-storage

ENV USER jupyter
USER $USER

RUN  jupyter nbextension install --user  --py vega \
  && jupyter nbextension enable --user  --py vega \
  && jupyter nbextension install --user  --py ipycanvas \
  && jupyter nbextension enable --user  --py ipycanvas

ENV PIP_USER=true
