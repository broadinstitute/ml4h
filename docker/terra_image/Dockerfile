FROM us.gcr.io/broad-dsp-gcr-public/terra-jupyter-gatk:2.0.2
# https://github.com/DataBiosphere/terra-docker/blob/master/terra-jupyter-gatk/CHANGELOG.md

USER root
ENV PIP_USER=false

WORKDIR /home/jupyter

# Now that the Terra image is built from a TensorFlow base image, if TF is included in the
# ML4h requirements.txt, this causes problems with the Terra 'cloud environment' launch.
# In addition, other tensorflow-related libraries need to be unpinned so that they remain consistent
# with the TF version used by the Terra base image.
RUN wget --no-check-certificate https://raw.githubusercontent.com/broadinstitute/ml4h/master/docker/vm_boot_images/config/tensorflow-requirements.txt -O /home/jupyter/tensorflow-requirements.txt
RUN grep -v 'tensorflow==' /home/jupyter/tensorflow-requirements.txt \
 | sed -E 's/(tensorflow-addons)(==.*)$/\1/g' \
 | sed -E 's/(tensorflow_probability)(==.*)$/\1/g' > /home/jupyter/terra-tensorflow-requirements.txt \
&& pip3 install --upgrade --requirement /home/jupyter/terra-tensorflow-requirements.txt \
&& pip3 install --upgrade --no-deps git+https://github.com/broadinstitute/ml4h.git


# Remove this after https://broadworkbench.atlassian.net/browse/CA-1240
# As of release [google-cloud-bigquery 1.26.0 (2020-07-20)](https://github.com/googleapis/python-bigquery/blob/master/CHANGELOG.md#1260-2020-07-20)
# the BigQuery Python client uses the BigQuery Storage client by default.
# This currently causes an error on Terra Cloud Runtimes `the user does not have 'bigquery.readsessions.create'
# permission for '<Terra billing project id>'`. To work-around this uninstall the dependency so that flag
# `--use_rest_api` can be used with `%%bigquery` to use the older, slower mechanism for data transfer.
RUN pip3 uninstall -y google-cloud-bigquery-storage

ENV USER jupyter
USER $USER
ENV PIP_USER=true

RUN  jupyter nbextension install --user  --py vega \
  && jupyter nbextension enable --user  --py vega \
  && jupyter nbextension install --user  --py ipycanvas \
  && jupyter nbextension enable --user  --py ipycanvas
