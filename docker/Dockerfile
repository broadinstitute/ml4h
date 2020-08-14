ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="Steven Song <ssong20@mgh.harvard.edu>"

# Setup time zone (or else docker build hangs)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./config/* /app/
WORKDIR /app

# Basic setup
RUN ./ubuntu.sh

# Python packages
RUN pip3 install --upgrade pip
RUN pip3 install -r pre_requirements.txt
RUN pip3 install -r requirements.txt
