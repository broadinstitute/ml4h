#!/bin/bash

# Other necessities
apt-get update
echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
apt-get install -y wget unzip curl python-pydot python-pydot-ng graphviz ttf-mscorefonts-installer

# Dependencies for graphics, 3-D outputs
apt-get install python3-tk libgl1-mesa-glx libxt-dev -y

# Dependencies to interact with S3 buckets
apt-get install s3fs s3cmd -y

apt-get install python3-pip -y

# Clean cache for leaner images
apt-get clean