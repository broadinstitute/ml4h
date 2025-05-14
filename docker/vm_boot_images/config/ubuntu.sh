#!/bin/bash

# Other necessities
apt-get update

echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

#wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
#dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
#cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-local-8138232B-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn
apt-get install -y wget unzip curl python3-pydot python3-pydot-ng graphviz ttf-mscorefonts-installer git pip ffmpeg
