#!/bin/bash

# Other necessities
apt-get update
echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
apt-get install -y wget unzip curl python3-pydot python3-pydot-ng graphviz ttf-mscorefonts-installer git pip