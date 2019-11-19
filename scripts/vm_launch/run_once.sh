#!/usr/bin/env bash

# Source this once
# Exit and reconnect after adding any groups

# Allow this user to run docker images without sudo
sudo usermod -aG docker $(whoami)

# Use the docker-credential-gcr that we installed on bootup
docker-credential-gcr configure-docker

# Install pre-commit
curl https://pre-commit.com/install-local.py | python -
pre-commit install
