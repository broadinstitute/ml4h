#!/usr/bin/env bash

# Source this once
# Exit and reconnect after adding any groups

# Allow this user to run docker images without sudo
sudo usermod -aG docker $(whoami)

# Install pre-commit
sudo apt install python3-pip
pip3 install pre-commit
pre-commit install
