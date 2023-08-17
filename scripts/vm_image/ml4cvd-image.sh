#!/usr/bin/env bash

# server-conf-scripts are for configuration of a *fresh* VM and should not be
# treated as startup scripts. (They are not idempotent.)

GCP_BUCKET="ml4h-core"

# We assume we are running as a regular user, not root.

# Enable gcsfuse to allow mounting of the google storage bucket as if it were a drive
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install frequently-used packages
# First, update apt since we have added new repos (above)
sudo apt-get update

sudo apt install -y r-base r-base-core unzip wget bzip2 python sqlite3 gcsfuse

# Make gcsfuse auto-mount to /mnt/${GCP_BUCKET} in the future. Modify fstab to
# do this automatically. Via
# https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/mounting.md
# and https://serverfault.com/a/830726/118452 to enable easier mount with read and
# write access by non-root users.
echo "${GCP_BUCKET} /mnt/${GCP_BUCKET} gcsfuse rw,allow_other,implicit_dirs,default_permissions,file_mode=777,dir_mode=777" | sudo tee -a /etc/fstab
echo "fc-9a7c5487-04c9-4182-b3ec-13de7f6b409b /mnt/imputed_v2 gcsfuse ro,allow_other,implicit_dirs,default_permissions,file_mode=777,dir_mode=777" | sudo tee -a /etc/fstab
echo "fc-7d5088b4-7673-45b5-95c2-17ae00a04183 /mnt/imputed_v3 gcsfuse ro,allow_other,implicit_dirs,default_permissions,file_mode=777,dir_mode=777" | sudo tee -a /etc/fstab


# Enable docker (assumes Ubuntu, of any supported version)
# See https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo service docker start

sudo systemctl enable docker
sudo groupadd -f docker

# Manually install gcr
# Via https://cloud.google.com/container-registry/docs/advanced-authentication#standalone_docker_credential_helper
VERSION=1.5.0
OS=linux
ARCH=amd64
curl -fsSL "https://github.com/GoogleCloudPlatform/docker-credential-gcr/releases/download/v${VERSION}/docker-credential-gcr_${OS}_${ARCH}-${VERSION}.tar.gz" \
  | tar xz --to-stdout ./docker-credential-gcr | sudo tee -a /usr/bin/docker-credential-gcr 1>/dev/null && sudo chmod +x /usr/bin/docker-credential-gcr
docker-credential-gcr configure-docker

sudo apt-get install -y python-setuptools


#
# Do last
#

# Cleanup apt cache
sudo apt autoremove -y
