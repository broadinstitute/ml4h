# This setup script is separated from the default because it seems that there
# may be issues with GPU-enabled docker machines if you run on a non-GPU
# instance. See https://github.com/NVIDIA/nvidia-docker/issues/652 It should not
# be run on a blank Ubuntu 18.04 VM, but instead on a ml4cvd-image. It is 
# a 2-part script.

# 
# 2018/09/11 additions
# Enable NVidia-docker
# Via https://askubuntu.com/a/1036265/411855
# 
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Reboot. Then return and move on to part 2
sudo reboot

# # Prerequisites: Get the build tools so you can install gcc
# # See https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions
# sudo apt-get install -y build-essential
# sudo apt-get install -y linux-headers-$(uname -r)
# sudo dpkg -i cuda-repo-<distro>_<version>_<architecture>.deb
# # Install cuda
# sudo apt-get install cuda

# # Add the package repositories
# curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
#   sudo apt-key add -
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
#   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
# sudo apt-get update

# # Install nvidia-docker2 and reload the Docker daemon configuration
# sudo apt-get install -y nvidia-docker2
# sudo pkill -SIGHUP dockerd
