#!/usr/bin/env bash

# Script to enable running Python modules within Docker containers
# Note: If 'nvidia-docker' is specified as $DOCKER_COMMAND, this script must be run on a dl-image machine
# (rather than a 'ukbb-image' machine) on a GPU-enabled machine.

################### VARIABLES ############################################

# The default images are based on ufoym/deepo:all-py36-jupyter
DOCKER_IMAGE_GPU="gitlab-registry.ccds.io/paolo.achille/ml4cvd:tf2-latest-gpu"
DOCKER_IMAGE_CPU="gitlab-registry.ccds.io/paolo.achille/ml4cvd:tf2-latest-cpu"
DOCKER_IMAGE=${DOCKER_IMAGE_GPU}
DOCKER_COMMAND="docker"
PORT="8888"
SCRIPT_NAME=$( echo $0 | sed 's#.*/##g' )
GPU_DEVICE="--nv"

################### HELP TEXT ############################################

usage()
{
    cat <<USAGE_MESSAGE

    This script can be used to run a Jupyter server ona VM with a tunnel to specific port.

    Usage: ${SCRIPT_NAME} [-nth] [-i <image>] module [arg ...]

    Example: ./${SCRIPT_NAME} -n -p 8889  -i gcr.io/broad-ml4cvd/deeplearning:latest-cpu

        -c                  Use CPU docker image and use the regular 'docker' launcher.
                            By default, 'nvidia-docker' wrapper is used to launch Docker assuming the machine is GPU-enabled.

        -h                  Print this help text.

        -p                  Port to use, by default '${PORT}'

        -i      <image>     Run Docker with the specified custom <image>. The default image is '${DOCKER_IMAGE}'.
USAGE_MESSAGE
}

################### OPTION PARSING #######################################

while getopts ":ip:b:ch" opt ; do
    case ${opt} in
        h)
            usage
            exit 1
            ;;
        i)
            DOCKER_IMAGE=$OPTARG
            ;;
        b)
            MOUNT_BUCKETS=$OPTARG
            ;;        
        p)
            PORT=$OPTARG
            ;;
        c)
            DOCKER_IMAGE=${DOCKER_IMAGE_NO_GPU}
	    GPU_DEVICE=""
            ;;
        :)
            echo "ERROR: Option -${OPTARG} requires an argument." 1>&2
            usage
            exit 1
            ;;
        *)
            echo "ERROR: Invalid option: -${OPTARG}" 1>&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))


################### SCRIPT BODY ##########################################

# Let anyone run this script
USER=$(whoami)
WORKDIR=$(pwd)

cd $SLURM_JOB_SCRATCHDIR
cp -r /home/${USER}/.mc ./
# /home/${USER}/mc cp --recursive ccds/${MOUNT_BUCKETS}/ ./

# for i in $(ls *.tar) 
# do
#     tar xf $i & 
# done
# wait
# rm *.tar

singularity exec \
    ${GPU_DEVICE} \
    ${MOUNTS} \
    docker://${DOCKER_IMAGE} /bin/bash -c \
        "pip install -e /home/$USER/ml4cvd; \
         export MOUNT_BUCKETS=${MOUNT_BUCKETS};
         jupyter notebook --no-browser --ip 0.0.0.0 --port=${PORT} --NotebookApp.token= --notebook-dir=/home/$USER"
        "