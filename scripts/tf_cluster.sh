#!/usr/bin/env bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time 00:10:00
#SBATCH --job-name=ml4cvd_tf2
# Script to enable running Python modules within Singularity containers of Docker images

################### VARIABLES ############################################

# The default images are based on tensorflow/tensorflow:2.1.0
DOCKER_IMAGE_GPU="gitlab-registry.ccds.io/${USER}/ml4cvd:tf2-latest-gpu"
DOCKER_IMAGE_CPU="gitlab-registry.ccds.io/${USER}/ml4cvd:tf2-latest-cpu"
DOCKER_IMAGE=${DOCKER_IMAGE_GPU}
GPU_DEVICE="--nv"
INTERACTIVE=""
MOUNTS=""
PYTHON_COMMAND="python"
TEST_COMMAND="python -m pytest"

################### HELP TEXT ############################################

usage()
{
    cat <<USAGE_MESSAGE

    This script can be used to run a Python module within a Singularity container.

    Usage: tf_cluster.sh [-ctTh] [-i <image>] [-m <mount_directories>] module [arg ...]

    Example: ./tf_cluster.sh -i gitlab-registry.ccds.io/${USER}/ml4cvd:tf2-latest-gpu recipes.py --mode tensorize ...

        -c                  if set use CPU docker image and machine and use the regular 'docker' launcher.
                            By default, we assume the machine is GPU-enabled.

        -m <mount_dirs>     Directories to mount at the same path in the docker image.

        -b <sync_buckets>   Buckets to sync.

        -t                  Run Docker container interactively.

        -h                  Print this help text.

        -i <image>          Run Docker with the specified custom <image>. The default image is '${DOCKER_IMAGE}'.

        -T                  Run tests.
USAGE_MESSAGE
}

################### OPTION PARSING #######################################

while getopts ":i:m:b:cthT" opt ; do
    case ${opt} in
        h)
            usage
            exit 1
            ;;
        i)
            DOCKER_IMAGE=$OPTARG
            ;;
        m)
            MOUNTS="--bind ${OPTARG}:${OPTARG}"
            ;;
        b)
            MOUNT_BUCKETS=$OPTARG
            ;;
        c)
            DOCKER_IMAGE=${DOCKER_IMAGE_NO_GPU}
            GPU_DEVICE=
            ;;
        t)
            INTERACTIVE="-it"
            ;;
        T)
            PYTHON_COMMAND=${TEST_COMMAND}
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

if [[ $# -eq 0 ]]; then
    echo "ERROR: No Python module was specified." 1>&2
    usage
    exit 1
fi

################### SCRIPT BODY ##########################################

# Let anyone run this script
USER=$(whoami)
WORKDIR=$(pwd)

echo $WORKDIR

PYTHON_ARGS="$@"
cat <<LAUNCH_MESSAGE
Attempting to run singularity with
    singularity exec ${INTERACTIVE} \
        ${GPU_DEVICE} \
        ${MOUNTS} \
        docker://${DOCKER_IMAGE} /bin/bash -c \
            "pip install -e ${WORKDIR}/ml
            mkdir -p ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}; \
            s3fs -o use_path_request_style \
                 -o url=http://ogw.ccds.io \
                 -o passwd_file=${WORKDIR}/.passwd-s3fs \
                 ${MOUNT_BUCKETS} ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/; \
            ${PYTHON_COMMAND} ${PYTHON_ARGS}"
LAUNCH_MESSAGE

## Mount bucket
# s3cmd sync ${MOUNT_BUCKETS} ${SLURM_JOB_SCRATCHDIR}/
# echo ${SLURM_JOB_SCRATCHDIR}
# ls -l ${SLURM_JOB_SCRATCHDIR}

singularity exec ${INTERACTIVE} \
    ${GPU_DEVICE} \
    ${MOUNTS} \
    docker://${DOCKER_IMAGE} /bin/bash -c \
        "pip install -e ${WORKDIR}/ml; \
        mkdir -p ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}; \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_1.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_2.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_3.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_4.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_5.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_6.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_7.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_8.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        s3cmd sync s3://${MOUNT_BUCKETS}/mgh_9.tar ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/ & \
        wait; \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_1.tar & \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_2.tar & \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_3.tar & \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_4.tar & \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_5.tar & \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_6.tar & \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_7.tar & \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_8.tar & \
        tar xf ${SLURM_JOB_SCRATCHDIR}/${MOUNT_BUCKETS}/mgh_9.tar & \
        wait; \
        ${PYTHON_COMMAND} ${PYTHON_ARGS}"
