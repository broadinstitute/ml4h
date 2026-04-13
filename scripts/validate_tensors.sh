# use this script to validate the tensors created by the tensorize.sh script
# expects two positional arguments: directory containing the tensors and the number of threads to use
# example: ./validate_tensors.sh /mnt/disks/tensors/ 20 | tee completed_tensors.txt
# the output will be in the following form:
# OK - /mnt/disks/tensors/ukb1234.hd5
# BAD - /mnt/disks/tensors/ukb5678.hd5


INPUT_TENSORS_DIR=$1
NUMBER_OF_THREADS=$2


find ${INPUT_TENSORS_DIR} | grep ".hd5" | \
    xargs -P ${NUMBER_OF_THREADS} -I {} \
        bash -c "h5dump -n {} | (grep -q 'HDF5 \"{}\"' && echo 'OK - {}' || echo 'BAD - {}')"
