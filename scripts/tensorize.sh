#!/usr/bin/env bash

# Script to generate tensors from UKBB data

################### VARIABLES ############################################

TENSOR_PATH=
NUM_JOBS=96
SAMPLE_IDS_START=1000000
SAMPLE_IDS_END=6030000
XML_FIELD=  # exclude ecg data
MRI_FIELD=  # exclude mri data
RECIPES_ARGS=
TENSORIZE_MODE="tensorize"


SCRIPT_NAME=$( echo $0 | sed 's#.*/##g' )

################### HELP TEXT ############################################

usage()
{
    cat <<USAGE_MESSAGE

    This script can be used to create tensors from the UKBB data.

    Usage: ${SCRIPT_NAME}    -t <tensor_path>
                          [-n <num_jobs>]
                          [-s <sample_id_start>]
                          [-e <sample_id_end>]
                          [-a <RECIPES_ARGS>]
                          [-m <tensorize mode>]
                          [-h]

    Example: ./${SCRIPT_NAME} -t /mnt/disks/data/generated/tensors/test/2019-02-05/ -n 96 -s 1000000 -e 6030000 -a "--xml_field_ids 20205 6025 --mri_field_ids 20208 20209"

        -t      <path>      (Required) Absolute path to directory to write output tensors to.

        -n      <num>       Number of jobs to run in parallel. Default: 96.

        -s      <id>        Smallest sample ID to start with. Default: 1000000.

        -e      <id>        Largest sample ID to start with. Default: 6030000.

        -a      <ids>       Argument string to pass directly to recipes.py

        -m      <mode>      Mode argument for recipes.py

        -h                  Print this help text

USAGE_MESSAGE
}

display_time() {
  local T=$1
  local D=$((T/60/60/24))
  local H=$((T/60/60%24))
  local M=$((T/60%60))
  local S=$((T%60))
  (( $D > 0 )) && printf '%d days ' $D
  (( $H > 0 )) && printf '%d hours ' $H
  (( $M > 0 )) && printf '%d minutes ' $M
  (( $D > 0 || $H > 0 || $M > 0 )) && printf 'and '
  printf '%d seconds\n' $S
}

################### OPTION PARSING #######################################

if [[ $# -eq 0 ]]; then
    echo "ERROR: No arguments were specified." 1>&2
    usage
    exit 1
fi

while getopts ":t:a:n:m:s:e:h" opt ; do
    case ${opt} in
        h)
            usage
            exit 1
            ;;
        t)
            TENSOR_PATH=$OPTARG
            ;;
        n)
            NUM_JOBS=$OPTARG
            ;;
        s)
            SAMPLE_IDS_START=$OPTARG
            ;;
        e)
            SAMPLE_IDS_END=$OPTARG
            ;;
        a)
            PYTHON_ARGS=$OPTARG
            ;;
        m)
            TENSORIZE_MODE=$OPTARG
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

# Keep track of and display the elapsed time
START_TIME=$(date +%s)

# Variables used to bin sample IDs so we can tensorize them in parallel
INCREMENT=$(( ( $SAMPLE_IDS_END - $SAMPLE_IDS_START ) / $NUM_JOBS ))
COUNTER=1
MIN_SAMPLE_ID=$SAMPLE_IDS_START
MAX_SAMPLE_ID=$(( $MIN_SAMPLE_ID + $INCREMENT - 1 ))

# We want to get the zip folder that was passes to recipes.py - look for the --zip_folder argument and extract the value passed after that
ZIP_FOLDER=$(echo ${PYTHON_ARGS} | sed 's/--zip_folder \([^ ]*\).*/\1/')
if [ ! -e $ZIP_FOLDER ]; then
    echo "ERROR: Zip folder passed was not valid, found $ZIP_FOLDER but expected folder path." 1>&2
    exit 1
fi

ZIP_FOLDER=$(echo ${PYTHON_ARGS} | sed 's/--zip_folder \([^ ]*\).*/\1/')

# create a directory in the /tmp/ folder to store some utilities for use later
mkdir -p /tmp/ml4h
# Write out a file with the ids of every sample in the input folder
echo "Gathering list of input zips to process between $MIN_SAMPLE_ID and $MAX_SAMPLE_ID, this takes several seconds..."
find $ZIP_FOLDER -name '*.zip' | xargs -I {} basename {} | cut -d '_' -f 1 \
                               | awk -v min="$MIN_SAMPLE_ID" -v max="$MAX_SAMPLE_ID" '$1 > min && $1 < max' \
                               | sort | uniq > /tmp/ml4h/sample_ids_trimmed.txt

echo "Including $(cat /tmp/ml4h/sample_ids_trimmed.txt | wc -l) samples in this tensorization job."


echo -e "\nLaunching job for sample IDs starting with $MIN_SAMPLE_ID and ending with $MAX_SAMPLE_ID via:"

# we need to run a command using xargs in parallel, and it gets rather complex and messy unless we can just run
# a shell script - the string below is written to a shell script that takes in positional arguments and sets
# min and max sample id to be the incoming sample id (min) to incoming sample id + 1 (max) - this lets us
# run on a single sample at a time
SINGLE_SAMPLE_SCRIPT='HOME=$1
                    TENSORIZE_MODE=$2
                    TENSOR_PATH=$3
                    SAMPLE_ID=$4
                    # drop those first 4 above and get all the rest of the arguments
                    shift 4
                    PYTHON_ARGS="$@"
                    python $HOME/ml4h/ml4h/recipes.py --mode $TENSORIZE_MODE \
        --tensors $TENSOR_PATH \
        --output_folder $TENSOR_PATH \
        $PYTHON_ARGS \
        --min_sample_id $SAMPLE_ID \
        --max_sample_id $(($SAMPLE_ID+1))'
echo "$SINGLE_SAMPLE_SCRIPT" > /tmp/ml4h/tensorize_single_sample.sh
chmod +x /tmp/ml4h/tensorize_single_sample.sh

# NOTE: the < " --version;  > below is very much a hack - it's a way to escape tf.sh's running "python" followed by 
#       whatever you pass with -c.  This causes it to run "python --version; " and then whatever you have after the semicolon.
read -r -d '' TF_COMMAND <<LAUNCH_CMDLINE_MESSAGE
    $HOME/ml4h/scripts/tf.sh -m "/tmp/ml4h/" -c " --version; \
        cat /tmp/ml4h/sample_ids_trimmed.txt | \
                xargs -P 11 -I {} /tmp/ml4h/tensorize_single_sample.sh $HOME $TENSORIZE_MODE $TENSOR_PATH {} $PYTHON_ARGS"
LAUNCH_CMDLINE_MESSAGE

echo "Executing command within tf.sh: $TF_COMMAND"
bash -c "$TF_COMMAND"


################### DISPLAY TIME #########################################

END_TIME=$(date +%s)
ELAPSED_TIME=$(($END_TIME - $START_TIME))
printf "\nDispatched $((COUNTER - 1)) tensorization jobs in "
display_time $ELAPSED_TIME
