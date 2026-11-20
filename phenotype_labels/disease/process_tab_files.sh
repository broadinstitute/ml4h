#!/bin/bash
set -e

TAB_FILES_PATH="/Users/asmirnov/Desktop/ml4h/ukbb/tab_files/batch-13"
PROJECT="broad-ml4cvd"
DATABASE="ukbb7089_2024_01_20"
GO_CODE_PATH="/Users/asmirnov/Desktop/repos/ml4h/phenotype_labels/disease"
TEMP_OUTPUT_DIR="/Users/asmirnov/Desktop/ml4h/ukbb/temp_output/"

# Check if use-gp-data positional argument is provided
if [ -z "$1" ]; then
    echo "Error: Please specify whether to use general practitioner (GP) data for generating disease table."
    echo "Usage: $0 <true|false>"
    exit 1
fi

# Check if the use-gp-data argument is either "true" or "false"
if [ "$1" != "true" ] && [ "$1" != "false" ]; then
    echo "Error: Invalid argument. Only 'true' or 'false' are allowed."
    echo "Usage: $0 <true|false>"
    exit 1
fi

USE_GP_DATA=$1
if [ ${USE_GP_DATA} == "false" ]; then
    DATABASE_NAME=${DATABASE}.disease
else
    DATABASE_NAME=${DATABASE}.disease_gp
fi

mkdir -p ${TEMP_OUTPUT_DIR}

if [ ! -d ${TAB_FILES_PATH} ]; then
    echo "Error: ${TAB_FILES_PATH} is not a directory!"
    exit 1
fi

cd ${GO_CODE_PATH}
for file in ${TAB_FILES_PATH}/*.tab; do
    # Check if it's a regular file or a directory
    file_name="$(basename "$file")"
	go run . -database "${PROJECT}.${DATABASE}" -materialized "${PROJECT}.${DATABASE}" -project "broad-ml4cvd" -override=true -use-gp-data=${USE_GP_DATA} -tabfile ${file} > ${TEMP_OUTPUT_DIR}/${file_name}.tsv
	bq load --source_format=CSV --field_delimiter='\t' --autodetect ${DATABASE_NAME} ${TEMP_OUTPUT_DIR}/${file_name}.tsv
    rm ${TEMP_OUTPUT_DIR}/${file_name}.tsv
done
