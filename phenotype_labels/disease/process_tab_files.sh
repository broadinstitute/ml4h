#!/bin/bash
set -e

TAB_FILES_PATH="/Users/asmirnov/Desktop/ml4h/ukbb/tab_files/batch-13"
PROJECT="broad-ml4cvd"
DATABASE="ukbb7089_2024_01_20"
GO_CODE_PATH="/Users/asmirnov/Desktop/repos/ml4h/phenotype_labels/disease"
TEMP_OUTPUT_DIR="/Users/asmirnov/Desktop/ml4h/ukbb/temp_output/"

mkdir -p ${TEMP_OUTPUT_DIR}

if [ ! -d ${TAB_FILES_PATH} ]; then
    echo "Error: ${TAB_FILES_PATH} is not a directory!"
    exit 1
fi

cd ${GO_CODE_PATH}
for file in ${TAB_FILES_PATH}/*.tab; do
    # Check if it's a regular file or a directory
    file_name="$(basename "$file")"
	go run . -database "${PROJECT}.${DATABASE}" -materialized "${PROJECT}.${DATABASE}" -project "broad-ml4cvd" -override=true -tabfile ${file} > ${TEMP_OUTPUT_DIR}/${file_name}.tsv
	bq load --source_format=CSV --field_delimiter='\t' --autodetect ${DATABASE}.disease ${TEMP_OUTPUT_DIR}/${file_name}.tsv
    rm ${TEMP_OUTPUT_DIR}/${file_name}.tsv
done
