import os
import csv
import logging
import operator
import numpy as np
from typing.io import TextIO
from typing import List, Tuple

from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.tensormap.general import build_tensor_from_file
from ml4h.DatabaseClient import BigQueryDatabaseClient, DatabaseClient
from ml4h.defines import TENSOR_MAPS_FILE_NAME, dataset_name_from_meaning
from ml4h.defines import DICTIONARY_TABLE, CODING_TABLE, PHENOTYPE_TABLE, JOIN_CHAR
from ml4h.tensormap.text import random_text_window_tensor, token_dictionary_and_text_from_file
from ml4h.tensorize.tensor_writer_ukbb import disease_prevalence_status, get_disease2tsv, disease_incidence_status, disease_censor_status


LESS_THAN_CODES = "('Less than a year', 'Less than once a week', 'Less than one mile', 'Less than an hour a day', 'Less than one a day', 'Less than one', 'Less than once a year', 'Less than 1 year ago', 'Less than a year ago', 'Less than one year', 'Less than one cigarette per day')"


def write_tensor_maps(args) -> None:
    logging.info("Making tensor maps...")

    tensor_maps_file = f"{args.output_folder}/{TENSOR_MAPS_FILE_NAME}.py"
    db_client = BigQueryDatabaseClient(credentials_file=args.bigquery_credentials_file)
    with open(tensor_maps_file, 'w') as f:
        f.write(_get_tensor_map_file_imports())
        _write_disease_tensor_maps(args.phenos_folder, f)
        _write_disease_tensor_maps_incident_prevalent(args.phenos_folder, f)
        _write_phecode_tensor_maps(f, args.phecode_definitions, db_client)
        _write_continuous_tensor_maps(f, db_client)

        f.write('\n')
        logging.info(f"Wrote the tensor maps to {tensor_maps_file}.")


def _get_tensor_map_file_imports() -> str:
    return (
        f"# TensorMaps automatically generated by {os.path.basename(__file__)}\n"
        f"# DO NOT EDIT\n\n"
        f"from ml4h.defines import StorageType\n"
        f"from ml4h.metrics import weighted_crossentropy\n"
        f"from ml4h.tensormap.ukb.demographics import prevalent_incident_tensor\n"
        f"from ml4h.TensorMap import TensorMap, Interpretation, make_range_validator\n\n"
    )


def _write_disease_tensor_maps(phenos_folder: str, f: TextIO) -> None:
    f.write(f"\n\n#  TensorMaps for MPG disease phenotypes\n")
    disease2tsv = get_disease2tsv(phenos_folder)
    logging.info(f"Got:{len(disease2tsv)} disease TSVs from:{phenos_folder}")
    status = disease_censor_status(disease2tsv, 1000000, 5000000)
    logging.info(f"Got status for all diseases.")
    for d in sorted(list(disease2tsv.keys())):
        total = len(status[d])
        diseased = np.sum(list(status[d].values()))
        factor = int(total / (1 + diseased * 2))
        f.write(
            f"{d} = TensorMap('{d}', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_FLAG, path_prefix='categorical', "
            f"channel_map={{'no_{d}':0, '{d}':1}}, loss=weighted_crossentropy([1.0, {factor}], '{d}'))\n",
        )
    logging.info(f"Done writing TensorMaps for diseases.")


def _write_disease_tensor_maps_incident_prevalent(phenos_folder: str, f: TextIO) -> None:
    f.write(f"\n\n#  TensorMaps for prevalent and incident MPG disease phenotypes\n")
    disease2tsv = get_disease2tsv(phenos_folder)
    logging.info(f"Got:{len(disease2tsv)} disease TSVs from:{phenos_folder}")
    status_p = disease_prevalence_status(disease2tsv, 1000000, 2500000)
    status_i = disease_incidence_status(disease2tsv, 1000000, 2500000)
    logging.info(f"Got prevalence and incidence status for all diseases.")
    for disease in sorted(list(disease2tsv.keys())):
        total = len(status_p[disease])
        diseased_p = np.sum(list(status_p[disease].values()))
        factor_p = int(total / (1 + (diseased_p * 3)))
        diseased_i = np.sum(list(status_i[disease].values()))
        factor_i = int(total / (1 + (diseased_i * 3)))
        f.write(
            f"{disease}_prevalent_incident = TensorMap('{disease}', Interpretation.CATEGORICAL,  storage_type=StorageType.CATEGORICAL_FLAG, "
            f"path_prefix='categorical', tensor_from_file=prevalent_incident_tensor('dates/enroll_date', 'dates/{disease}_date'), "
            f"channel_map={{'no_{disease}':0, 'prevalent_{disease}':1, 'incident_{disease}':2}}, "
            f"loss=weighted_crossentropy([1.0, {factor_p}, {factor_i}], '{disease}_prevalent_incident'))\n",
        )
    logging.info(f"Done writing TensorMaps for prevalent and incident diseases.")


def _write_phecode_tensor_maps(f: TextIO, phecode_csv, db_client: DatabaseClient):
    # phecode_csv = '/home/sam/phecode_definitions1.2.csv'
    total_samples = 500000
    remove_chars = ";.,/()-[]&' "
    phecode2phenos = {}
    with open(phecode_csv, 'r') as my_csv:
        lol = list(csv.reader(my_csv, delimiter=','))
        for row in lol[1:]:
            pheno = row[1].strip().replace("'s", "s")
            for c in remove_chars:
                pheno = pheno.replace(c, '_')
            pheno = pheno.lower().strip('_').replace('___', '_').replace('__', '_')
            phecode2phenos['phecode_'+row[0].lstrip('0').strip()] = pheno
    query = f"select disease, count(disease) as total from `broad-ml4h.ukbb7089_201904.phecodes_nonzero` GROUP BY disease"
    count_result = db_client.execute(query)
    phecode2counts = {}
    for row in count_result:
        phecode2counts[row['disease']] = float(row['total'])

    f.write(f"\n\n#  TensorMaps for Phecode disease phenotypes\n")
    for k, p in sorted(phecode2phenos.items(), key=operator.itemgetter(1)):
        if k in phecode2counts:
            factor = int(total_samples / (1+phecode2counts[k]))
            f.write(
                f"{p}_phe = TensorMap('{k}', Interpretation.CATEGORICAL, channel_map={{'no_{p}':0, '{p}':1}}, path_prefix='categorical', "
                f"storage_type=StorageType.CATEGORICAL_FLAG, loss=weighted_crossentropy([1.0, {factor}], '{k.replace('.', '_')}'))\n",
            )

    query = f"select disease, count(disease) as total from `broad-ml4h.ukbb7089_201904.phecodes_nonzero` WHERE prevalent_disease=1 GROUP BY disease"
    count_result = db_client.execute(query)
    phecode2prevalent = {}
    for row in count_result:
        phecode2prevalent[row['disease']] = float(row['total'])

    query = f"select disease, count(disease) as total from `broad-ml4h.ukbb7089_201904.phecodes_nonzero` WHERE incident_disease=1 GROUP BY disease"
    count_result = db_client.execute(query)
    phecode2incident = {}
    for row in count_result:
        phecode2incident[row['disease']] = float(row['total'])

    f.write(f"\n\n#  TensorMaps for prevalent and incident Phecode disease phenotypes\n")
    for k, p in sorted(phecode2phenos.items(), key=operator.itemgetter(1)):
        if k in phecode2incident and k in phecode2prevalent:
            factor_i = int(total_samples / (1 + phecode2incident[k]))
            factor_p = int(total_samples / (1 + phecode2prevalent[k]))
            f.write(
                f"{p}_phe_pi = TensorMap('{k}',  Interpretation.CATEGORICAL,  storage_type=StorageType.CATEGORICAL_FLAG, "
                f"path_prefix='categorical', tensor_from_file=prevalent_incident_tensor('dates/enroll_date', 'dates/{k}_date'), "
                f"channel_map={{'no_{p}':0, '{p}_prevalent':1, '{p}_incident':2}}, "
                f"loss=weighted_crossentropy([1.0, {factor_p}, {factor_i}], '{p}_pi'))\n",
            )


def _write_continuous_tensor_maps(f: TextIO, db_client: DatabaseClient):
    # Handle special coding values in continuous variables in order to generate summary statistics (mean and std dev) for
    # each field across all samples. This will remove missing samples from the calculation and change the value of 'Less than one'
    query = f"""
    WITH coding_tmp AS (
        SELECT
            *,
            CASE
                WHEN meaning IN ('Do not know',  'Prefer not to answer', 'Ongoing when data entered') OR meaning LIKE "Still taking%" THEN TRUE
            END AS missing,
            CASE
                WHEN meaning IN {LESS_THAN_CODES} THEN '.5'
            END AS value
        FROM
            {CODING_TABLE}
    ), pheno_tmp AS (
    SELECT
        sample_id,
        FieldID,
        instance,
        array_idx,
        COALESCE(c.value, p.value) new_value,
        COALESCE(c.missing, FALSE) missing
    FROM {PHENOTYPE_TABLE} AS p
    LEFT JOIN coding_tmp AS c
        ON TRUE
        AND SAFE_CAST(p.value AS FLOAT64) = SAFE_CAST(c.coding AS FLOAT64)
        AND p.coding_file_id = c.coding_file_id
    )

    SELECT
        t.FieldID,
        Field,
        t.instance,
        AVG(CAST(new_value AS FLOAT64)) mean,
        STDDEV(CAST(new_value AS FLOAT64)) std,
        MAX(array_idx) AS max_array
    FROM pheno_tmp AS t
    LEFT JOIN {DICTIONARY_TABLE} AS d ON d.FieldID = t.FieldID
    WHERE TRUE
        AND ValueType IN ('Integer', 'Continuous')
        AND NOT missing
    GROUP BY t.FieldID, t.instance, Field ORDER BY t.FieldID
    """

    field_data_for_tensor_maps = db_client.execute(query)

    f.write(f"\n\n#  Continuous tensor maps\n")
    for row in field_data_for_tensor_maps:
        name = dataset_name_from_meaning(None, [str(row.FieldID), row.Field, str(row.instance)])
        channel_map = "channel_map={"
        for i in range(0, row.max_array + 1):
            channel_map += f"'{name}{JOIN_CHAR}{i}': {i}, "
        channel_map += "}"
        f.write(f"ukb_{row.FieldID}_{row.instance} = TensorMap('{name}{JOIN_CHAR}{i}', loss='logcosh', path_prefix='continuous', ")
        f.write(f"normalization={{'mean': {row.mean}, 'std': {row.std}}}, annotation_units={row.max_array+1}, {channel_map})\n")


def _get_pkl_path_for_field(field_id: int, pyukbb_data_path: str):
    """Returns the path to the .pkl file that contained `UKBioBankParsedField` for a given FieldID."""
    for _, _, files in os.walk(pyukbb_data_path):
        for file in files:
            if file.find(f'FieldID_{field_id}.pkl') > -1:
                return os.path.join(pyukbb_data_path, file)
    raise FileNotFoundError('Cannot find pyukbb .pkl file for field ID {field_id}!')


def _get_all_available_fields(available_fields_pd, keyword: str = None, category: int = None):
    filtered = available_fields_pd
    if category is not None:
        filtered = filtered[filtered.Category == category]
    if keyword is not None:
        filtered = filtered[filtered.Field.str.contains(keyword, case=False)]
    return filtered


def generate_continuous_tensor_map_from_file(
    file_name: str,
    column_name,
    tensor_map_name: str,
    normalization: bool,
    discretization_bounds: List[float],
) -> TensorMap:
    if discretization_bounds:
        return TensorMap(
            f'{tensor_map_name}', Interpretation.DISCRETIZED, channel_map={tensor_map_name: 0},
            tensor_from_file=build_tensor_from_file(file_name, column_name, normalization),
            discretization_bounds=discretization_bounds,
        )
    else:
        return TensorMap(
            f'{tensor_map_name}', channel_map={tensor_map_name: 0},
            tensor_from_file=build_tensor_from_file(file_name, column_name, normalization),
        )


def generate_random_text_tensor_maps(text_file: str, window_size: int, one_hot: bool = True) -> Tuple[TensorMap, TensorMap]:
    name = os.path.basename(text_file).split('.')[0]
    text, token_dictionary = token_dictionary_and_text_from_file(text_file)
    shape = (window_size, len(token_dictionary)) if one_hot else (window_size,)
    burn_in = TensorMap(
        f'next_{name}', Interpretation.LANGUAGE, shape=shape,
        channel_map=token_dictionary,
        cacheable=False,
    )
    output_map = TensorMap(
        f'next_next_{name}', Interpretation.LANGUAGE,
        shape=(len(token_dictionary),) if one_hot else shape,
        loss='categorical_crossentropy',
        channel_map=token_dictionary,
        cacheable=False,
    )
    input_map = TensorMap(
        name, Interpretation.LANGUAGE, shape=shape,
        tensor_from_file=random_text_window_tensor(text, window_size, one_hot=one_hot),
        dependent_map=[burn_in, output_map],
        channel_map=token_dictionary,
        annotation_units=128,
        cacheable=False,
    )
    return input_map, burn_in, output_map
