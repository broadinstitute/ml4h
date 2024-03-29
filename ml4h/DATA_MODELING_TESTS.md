# Data/Modeling/Tests
## Running tests
Tests can be run in Docker with
```
${HOME}/ml4h/scripts/tf.sh -T ${HOME}/ml4h/tests
```
Tests can be run locally in a conda environment with
```
python -m pytest ${HOME}/ml4h/tests
```
Some of the tests are slow due to creating, saving and loading `tensorflow` models.
To skip those tests to move quickly, run
```
python -m pytest ${HOME}/ml4h/tests -m "not slow"
```
pytest can also run specific tests using `::`. For example
```
python -m pytest ${HOME}/ml4h/tests/test_models.py::TestMakeMultimodalMultitaskModel::test_u_connect_segment
```
For more pytest usage information, checkout the [usage guide](https://docs.pytest.org/en/latest/usage.html).

### Phenotypic SQLite database
Data for 500k people containing almost everything available in the UK Biobank Showcase

`/mnt/disks/data/raw/sql/ukbb7089.r10data.db`

To access the data using `sqlite`:

`sqlite3 /mnt/disks/data/raw/sql/ukbb7089.r10data.db`

The data can also be accessed through [BigQuery](https://console.cloud.google.com/bigquery?project=broad-ml4cvd&p=broad-ml4cvd&page=project).


### Cardiac MRI
212,158 individual zip files in ~20k people. Dicom-formatted files inside:

`/mnt/disks/data/raw/mris/cardiac/*.zip`

### Liver MRI
10,132 individual zip files in ~10k people. Dicom-formatted files inside:

`/mnt/disks/data/raw/mris/liver/*.zip`

### ECG: XML
119,097 ECGGs (12-lead resting and 3-lead exercise):

`/mnt/disks/data/raw/ecgs/*.xml`

### Direct Genotypes 
~800k/person:

`/mnt/imputed_v2`

### Imputed Genotypes
90 million/person:

`/mnt/imputed_v3`

## Modeling with TensorFlow
Once you have a virtual machine and an environment setup it is time to start learning.
The first step is to create training data by writing tensors to the disk.  

To write tensors with default categorical and continuous phenotypes, and no MRI or EKG data
```
${HOME}/ml/scripts/tf.sh ${HOME}/ml/ml4h/recipes.py --mode tensorize --tensors ${HOME}/my_tensors/ --max_sample_id 1003000 --mri_field_id  --xml_field_id
```
This should take about a minute to run and will output the SQL queries as well as the counts for the phenotype categories and responses that it finds.  Now let's train a model:
```
${HOME}/ml/scripts/tf.sh ${HOME}/ml/ml4h/recipes.py --mode train --tensors ${HOME}/my_tensors/ --input_tensors categorical-phenotypes-94 --output_tensors coronary_artery_disease_soft --id my_first_mlp_for_cvd
```
This model should achieve about 75% validation set accuracy on predicting from the phenotypes whether this person was labelled with an ICD code corresponding to cardivascular disease.
