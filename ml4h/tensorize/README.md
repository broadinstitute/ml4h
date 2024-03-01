# Run Dataflow
The following steps will run a Dataflow pipeline remotely, which in turn will, tensorize fields of a type
specified by the user (e.g. categorical, continuous) and write them onto a GCS bucket in the form of
one `hd5` file per sample id.

* Clone the repo and cd into it:
```
    git clone git@github.com:broadinstitute/ml4h.git
    cd ml4h
```

* Create and activate the right Python environment:
```
    conda env create -f ml4h/tensorize/dataflow/ml4h_dataflow.yml
    conda activate ml4h_dataflow
```

* Make sure you are authenticated by Google Cloud:
```
    gcloud auth application-default login
```

* Re install ml4h if you have made any changes:
```
    pip install .
```

* Run with the help option to see the list of command line arguments.
```
    python ml4h/tensorize/tensorize_dataflow.py -h
```

* Comment out the requirements in setup.py. Because some dataflow requirements conflict with ml4h base requirements you must comment out the lines (currently lines 6 and 16) in setup.py in the repo root:
```
requirements = (here / 'docker/vm_boot_images/config/tensorflow-requirements.txt').read_text(encoding='utf-8')
...
install_requires=requirements,
```
* Run the application to submit the pipeline to Dataflow to be executed remotely provided the command line argument `--beam_runner` is set to `DataflowRunner`. Set it to `DirectRunner` for local execution. For Example: 

```
python ml4h/tensorize/tensorize_dataflow.py  \
    --id example_id  \
    --tensor_type categorical \
    --bigquery_dataset example_dataset  \
    --beam_runner DataflowRunner \
    --repo_root /Users/johndoe/Dropbox/Code/ml4h \
    --gcs_output_path /path/to/Example_Folder
```

* Parameters of tensorize_dataflow.py:
  * id: The user-defined identifier for this pipeline run. **Note** that Google requires the `id` consist of only the characters `[-a-z0-9]`, i.e. starting with a letter and ending with a letter or number.

  * tensor_type: The type of data to be tensorized. Options are 'categorical', 'continuous', 'icd', 'disease', 'death', or 'phecode_disease'.

  * bigquery_dataset: The BigQuery dataset where the data will be drawn from. Defaults to 'ukbb_dev'.

  * beam_runner: The Apache Beam runner that will execute the pipeline. DataflowRunner is for remote execution. DirectRunner is for local execution.

  * repo_root: The root directory of the cloned ml repo.

  * gcp_project: The name of the Google Cloud Platform project. Defaults to "broad-ml4cvd". 

  * gcp_region: The Google Cloud Platform region. Defaults to "us-central1".

  * gcs_output_path: gs:// folder path excluding the bucket name where tensors will be written to. (e.g. specifying /path/to/folder will write to gs://<gcs_bucket>/path/to/folder)
  
  * logging_level: The Logging level the command should be run with. Options are "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL". Defaults to "info". 



* The pipeline can be run multiple times to tensorize different types of fields. This will populate the per-sample tensors
in specified GCS buckets. In order to unify them, they can be downloaded via `gsutil` as shown below
and merged using `merge_hd5s.py` script.
```
    gsutil -m cp -r <gcs bucket with tensors> <local directory>
```
