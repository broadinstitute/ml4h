# ml4cvd
Machine Learning for CardioVascular Disease - MGH/MIT edition!

## Contents
- Setup
- Run scripts
- [Tensorize ECGs](docs/tensorize_ecgs.md)
- Contribute

## Setup
1. install [docker](https://docs.docker.com/get-docker/) and [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
2. add user to group to run docker
    ```
    sudo usermod -aG docker $USER
    ```
3. build docker image
    ```
    ./docker/build.sh [-c for CPU image]
    ```
4. setup conda environment to install pre-commit into
    ```
    make setup
    ```
5. activate conda environment so that pre-commit hooks are run
    ```
    conda activate ml4cvd
    ```

## Modes

#### train
Dynamically generate, train, and evaluate a machine learning model.
```
./scripts/run.sh -t $PWD/ml4cvd/recipes.py \
--mode train \
--tensors /path/to/data \
--input_tensors ecg_signal \
--output_tensors patient_outcome \
--output_folder results \
--id my-experiment
```

## Notebook
A Jupyter Notebook can be run inside docker containers with `ml4cvd` installed.
```
./scripts/run.sh -N [-p PORT, default is 8888]
```

If the notebook docker container is running locally, navigate to the link printed by the notebook server.
If the container is running remotely, a local port must be mapped to the remote port using an ssh tunnel before opening the link.
```
ssh -NL PORT:localhost:PORT USER@HOST
```

If changes to the code are made after a notebook is launched, update the package within the notebook by reinstalling and reimporting `ml4cvd`. The following code is run inside the notebook.
```
! pip install --user /path/to/ml/repo
import ml4cvd
```

## Run scripts
Run scripts are stored in [this Dropbox folder](https://www.dropbox.com/sh/hjz7adj01x1erfs/AABnZifp1mUqs7Z_26zm4ly9a?dl=0).

### Script dispatcher

To distribute training scripts across bootstraps and GPUs, use [`scripts/dispatch.py`](https://github.com/aguirre-lab/ml/blob/er_dispatcher/scripts/dispatch.py):

```zsh
python scripts/dispatch.py \
--gpus 0-3 \
--bootstraps 0-9 \
--scripts \
    ~/dropbox/ml4cvd_run_scripts/sts_ecg/train-simple.sh
    ~/dropbox/ml4cvd_run_scripts/sts_ecg/train-varied.sh
    ~/dropbox/ml4cvd_run_scripts/sts_ecg/train-deeper.sh
```

## Tests
From in the `ml` repo directory, run integration and pre-pytest unit tests:
```
scripts/run.sh -t $PWD/ml4cvd/tests.py
```

### Unit tests
To run unit tests in Docker:
```
/scripts/tf.sh -T $PWD/tests
```

Some of the unit tests are slow due to creating, saving and loading `tensorflow` models.
To skip those tests to move quickly, run
```
/scripts/tf.sh -T $PWD/tests -m "not slow"
```

pytest can also run specific tests using `::`. For example
```
python -m pytest $PWD/ml/tests/test_models.py::TestMakeMultimodalMultitaskModel::test_u_connect_segment
```

For more pytest usage information, checkout the [usage guide](https://docs.pytest.org/en/latest/usage.html).


## Contribute

Submit a new feature request, bug report, etc. by creating a [new issue with the ml4cvd template](https://github.com/aguirre-lab/ml/issues/new/choose).

TODO discuss PR and review norms
