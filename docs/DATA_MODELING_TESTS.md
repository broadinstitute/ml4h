# Data/Modeling/Tests

## Modeling with TensorFlow
Once you have an environment setup it is time to start learning.
The first step is to create training data by writing tensors to the disk.

See [tensorization docs](tensorize_ecgs.md) on how to tensorize ECGs.

To trian a model, run the command below in the root directory of the codebase:
```bash
./scripts/tf.sh $PWD/ml4cvd/recipes.py --mode train --tensors /path/to/tensors --input_tensors ecg_2500 --output_tensors ecg_rate --output_foler /path/to/output --id my_results
```

## Running tests
### Integration tests
The command below will run integration tests (and some pre-pytest unit tests):
```
${HOME}/ml/scripts/tf.sh -t ${HOME}/ml/ml4cvd/tests.py
```

### Unit tests
Unit tests can be run in Docker with
```
${HOME}/ml/scripts/tf.sh -T ${HOME}/ml/tests
```
Unit tests can be run locally in a conda environment with
```
python -m pytest ${HOME}/ml/tests
```
Some of the unit tests are slow due to creating, saving and loading `tensorflow` models.
To skip those tests to move quickly, run
```
python -m pytest ${HOME}/ml/tests -m "not slow"
```
pytest can also run specific tests using `::`. For example
```
python -m pytest ${HOME}/ml/tests/test_models.py::TestMakeMultimodalMultitaskModel::test_u_connect_segment
```
For more pytest usage information, checkout the [usage guide](https://docs.pytest.org/en/latest/usage.html).
