# Command line recipes introduction
All of the functionalilty of the `ml4h` package is available from the command line through [recipes](ml4h/recipes.py).
Which `recipe` you use is specified by `--mode`.
E.g.
```bash
python ml4h/recipes.py --mode train ...
```
or
```bash
python ml4h/recipes.py --mode explore ...
```

The command line arguments are specified in [arguments](ml4h/arguments.py).
Almost all of the arguments are optional and depend on the `recipe` you want to run.

# Index
Pipelines:
* [Tensorization](#tensorization)
* [Modeling](#modeling)

Recipes modes:
* [explore](#explore)

# Examples

## Tensorization
TODO

## Modeling
For all of our modeling examples, we will use MNIST data, which requires you to have MNIST data in `hd5` format.
To set that up, run the [MNIST demo](notebooks/mnist_demo.ipynb) at least through **Tensorization**.

You also should have docker set up following the instructions in the [readme](README.md).
You can run the recipes from the docker image
```bash
cd [path_to_repo]/ml  # navigate to repo
docker run -it --rm --ipc=host -v $PWD:$PWD gcr.io/broad-ml4cvd/deeplearning:tf2-latest-cpu  # enter cpu docker image
cd [path_to_repo]/ml  # navigate to repo in docker
pip install .  # install ml4h package
```
To run recipes with the gpu, use
```bash
docker run --gpus -it --rm --ipc=host -v $PWD:$PWD gcr.io/broad-ml4cvd/deeplearning:tf2-latest-gpu
```

### explore
The first step of modeling is to explore your dataset.
```bash
python ml4h/recipes.py --mode explore --input_tensors
```


### train

### test

### compare

### infer
