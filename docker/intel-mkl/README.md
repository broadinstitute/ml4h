# Tensorflow 1.15.0, Intel MKL, Python 3, and Jupyter notebook

Build this from the `ml4cvd/ml` directory

```bash
docker build -f docker/intel-mkl/Dockerfile -t ml4cvd-tensorflow-1.15.0-mkl-py3-jupyter:latest .
```

Run Jupyter instance as your current user to avoid saving files as `root`:

```bash
docker run -u $(id -u):$(id -g) -p 8888:8888 -v <local directory>:/<mount point> -it ml4cvd-tensorflow-1.15.0-mkl-py3-jupyter:latest
```
