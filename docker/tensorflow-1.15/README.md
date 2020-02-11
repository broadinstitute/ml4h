# Tensorflow 1.15.0, Python 3, and Jupyter notebook with Nvidia GPU support

Build this from the `ml4cvd/ml` directory

```bash
docker build -f docker/tensorflow-1.15/Dockerfile -t ml4cvd-tensorflow-1.15.0-gpu-py3-jupyter:latest .
```

Run Jupyter instance as your current user to avoid saving files as `root`:

```bash
docker run --gpus all -u $(id -u):$(id -g) -p 8888:8888 -v <local directory>:/<mount point> -it ml4cvd-tensorflow-1.15.0-gpu-py3-jupyter:latest
```

If you get the error

```text
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
ERRO[0000] error waiting for container: context canceled
```

then install the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart) for Linux.
