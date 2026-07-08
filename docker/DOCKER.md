# ML4H Docker

## Editing and pushing the docker

To edit the packages inside the ML4H docker container, first edit:
```
ml4h/ml4h/docker/vm_boot_images/config/tensorflow-requirements.txt
```
Add a line for each package, with optional version numbers.

Then, the docker container should be pushed to both the [Google Container Registry](https://console.cloud.google.com/gcr/images/broad-ml4cvd/GLOBAL/deeplearning) and the [Github GHCR Repository](https://github.com/broadinstitute/ml4h/pkgs/container/ml4h).

For GHCR, you will to generate a [personal access token](https://github.com/settings/tokens) on github, and grant docker access:
```
docker login ghcr.io -u GITHUB_USERNAME -p ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Finally, use the ```jupyter.sh``` script to build, tag and push an ML4H image. Use ```-c``` for the CPU-only image:
```
cd ml4h
./docker/vm_boot_images/build.sh -P
./docker/vm_boot_images/build.sh -c -P
```
Note that each image will have two tags: a short unique SHA1 tag from ```HEAD```, and either ```tf2.9-latest-gpu``` or ```tf2.9-latest-cpu```.
