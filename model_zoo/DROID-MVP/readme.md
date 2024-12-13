# DROID-MVP

1. Download docker image and pull github repo, including model checkpoints

`docker pull`

`github clone`

`git lfs pull`

2. Run docker container while mounting git folder

`docker run -it -v XXX:XXX droid:latest`

3. Run example inference script

`python droid_inference.py`

To use with your own data, format at 