Release process


For PRs and after merge, testing is run with:
[python-package.yml](python-package.yml)



Manually navigate to GitHub's Releases page and select Draft a new release. 
https://github.com/broadinstitute/ml4h/releases

This process should automatically kick off the following workflows

Creation of updated docker images on CPU and GPU base images and published in GCR and GHCR
[publish-to-gcr-ghcr.yml](publish-to-gcr-ghcr.yml)

Images are named:
tf2.9-latest-cpu
tf2.9-latest-gpu
And can be found on [GitHubs Container Registry](https://github.com/broadinstitute/ml4h/pkgs/container/ml4h)

Updating of ml4h library and published to Pypi
[publish-to-pypi.yml](publish-to-pypi.yml)



