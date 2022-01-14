# Terra image

GitHub Action [docker-publish.yml](../../.github/workflows/docker-publish.yml) is used to publish a public copy of this container to [ghcr.io/broadinstitute/ml4h/ml4h_terra](https://github.com/orgs/broadinstitute/packages/container/package/ml4h%2Fml4h_terra).

If you wish to build your own container, you can use a command similar to the following to build and push to Google Container Registry:
```
gcloud --project YOUR-PROJECT-ID builds submit \
  --timeout 20m \
  --tag gcr.io/YOUR-PROJECT-ID/ml4h_terra:`date +"%Y%m%d_%H%M%S"` .
```

This image installs `tensorflow==2.4.3` (and `tensorflow-probability==0.12.2`), overriding the TF 2.5 specification in the [tensorflow-requirements.txt](../vm_boot_images/config/tensorflow-requirements.txt). This is because the TF 2.5 installation prevents GPUs from being used on Terra notebooks, due to notebook base image library incompatibilities.
See [this issue](https://github.com/DataBiosphere/terra-docker/issues/244) for more detail.  Once this issue is resolved, the Dockerfile will be updated to again use TF 2.5.
