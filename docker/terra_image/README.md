# Terra image

GitHub Action [docker-publish.yml](../../.github/workflows/docker-publish.yml) is used to publish a public copy of this container to [ghcr.io/broadinstitute/ml4h/ml4h_terra](https://github.com/orgs/broadinstitute/packages/container/package/ml4h%2Fml4h_terra).

If you wish to build your own container, you can use a command similar to the following to build and push to Google Container Registry:
```
gcloud --project YOUR-PROJECT-ID builds submit \
  --timeout 50m \
  --tag gcr.io/YOUR-PROJECT-ID/ml4h_terra:`date +"%Y%m%d_%H%M%S"` .
```

Note that use of this custom Terra image is optional. The ml4h setup is now brief enough that it is reasonable to put it all in [a setup notebook](../../notebooks/notebooks/terra_featured_workspace/ml4h_setup.ipynb).