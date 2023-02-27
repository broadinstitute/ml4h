# ML4H
`ML4H` is a toolkit for machine learning on clinical data of all kinds including genetics, labs, imaging, clinical notes, and more. The diverse data modalities of biomedicine offer different perspectives on the underlying challenge of understanding human health. For this reason, `ML4H` is built on a foundation of multimodal multitask modeling, hoping to leverage all available data to help power research and inform clinical care. Our tools help apply clinical research standards to ML models by carefully considering bias and longitudinal outcomes. Our project grew out of efforts at the Broad Institute to make it easy to work with the UK Biobank on the Google Cloud Platform and has since expanded to include proprietary data from academic medical centers. To put cutting-edge AI and ML to use making the world healthier, we're fostering interdisciplinary collaborations across industry and academia.  We'd love to work with you too!    

`ML4H` is best described with Five Verbs: Ingest, Tensorize, TensorMap, Model, Evaluate
* Ingest: collect files onto one system
* Tensorize: write raw files (XML, DICOM, NIFTI, PNG) into HD5 files
* TensorMap: tag data (typically from an HD5) with an interpretation and a method for generation
* ModelFactory: connect TensorMaps with a trainable architectures
* Evaluate: generate plots that enable domain-driven inspection of models and results

# Getting Started
* [Setting up your local environment](#setting-up-your-local-environment)
* [Setting up a remote VM](#setting-up-a-remote-vm)
* Modeling/Data Sources/Tests [(`ml4h/DATA_MODELING_TESTS.md`)](ml4h/DATA_MODELING_TESTS.md)
* [Contributing Code](#contributing-code)
* [Command line interface](#command-line-interface)

Advanced Topics:
* Tensorizing Data (going from raw data to arrays suitable for modeling, in `ml4h/tensorize/README.md, TENSORIZE.md` )

## Setting up your local environment

Clone the repo to your home directory:
```
cd ~ \
git clone git clone https://github.com/broadinstitute/ml4h.git
```

Run the CPU docker (the first time you do this the docker will need to download which takes awhile).:
```
docker run -v ${HOME}:/home/ -it ghcr.io/broadinstitute/ml4h:tf2.9-latest-cpu
```

Then once inside the docker try to run the tests:
```
python -m pytest /home/ml4h/tests/test_recipes.py
```
If the tests pass (ignoring warnings) you're off to the races!
Next connect to some tensorized data, or checkout the introductory [Jupyter notebooks](notebooks/).


## Setting up your cloud environment (optional; currently only GCP is supported) 
Make sure you have installed the [Google Cloud SDK (gcloud)](https://cloud.google.com/sdk/docs/downloads-interactive). With [Homebrew](https://brew.sh/), you can use
```
brew cask install google-cloud-sdk
```

## Setting up a remote VM
To create a VM without a GPU run:
```
./scripts/vm_launch/launch_instance.sh ${USER}-cpu
```
With GPU (not recommended unless you need something beefy and expensive)
```
./scripts/vm_launch/launch_dl_instance.sh ${USER}-gpu
```
This will take a few moments to run, after which you will have a VM in the cloud.  Remember to shut it off from the command line or console when you are not using it!  

Now ssh onto your instance (replace with proper machine name, note that you can also use regular old ssh if you have the external IP provided by the script or if you login from the GCP console)
```
gcloud --project your-gcp-project compute ssh ${USER}-gpu --zone us-central1-a
```

Next, clone this repo on your instance (you should copy your github key over to the VM, and/or if you have Two-Factor authentication setup you need to generate an SSH key on your VM and add it to your github settings as described [here](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/#platform-linux)):
```
git clone git@github.com:broadinstitute/ml4h.git
```

Because we don't know everyone's username, you need to run one more script to make sure that you are added as a docker user and that you have permission to pull down our docker instances from GCP's gcr.io. Run this while you're logged into your VM:
```
./ml4h/scripts/vm_launch/run_once.sh
```

Note that you may see warnings like below, but these are expected:
```
WARNING: Unable to execute `docker version`: exit status 1
This is expected if `docker` is not installed, or if `dockerd` cannot be reached...
Configuring docker-credential-gcr as a registry-specific credential helper. This is only supported by Docker client versions 1.13+
/home/username/.docker/config.json configured to use this credential helper for GCR registries
```

You need to log out after that (`exit`) then ssh back in so everything takes effect.


### Finish setting up docker, test out a jupyter notebook
Now let's run a Jupyter notebook.  On your VM run:

```
${HOME}/ml4h/scripts/jupyter.sh -p 8889
```
Add a ```-c``` if you want a CPU version.

This will start a notebook server on your VM. If you a Docker error like
```
docker: Error response from daemon: driver failed programming external connectivity on endpoint agitated_joliot (1fa914cb1fe9530f6599092c655b7036c2f9c5b362aa0438711cb2c405f3f354): Bind for 0.0.0.0:8888 failed: port is already allocated.
```
overwrite the default port (8888) like so
```
${HOME}/ml4h/scripts/dl-jupyter.sh 8889
```
The command also outputs two command lines in red.
Copy the line that looks like this:
```
ssh -i ~/.ssh/google_compute_engine -nNT -L 8888:localhost:8888 <YOUR VM's IP ADDRESS>
```
Open a terminal on your local machine and paste that command.  

If you get a public key error run: `gcloud compute config-ssh`

Now open a browser on your laptop and go to the URL `http://localhost:8888`

## Contributing code

Want to contribute code to this project? Please see [CONTRIBUTING](./CONTRIBUTING.md) for developer setup and other details.

## Citation
If you use ML4H for research, you can use this citation format:
```
@misc{ml4h,
	title = {ml4h},
	copyright = {BSD 3-Clause License, 2021},
	url = {https://github.com/broadinstitute/ml4h},
	author = {{Data Sciences Platform at Broad Institute of MIT and Harvard}},
	abstract = {ML4H is a toolkit for machine learning on clinical data of all kinds including genetics, labs, imaging, clinical notes, and more.},
	urldate = {2021-03-31},
	publisher = {Broad Institute},
	month = mar,
	year = {2021},
	note = {original-date: 2019-04-10}
}
```

## Command line interface
The ml4h package is designed to be accessable through the command line using "recipes".
To get started, please see [RECIPE_EXAMPLES](./RECIPE_EXAMPLES.md).


[![DOI](https://zenodo.org/badge/180627543.svg)](https://zenodo.org/badge/latestdoi/180627543)
