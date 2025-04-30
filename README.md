# ML4H
`ML4H` is a toolkit for machine learning on clinical data of all kinds including genetics, labs, imaging, clinical notes, and more. The diverse data modalities of biomedicine offer different perspectives on the underlying challenge of understanding human health. For this reason, `ML4H` is built on a foundation of multimodal multitask modeling, hoping to leverage all available data to help power research and inform clinical care. Our tools help apply clinical research standards to ML models by carefully considering bias and longitudinal outcomes. Our project grew out of efforts at the Broad Institute to make it easy to work with the UK Biobank on the Google Cloud Platform and has since expanded to include proprietary data from academic medical centers. To put cutting-edge AI and ML to use making the world healthier, we're fostering interdisciplinary collaborations across industry and academia.  We'd love to work with you too!    

`ML4H` is best described with Five Verbs: Ingest, Tensorize, TensorMap, Model, Evaluate
* **Ingest**: collect files onto one system
* **Tensorize**: write raw files (XML, DICOM, NIFTI, PNG) into HD5 files
* **TensorMap**: tag data (typically from an HD5) with an interpretation and a method for generation
* **ModelFactory**: connect TensorMaps with a trainable neural network architecture loss function, and optimization strategy
* **Evaluate**: generate plots that enable domain-driven inspection of models and results

# Getting Started
* [Setting up your local environment](#setting-up-your-local-environment)
* [Setting up a remote VM](#setting-up-a-remote-vm)
* Modeling/Data Sources/Tests [(`ml4h/DATA_MODELING_TESTS.md`)](ml4h/DATA_MODELING_TESTS.md)
* [Contributing Code](#contributing-code)
* [Releases and Versioning](#releases)
* [Command line interface](#command-line-interface)

Advanced Topics:
* Tensorizing Data (going from raw data to arrays suitable for modeling, in `ml4h/tensorize/README.md, TENSORIZE.md` )

## Setting up your local environment

Clone the repo to your home directory:
```
cd ~ \
git clone https://github.com/broadinstitute/ml4h.git
```

Run the CPU docker (this step does not work on Apple silicon). The first time you do this the docker will need to download which takes awhile:
```
docker run -v ${HOME}:/home/ -it ghcr.io/broadinstitute/ml4h:tf2.9-latest-cpu
```

Then once inside the docker try to run the tests (again, not on Apple silicon):
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

Make sure you have [configured your development environment](https://cloud.google.com/api-gateway/docs/configure-dev-env#prerequisites). In particular, you will probably have to complete the steps to [prepare the Google Cloud CLI](https://cloud.google.com/api-gateway/docs/configure-dev-env#preparing_the_for_deployment) and [enable the required Google services](https://cloud.google.com/api-gateway/docs/configure-dev-env#enabling_required_services).


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

Now ssh onto your instance (replace with proper machine name and project name, note that you can also use regular old ssh if you have the external IP provided by the script or if you login from the GCP console)
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
${HOME}/ml4h/scripts/jupyter.sh
```
Add a ```-c``` if you want a CPU version.

This will start a notebook server on your VM. If you a Docker error like
```
docker: Error response from daemon: driver failed programming external connectivity on endpoint agitated_joliot (1fa914cb1fe9530f6599092c655b7036c2f9c5b362aa0438711cb2c405f3f354): Bind for 0.0.0.0:8888 failed: port is already allocated.
```
overwrite the default port (8888) like so
```
${HOME}/ml4h/scripts/jupyter.sh -p 8889
```
The command also outputs two command lines in red.
Copy the line that looks like this:
```
gcloud compute ssh ${USER}@${USER}-gpu -- -NnT -L 8889:localhost:8889
```
Open a terminal on your local machine and paste that command.  

If you get a public key error run: `gcloud compute config-ssh`

Now open a browser on your laptop and go to the URL `http://localhost:8888`


### Attach a disk with tensorized data to a VM
Running

```
gcloud compute instances attach-disk my-vm --disk my-disk --device-name my-disk --mode rw --zone us-central1-a
```

will attach `my-disk` to `my-vm`. You can also do this by clicking edit on VM in the [GCP console](https://console.cloud.google.com/compute/instances). Then selecting Attach Existing Disk. If you create a new disk you will also need to format it.

Now we can mount the disk.   

### Mount the disk
SSH into your VM and run:
```
ls -l /dev/disk/by-id
```

That will output something like

```
total 0
lrwxrwxrwx 1 root root  9 Feb 11 19:13 google-mri-october -> ../../sdb
lrwxrwxrwx 1 root root  9 Feb 15 21:42 google-my-disk -> ../../sdd
lrwxrwxrwx 1 root root  9 Feb 11 19:13 scsi-0Google_PersistentDisk_mri-october -> ../../sdb
lrwxrwxrwx 1 root root  9 Feb 15 21:42 scsi-0Google_PersistentDisk_my-disk -> ../../sdd
``` 

The line that contains the name of our disk (`my-disk`) 

```
lrwxrwxrwx 1 root root  9 Feb 15 21:42 google-my-disk -> ../../sdd
```

indicates that our disk was assigned to the device named `sdd`. The subsequent steps will use this device name.
Make sure to replace it with yours, if different.
 
Next, we'll create a directory that will serve as the mount point for the new disk:

```
sudo mkdir -p /mnt/disks/my-disk
```

The directory name doesn't have to match the name of the disk but it's one less name to keep track of that way.

Finally, we can do the mounting:

```
sudo mount -o norecovery,discard,defaults /dev/sdd /mnt/disks/my-disk
```

We will also add the persistent disk to the `/etc/fstab` file so that the device automatically mounts again 
if/when the VM restarts:

```
echo UUID=`sudo blkid -s UUID -o value /dev/sdd` /mnt/disks/my-disk ext4 norecovery,discard,defaults,nofail 0 2 | sudo tee -a /etc/fstab
```

If you detach this persistent disk or create a snapshot from the boot disk for this instance, edit the `/etc/fstab`
file and remove the entry for the disk. Even with the `nofail` or `nobootwait` options in place,
keep the `/etc/fstab` file in sync with the devices that are attached to your instance.

Voilà! You now have a shiny new disk where you can persist the tensors you will generate next.


### Set up VScode to connect to the GCP VM (which makes your coding much easier)


step 1: install VSdoe

step 2:config the ssh key
gcloud compute config-ssh --project "broad-ml4cvd"

Step 3: install remote-SSH extension in VS Code

Step 4: connect to the VM by pressing F1 and type "Remote-SSH: Connect to Host..." and select the VM you want to connect to (eg dianbo-dl.us-central1-abroad-ml4cvd)

Step 5: open the folder you want to work on in the VM, type in your Broad password, and you are good to go!


## Contributing code

Want to contribute code to this project? Please see [CONTRIBUTING](./CONTRIBUTING.md) for developer setup and other details.

## Releases
Ideally, each release should be available on our [github releases page](https://github.com/broadinstitute/ml4h/releases)
In addition, the version # in setup.py should be incremented. 
The pip installable [ml4h package on pypi should also be updated](https://pypi.org/project/ml4h/). 

If the release changed the docker image, the new dockers both (CPU & GPU) should update the “latest” tag and should be pushed to both gcr: `gcr.io/broad-ml4cvd/deeplearning`, and the [ml4h github container repo](https://github.com/broadinstitute/ml4h/pkgs/container/ml4h) with appropriate tags (e.g. `tf2.9-latest-gpu` for the latest GPU docker image or `tf2.9-latest-cpu` for the CPU) at: ` ghcr.io/broadinstitute/ml4h`


## Command line interface
The ml4h package is designed to be accessable through the command line using "recipes".
To get started, please see [RECIPE_EXAMPLES](./RECIPE_EXAMPLES.md).


[![DOI](https://zenodo.org/badge/180627543.svg)](https://zenodo.org/badge/latestdoi/180627543)
