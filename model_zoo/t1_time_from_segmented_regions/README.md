# T1 time from segmented regions

Test on an x86_64 Linux machine

"The Docker image is x86_64; on Apple Silicon Macs, TensorFlow's AVX requirement prevents it from running under Docker Desktop's emulation — use an x86_64 machine, or Colima/QEMU." 

## Get public UK Biobank example experimental shMOLLI sequence image in DICOM format
cd ~
mkdir demo
cd demo
curl -sL -o 20214_2_0.zip "https://biobank.ndph.ox.ac.uk/ukb/ukb/examples/20214_2_0.zip"
mkdir zip_folder
mkdir dicoms
mv 20214_2_0.zip zip_folder/1000001_20214_2_0.zip

# Get ml4h repo and the correct branch
cd ~
git clone https://github.com/broadinstitute/ml4h.git
git checkout --track origin/dfp_pap_gt_update_before_merge

## Tensorize
## Includes Docker image pull so takes 5-15 minutes
cd ~/ml4h
./scripts/tensorize.sh -t $HOME/demo/inputs/ -n 1 -a " --zip_folder $HOME/demo/zip_folder/ --mri_field_ids 20214 --xml_field_ids --dicoms $HOME/demo/dicoms/ "

## 

