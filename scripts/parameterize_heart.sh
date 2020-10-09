#!/bin/bash
gcp_zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
VMTAG=$1
STEP=$2

cd /home/pdiachil/ml
sudo mount -o norecovery,discard,defaults /dev/sdb /mnt/disks/segmented-sax-lax-v20200901/

cnt1=$((VMTAG*STEP))
cnt2=$((VMTAG*STEP+STEP))

/home/pdiachil/ml/scripts/tf.sh -c /home/pdiachil/ml/notebooks/mri/parameterize_heart_geom.py $cnt1 $cnt2

cd /home/pdiachil/ml/notebooks/mri
/snap/bin/gsutil cp *processed* gs://ml4cvd/pdiachil/leftheart_boundary_v20201006/

cd /home/pdiachil/projects/chambers
/snap/bin/gsutil cp poisson* gs://ml4cvd/pdiachil/leftheart_boundary_xdmf_v20201006

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}