#!/bin/bash
gcp_zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
VMTAG=$1
STEP=$2

cd /home/pdiachil/ml
git checkout pd_atria
git pull

# sudo mkdir -p /mnt/disks/segmented-ml4h-v20201203-v20201122-petersen
# sudo mount -o norecovery,discard,defaults /dev/sdb /mnt/disks/segmented-ml4h-v20201203-v20201122-petersen/

cnt1=$((VMTAG*STEP))
cnt2=$((VMTAG*STEP+STEP))

# cnt1=$1
# cnt2=$2

/home/pdiachil/ml/scripts/tf.sh -c /home/pdiachil/ml/notebooks/mri/aorta_projections.py $cnt1 $cnt2
cd /home/pdiachil/ml/notebooks/mri
/snap/bin/gsutil cp *.npy gs://ml4cvd/pdiachil/aorta_for_marcus/aorta_numpy_46k/
/snap/bin/gsutil cp -r bodymri* gs://ml4cvd/pdiachil/aorta_for_marcus/aorta_vti_46k/
#yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}