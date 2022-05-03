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
/home/pdiachil/ml/scripts/tf.sh -c \
"pip install imageio; python /home/pdiachil/ml/notebooks/mri/aorta_get_iliacs.py $cnt1 $cnt2"

cd /home/pdiachil/projects/aorta
/snap/bin/gsutil cp *.gif gs://ml4cvd/pdiachil/aorta_for_marcus/aorta_gifs_46k/
/snap/bin/gsutil cp gifs_ratio*.csv gs://ml4cvd/pdiachil/aorta_for_marcus/aorta_gifs_46k/
yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}