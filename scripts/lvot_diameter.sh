#!/bin/bash
gcp_zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
VMTAG=$1
STEP=$2

cd /home/pdiachil/ml
git checkout pd_atria
git pull

cnt1=$((VMTAG*STEP))
cnt2=$((VMTAG*STEP+STEP))

/home/pdiachil/ml/scripts/tf.sh -c /home/pdiachil/ml/notebooks/mri/lvot_diameter.py $cnt1 $cnt2

/snap/bin/gsutil cp /home/pdiachil/projects/chambers/*.gif gs://ml4cvd/pdiachil/lvot_diameter/6points_gif/
/snap/bin/gsutil cp /home/pdiachil/projects/chambers/*.csv gs://ml4cvd/pdiachil/lvot_diameter/6points_csv/

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}
