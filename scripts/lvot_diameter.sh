#!/bin/bash
gcp_zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
VMTAG=$1
STEP=$2

cd /home/pdiachil/ml
git checkout pd_atria
git pull

cnt1=$((VMTAG*STEP))
cnt2=$((VMTAG*STEP+STEP))

for i in $(seq $cnt1 10 $cnt2)
do
start=$i
end=$((i+10))
/home/pdiachil/ml/scripts/tf.sh -c -r /home/pdiachil/ml/notebooks/mri/lvot_diameter.py $start $end

/snap/bin/gsutil cp /home/pdiachil/projects/chambers/*.gif gs://ml4cvd/pdiachil/lvot_diameter/1cm_height_gif/
/snap/bin/gsutil cp /home/pdiachil/projects/chambers/*.csv gs://ml4cvd/pdiachil/lvot_diameter/1cm_height_csv/
done

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}
