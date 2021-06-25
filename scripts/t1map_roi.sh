#!/bin/bash
gcp_zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
VMTAG=$1
STEP=$2

cd /home/pdiachil/ml
git checkout pd_atria
git pull

sudo mkdir -p /mnt/disks/pdiachil-t1map
sudo mount -o norecovery,discard,defaults /dev/sdb /mnt/disks/pdiachil-t1map/

cnt1=$((VMTAG*STEP))
cnt2=$((VMTAG*STEP+STEP-1))

# cnt1=$1
# cnt2=$2

for i in $(seq $cnt1 10 $cnt2)
do
    start=$i
    end=$((i+10))
    cd /home/pdiachil/ml
    /home/pdiachil/ml/scripts/tf.sh -c /home/pdiachil/ml/notebooks/mri/t1map_inference.py $start $end    
    cd /home/pdiachil/projects/t1map/inference
    /snap/bin/gsutil cp *.csv gs://ml4cvd/pdiachil/t1map-pngs/inference_rois_iqr_3px/
    /snap/bin/gsutil cp *.png gs://ml4cvd/pdiachil/t1map-pngs/inference_rois_iqr_3px/
    cd /home/pdiachil/ml/notebooks/mri
    rm -f *.hd5
    # /snap/bin/gsutil cp *.png gs://ml4cvd/pdiachil/t1map-pngs/inference/
    rm -f *.png
    rm -f *.dcm
done

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}