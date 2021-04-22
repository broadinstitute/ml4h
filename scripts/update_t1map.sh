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
cnt2=$((VMTAG*STEP+STEP-1))

# cnt1=$1
# cnt2=$2

for i in $(seq $cnt1 10 $cnt2)
do
    start=$i
    end=$((i+10))
    cd /home/pdiachil/ml
    /home/pdiachil/ml/scripts/tf.sh -c /home/pdiachil/ml/notebooks/mri/update_t1map_images.py $start $end    
    cd /home/pdiachil/ml/notebooks/mri
    /snap/bin/gsutil cp *.hd5 gs://ml4cvd/pdiachil/segmented-sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122-t1map/
    rm -f *.hd5
    /snap/bin/gsutil cp *.png gs://ml4cvd/pdiachil/t1map-pngs/inference/
    rm -f *.png
    rm -f *.dcm
done

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}