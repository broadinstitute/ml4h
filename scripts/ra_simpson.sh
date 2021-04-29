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
    if ! $(/snap/bin/gsutil -q stat gs://ml4cvd/pdiachil/surface_reconstruction/sax_2ch_3ch_4ch/fastai_ra_simpson_ortho_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122/RA_simpson_fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122_${start}_${end}.csv)
    then
        /home/pdiachil/ml/scripts/tf.sh -c /home/pdiachil/ml/notebooks/mri/ra_simpson.py $start $end
        cd /home/pdiachil/ml/notebooks/mri
        /snap/bin/gsutil cp *simpson*.csv gs://ml4cvd/pdiachil/surface_reconstruction/sax_2ch_3ch_4ch/fastai_ra_simpson_ortho_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122/
    fi
done

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}