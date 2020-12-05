#!/bin/bash
gcp_zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
VMTAG=$1
STEP=$2

cd /home/pdiachil
/snap/bin/gsutil cp gs://ml4cvd/models/sax_slices_jamesp_4b_hyperopted_dropout_pap_dupe/sax_slices_jamesp_4b_hyperopted_dropout_pap_dupe.h5 ./

cd /home/pdiachil/ml

sudo mkdir -p /mnt/disks/annotated-cardiac-tensors-44k
sudo mount -o norecovery,discard,defaults /dev/sdb /mnt/disks/annotated-cardiac-tensors-44k/

cnt1=$((VMTAG*STEP))
cnt2=$((VMTAG*STEP+STEP-1))

/home/pdiachil/ml/scripts/tf.sh /home/pdiachil/ml/notebooks/mri/infer_to_hd5.py 49 $VMTAG 


cd /home/pdiachil/
/snap/bin/gsutil cp *.h5 gs://ml4cvd/pdiachil/surface_reconstruction/sax_4ch/ml4h_v20201203_v20201122/hd5/
/snap/bin/gsutil cp /home/pdiachil/out* gs://ml4cvd/pdiachil/surface_reconstruction/sax_4ch/ml4h_v20201203_v20201122/log-inference/

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}