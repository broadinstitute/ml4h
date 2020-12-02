#!/bin/bash
gcp_zone=$(curl -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone -s | cut -d/ -f4)
VMTAG=$1
STEP=$2

cd /home/pdiachil/ml
git checkout pd_atria
git pull

sudo mkdir -p /mnt/disks/segmented-sax-v20201124-lax-v20201122-petersen
sudo mount -o norecovery,discard,defaults /dev/sdb /mnt/disks/segmented-sax-v20201124-lax-v20201122-petersen/

cnt1=$((VMTAG*STEP))
cnt2=$((VMTAG*STEP+STEP-1))

for i in $(seq $cnt1 $cnt2)
do
    end=$((i+1))
    /home/pdiachil/ml/scripts/tf.sh -c /home/pdiachil/ml/notebooks/mri/parameterize_rv_geom.py $i $end
done

cd /home/pdiachil/ml/notebooks/mri
/snap/bin/gsutil cp *processed* gs://ml4cvd/pdiachil/surface_reconstruction/sax_4ch/fastai_sax_v20201124_lax_v20201122/csv-separation/
# /snap/bin/gsutil cp *hd5 gs://ml4cvd/pdiachil/rightheart_boundary_images_v20201102/
# /snap/bin/gsutil cp *xmf gs://ml4cvd/pdiachil/rightheart_boundary_images_v20201102/

cd /home/pdiachil/projects/chambers
/snap/bin/gsutil cp poisson* gs://ml4cvd/pdiachil/surface_reconstruction/sax_4ch/fastai_sax_v20201124_lax_v20201122/xdmf-separation/
/snap/bin/gsutil cp /home/pdiachil/out* gs://ml4cvd/pdiachil/surface_reconstruction/sax_4ch/fastai_sax_v20201124_lax_v20201122/logs-separation/

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}