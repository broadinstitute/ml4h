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

mdkir -p /home/pdiachil/projects/la_biplane

for i in $(seq $cnt1 10 $cnt2)
do
    end=$((i+10))
    /snap/bin/gsutil -q stat gs://ml4cvd/pdiachil/surface_reconstruction/2ch_4ch/fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122/csv/LA_processed_fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122_${i}_${end}.csv
    if [[ $? == 1 ]]
    then
        /home/pdiachil/ml/scripts/tf.sh -c "\
        pip install Cython;
        cd /home/pdiachil/src/pypoisson;
        python setup.py build;
        python setup.py install;
        cd /home/pdiachil/ml; 
        export PYTHONPATH=/home/pdiachil/ml:/usr/lib/python3.8/site-packages/pypoisson-0.10-py3.8-linux-x86_64.egg;
        pip install scikit-learn; 
        pip install numcodecs;
        pip install imageio;
        pip install opencv-python-headless;
        pip install open3d;
        python3 /home/pdiachil/ml/notebooks/mri/parameterize_la_geom.py $i $end"
    fi 
done

cd /home/pdiachil/ml/
/snap/bin/gsutil cp *processed_biplane* gs://ml4cvd/pdiachil/surface_reconstruction/2ch_4ch/fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122/csv/
#/snap/bin/gsutil cp *hd5 gs://ml4cvd/pdiachil/rightheart_boundary_images_v20201102/
#/snap/bin/gsutil cp *xmf gs://ml4cvd/pdiachil/rightheart_boundary_images_v20201102/

cd /home/pdiachil/projects/la_biplane
/snap/bin/gsutil cp poisson* gs://ml4cvd/pdiachil/surface_reconstruction/2ch_4ch/fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122/xdmf/
/snap/bin/gsutil cp /home/pdiachil/out* gs://ml4cvd/pdiachil/surface_reconstruction/2ch_4ch/fastai_sax-v20201202-2ch-v20200809-3ch-v20200603-4ch-v20201122/log/

yes | /snap/bin/gcloud compute instances delete $(hostname) --zone ${gcp_zone}