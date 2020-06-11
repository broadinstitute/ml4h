#!/usr/bin/bash

apt update; apt install git -y
git clone --recursive git://github.com/mmolero/pypoisson.git
cd pypoisson
python setup.py build
python setup.py install

mkdir -p /home/pdiachil/atria_poisson_output
cd /home/pdiachil/atria_poisson_output
cnt=0
for i in {1..1200}
do
    python /home/pdiachil/ml/scripts/mri_la_poisson.py $cnt &
    python /home/pdiachil/ml/scripts/mri_la_poisson.py $(($cnt+1)) &
    python /home/pdiachil/ml/scripts/mri_la_poisson.py $(($cnt+2)) &
    python /home/pdiachil/ml/scripts/mri_la_poisson.py $(($cnt+3)) &
    wait
    cnt=$((cnt+4))
    break
done    


