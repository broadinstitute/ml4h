#!/usr/bin/bash

# apt update; apt install git -y
# git clone --recursive git://github.com/mmolero/pypoisson.git
# cd pypoisson
# python setup.py build
# python setup.py install

mkdir -p /home/pdiachil/atria_poisson_output
cd /home/pdiachil/atria_poisson_output
cnt=0
for i in {0..5000}
do
    python /home/pdiachil/ml/scripts/mri_la_poisson.py $i 
done    


