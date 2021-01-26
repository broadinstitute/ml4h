for j in {0..4}
do
    start=$((j*10))
    end=$((start+10-1))
    for k in {0..2}
    do
    for i in $(seq $start $end)
    do
	gcloud compute instances create pdiachil-rv-inference-${i} \
	       --machine-type=n1-standard-8 \
	       --boot-disk-size=150 \
	       --image=rv-parameterization \
	       --maintenance-policy=TERMINATE \
	       --accelerator=type=nvidia-tesla-p4,count=1 &
	sleep 1
    done
    wait
    done
    wait
    for i in $(seq $start $end)
    do
	gcloud compute instances attach-disk pdiachil-rv-inference-${i} --disk=annotated-cardiac-tensors-45k --mode ro &
	sleep 1
    done
    wait
    sleep 25
    for i in $(seq $start $end)
    do
	gcloud compute ssh pdiachil-rv-inference-${i} --command="cd /home/pdiachil;cd ml;git pull;git checkout pd_sf_blox;git pull;nohup bash /home/pdiachil/ml/scripts/infer_to_hd5.sh $i > /home/pdiachil/out_${i}.out 2> /home/pdiachil/out_${i}.err < /dev/null &" &
	sleep 1
    done
done
