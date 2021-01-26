target_disk = annotated-cardiac-tensors-45k
target_vm_snapshot = rv-parameterization
vm_base_name = pdiachil-rv-inference

for j in {0..4}
do
    start=$((j*10))
    end=$((start+10-1))
    for k in {0..2}
    do
    for i in $(seq $start $end)
    do
    # Spawn a VM instance given prepared VM snapshot.
	gcloud compute instances create $vm_base_name-${i} \
	       --machine-type=n1-standard-8 \
	       --boot-disk-size=150 \
	       --image=$target_vm_snapshot \
	       --maintenance-policy=TERMINATE \
	       --accelerator=type=nvidia-tesla-p4,count=1 &
	sleep 1  # Sleep for a second
    done # End loop i
    wait # Wait until completion
    done # End loop k
    wait # Wait until completion
    for i in $(seq $start $end)
    do
    # Attach disk(s) to the spawned VM instance.
	gcloud compute instances attach-disk $vm_base_name-${i} --disk=$target_disk --mode ro &
	sleep 1 # Sleep for a second
    done # End loop i
    wait # Wait until completeion
    sleep 25 # Sleep for 25 seconds 
    for i in $(seq $start $end)
    do
	gcloud compute ssh $vm_base_name-${i} --command="cd /home/pdiachil;cd ml;git pull;git checkout pd_sf_blox;git pull;nohup bash /home/pdiachil/ml/scripts/infer_to_hd5.sh $i > /home/pdiachil/out_${i}.out 2> /home/pdiachil/out_${i}.err < /dev/null &" &
	sleep 1 # Sleep for a second
    done # End loop i
done # End outer VM loop
