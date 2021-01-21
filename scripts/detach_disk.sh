#!/usr/bin/env bash
DISK=${1:-data}
shift 1
VMS=$(gcloud compute instances list | awk '{print $1}')
ZONE=us-central1-a
for VM in $VMS;
  do gcloud compute instances detach-disk $VM --zone $ZONE --disk=$DISK ;
done