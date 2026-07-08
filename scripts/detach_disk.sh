#!/usr/bin/env bash
DISK=${1:-data}
shift 1
PROJECT=${1:-broad-ml4cvd}
shift 1

VMS=$(gcloud compute instances list --project ${PROJECT} | awk '{print $1}')
ZONE=us-central1-a
for VM in $VMS;
  do gcloud compute instances detach-disk $VM --project ${PROJECT} --zone $ZONE --disk=$DISK;
done