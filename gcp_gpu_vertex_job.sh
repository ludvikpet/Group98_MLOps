#!/bin/bash
gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name=test-a-train-run \
    --config=gcloud_vertex/config_gpu.yaml \
    # these are the arguments that are passed to the container, only needed if you want to change defaults
    --command 'python src/my_project/train.py' \
    --args '["--epochs", "10"]'
