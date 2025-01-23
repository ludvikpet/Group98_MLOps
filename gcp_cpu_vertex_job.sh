#!/bin/bash
# Load environment variables from .env
#effort for loading wandb from env -> seems impossible
#set -a
#source .env
#set +a

gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name=fullfit-1-epoch \
    --config=gcloud_vertex/config_cpu.yaml \
    #--args WANDB_API_KEY=$WANDB_API_KEY,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT
    # these are the arguments that are passed to the container, only needed if you want to change defaults
    #--command 'python src/my_project/train.py' \
    #--args '["--epochs", "10"]'
