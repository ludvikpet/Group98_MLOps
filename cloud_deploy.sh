#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Environment Variables
export PROJECT_ID="cleaninbox-448011"
export LOCATION="europe-west1"
#export SERVICE_NAME=load-and-transform-team-stats-to-bq-service  # Uncomment when needed
export REPO_NAME="email-frontend"
export IMAGE_NAME="backend"
export IMAGE_TAG="latest"

# Tag the Docker image
docker tag \
    $IMAGE_NAME:$IMAGE_TAG \
    $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$IMAGE_TAG

# Push the Docker image to Artifact Registry
docker push \
    $LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$IMAGE_TAG

# Deploy the image to Cloud Run
gcloud run deploy backend \
    --image=$LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$IMAGE_TAG \
    --region=$LOCATION \
    --platform=managed


echo gcloud run services describe backend --region=$LOCATION --format="value(status.url)")




