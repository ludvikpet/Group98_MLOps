# Environment Variables
export PROJECT_ID="cleaninbox-448011"
export LOCATION="europe-west1"
#export SERVICE_NAME=load-and-transform-team-stats-to-bq-service  # Uncomment when needed
export REPO_NAME="email-api"
export IMAGE_NAME="frontend_local"
export IMAGE_TAG="latest"

# Build docker image 
docker build -t $IMAGE_NAME:latest -f dockerfiles/$IMAGE_NAME.dockerfile .
docker run -p 8080:8080 $IMAGE_NAME

