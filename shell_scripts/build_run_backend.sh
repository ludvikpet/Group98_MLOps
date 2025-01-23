: ' Make sure to run this script from the root of the project directory '

# Environment Variables
export PROJECT_ID="cleaninbox-448011"
export LOCATION="europe-west1"
#export SERVICE_NAME=load-and-transform-team-stats-to-bq-service  # Uncomment when needed
export REPO_NAME="email-api"
export IMAGE_NAME="backend_local"
export IMAGE_TAG="latest"
export DOCKER_NAME="banking_api.dockerfile"

# Build docker image 
docker build -f dockerfiles/$DOCKER_NAME -t $IMAGE_NAME:$IMAGE_TAG .
docker run -p 8080:8080 $IMAGE_NAME
