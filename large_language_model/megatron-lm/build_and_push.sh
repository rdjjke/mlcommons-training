#!/bin/bash

set -e

REGISTRY="cr.ai.nebius.cloud/crnbu823dealq64cp1s6"
REPOSITORY="megatron"
TAG="mlcommons00f04c52-pytorch23.10-2"

echo "Build image"
docker build -t $REPOSITORY:$TAG --platform linux/amd64 .

echo "Tag image"
docker tag $REPOSITORY:$TAG $REGISTRY/$REPOSITORY:$TAG

echo "Push image"
docker push $REGISTRY/$REPOSITORY:$TAG

