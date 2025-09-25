#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-yourrepo/plysvc:latest}"

echo "Building $IMAGE"
docker build -t "$IMAGE" .

echo "Pushing $IMAGE"
docker push "$IMAGE"

echo "Done. Set your Runpod Serverless endpoint image to: $IMAGE"
