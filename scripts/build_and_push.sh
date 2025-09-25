#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-yourrepo/plysvc:latest}"

echo "Building $IMAGE"
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  -t "$IMAGE" .

echo "Pushing $IMAGE"
docker push "$IMAGE"

echo "Done. Set your Runpod Serverless endpoint image to: $IMAGE"
