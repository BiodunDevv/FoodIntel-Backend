#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
ECR_REPOSITORY="${ECR_REPOSITORY:-foodintel-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

if [[ -z "$AWS_REGION" || -z "$AWS_ACCOUNT_ID" ]]; then
  echo "Set AWS_REGION and AWS_ACCOUNT_ID before running this script." >&2
  echo "Example: AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012 ./scripts/build_push_ecr.sh" >&2
  exit 1
fi

IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

aws ecr describe-repositories \
  --repository-names "$ECR_REPOSITORY" \
  --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository \
    --repository-name "$ECR_REPOSITORY" \
    --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build --platform linux/amd64 -t "$ECR_REPOSITORY:$IMAGE_TAG" .
docker tag "$ECR_REPOSITORY:$IMAGE_TAG" "$IMAGE_URI"
docker push "$IMAGE_URI"

echo "$IMAGE_URI"
