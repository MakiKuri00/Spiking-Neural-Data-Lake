#!/usr/bin/env bash
# Build the training image (Cloud Build), push to Artifact Registry, run a Vertex AI
# custom GPU job: eth_mnist_bindsnet.py --gpu at 6400/60k (the ~95% config) on an L4.
#   PROJECT=my-proj bash gcp/submit_vertex.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"
REGION="${REGION:-us-central1}"
IMG="$REGION-docker.pkg.dev/$PROJECT/snn-images/snn-train:latest"

# build + push the image (uses gcp/Dockerfile, repo root as build context)
gcloud builds submit --project="$PROJECT" \
  --config=gcp/cloudbuild.yaml --substitutions=_IMG="$IMG" .

# launch the GPU training job
gcloud ai custom-jobs create \
  --project="$PROJECT" --region="$REGION" \
  --display-name="snn-6400-train" \
  --worker-pool-spec="machine-type=g2-standard-8,accelerator-type=NVIDIA_L4,accelerator-count=1,replica-count=1,container-image-uri=$IMG"

echo "submitted. stream logs with:"
echo "  gcloud ai custom-jobs stream-logs <JOB_ID> --region=$REGION"
