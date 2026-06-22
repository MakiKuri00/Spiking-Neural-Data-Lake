#!/usr/bin/env bash
# Submit the Medallion ETL to Dataproc Serverless (managed Spark — no cluster to run).
#   PROJECT=my-proj BUCKET=my-proj-snn-lake bash gcp/submit_dataproc.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"
: "${BUCKET:?set BUCKET (the GCS lake bucket)}"
REGION="${REGION:-us-central1}"

gcloud dataproc batches submit pyspark gcp/dataproc_medallion.py \
  --project="$PROJECT" \
  --region="$REGION" \
  --deps-bucket="gs://$BUCKET" \
  --version=2.2 \
  -- "$BUCKET"
