# Deploying on GCP (native)

Takes the local Medallion PoC (`lakehouse/medallion.py`) to a cloud lakehouse:
**GCS + BigLake/Iceberg + BigQuery + Dataproc Serverless + Vertex AI**. Pay-per-use;
near-zero at rest. You run these with your own auth/billing — nothing here provisions
your cloud for you.

## 0. Project + APIs
```bash
gcloud auth login
gcloud projects create snn-data-lake-prod --name="SNN Data Lake"
gcloud billing projects link snn-data-lake-prod --billing-account=XXXXXX-XXXXXX-XXXXXX
gcloud config set project snn-data-lake-prod
gcloud services enable storage.googleapis.com bigquery.googleapis.com \
  bigqueryconnection.googleapis.com dataproc.googleapis.com aiplatform.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com
```

## 1. Infrastructure (Terraform)
```bash
cd infra && terraform init && terraform apply -var project_id=snn-data-lake-prod
# outputs: bucket, dataset, connection, image_repo, service_acct
export PROJECT=snn-data-lake-prod
export BUCKET=$(terraform output -raw bucket); cd ..
```

## 2. Land data into Bronze
```bash
gcloud storage cp lakehouse/data/bronze.parquet gs://$BUCKET/bronze/
# production: stream via Pub/Sub + Dataflow instead (see Scale-out)
```

## 3. ETL — Medallion on Dataproc Serverless (managed Spark, no cluster)
```bash
PROJECT=$PROJECT BUCKET=$BUCKET bash gcp/submit_dataproc.sh
# writes gs://$BUCKET/silver/ and gs://$BUCKET/gold/
```

## 4. Query with BigQuery over BigLake
```sql
-- one-time: register the Gold Parquet as a BigLake table
CREATE OR REPLACE EXTERNAL TABLE `snn_lake.gold`
WITH CONNECTION `us-central1.biglake-snn`
OPTIONS (format = 'PARQUET', uris = ['gs://YOUR_BUCKET/gold/*']);
```
```bash
bq query --use_legacy_sql=false \
  'SELECT channel, rate, synchrony_cv FROM snn_lake.gold ORDER BY rate DESC LIMIT 5'
```

## 5. GPU training — Vertex AI custom job
```bash
PROJECT=$PROJECT bash gcp/submit_vertex.sh
# builds gcp/Dockerfile -> Artifact Registry, runs eth_mnist_bindsnet.py --gpu
# at NORD_M=6400 / 60k on an L4 -> ~95% (the run that was impractical on CPU)
```

## Scale-out (next, documented — not scripted here)
- **Streaming ingest**: Pub/Sub topic -> Dataflow (Beam) -> Bronze.
- **Orchestration**: Cloud Composer (Airflow) DAG ingest→Bronze→Silver→Gold.
- **Governance**: Dataplex catalog/zones, IAM, Cloud KMS (CMEK), Cloud DLP de-id (the FPE step).
- **Sharing**: Analytics Hub (Delta Sharing analog).

## Cost notes
GCS / BigQuery / Artifact Registry are ~free at rest. Dataproc Serverless bills per batch.
Vertex L4 ≈ $0.7–1/hr — the 6400/60k run is hours, so a few dollars. Don't leave jobs up.
```
