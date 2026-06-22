variable "project_id" {
  type        = string
  description = "GCP project id (e.g. snn-data-lake-prod)"
}

variable "region" {
  type    = string
  default = "us-central1"
}
