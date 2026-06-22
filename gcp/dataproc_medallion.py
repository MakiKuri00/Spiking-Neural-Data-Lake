"""
PySpark port of lakehouse/medallion.py for Dataproc Serverless (the cloud scale-out of
the local Medallion PoC). Same Bronze -> Silver -> Gold path, distributed over GCS.

  reads  gs://<bucket>/bronze/   (Parquet: columns t, channel)
  writes gs://<bucket>/silver/   (per-channel counts per time bin)
         gs://<bucket>/gold/     (per-channel firing rate + population synchrony)

Run via gcp/submit_dataproc.sh (no cluster to manage). ICR + latency handoff from the
local PoC are left as Gold-stage UDFs to add when needed.
"""
import sys
from pyspark.sql import SparkSession, functions as F

WINDOW = 100


def main(bucket):
    spark = SparkSession.builder.appName("snn-medallion").getOrCreate()
    base = f"gs://{bucket}"

    bronze = spark.read.parquet(f"{base}/bronze/")          # t, channel

    silver = (bronze
              .withColumn("bin", (F.col("t") / WINDOW).cast("long"))
              .groupBy("channel", "bin").count()
              .withColumnRenamed("count", "spikes"))
    silver.write.mode("overwrite").parquet(f"{base}/silver/")

    duration = bronze.agg(F.max("t")).collect()[0][0] + 1
    rate = (bronze.groupBy("channel").count()
            .withColumn("rate", F.col("count") / F.lit(float(duration))))

    pop = silver.groupBy("bin").agg(F.sum("spikes").alias("pop"))
    st = pop.agg(F.mean("pop").alias("m"), F.stddev("pop").alias("s")).collect()[0]
    synchrony = float((st["s"] or 0.0) / st["m"]) if st["m"] else 0.0

    gold = rate.withColumn("synchrony_cv", F.lit(synchrony))
    gold.write.mode("overwrite").parquet(f"{base}/gold/")

    print(f"medallion done: duration={duration}, synchrony_cv={synchrony:.3f}")
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: dataproc_medallion.py <bucket>")
    main(sys.argv[1])
