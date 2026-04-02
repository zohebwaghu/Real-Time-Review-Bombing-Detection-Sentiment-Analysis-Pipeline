"""
Spark Structured Streaming Pipeline — Real-Time Review Bombing Detection.

Dual-stream architecture:
  Stream A: Sentiment classification per review (TF-IDF + LR PipelineModel)
  Stream B: Sliding-window anomaly detection per business (frequency + rating patterns)

Both streams are derived from the same Kafka source.  They are joined via a
stream-stream left outer join with a time-range condition, which is the correct
way to bound state in Spark Structured Streaming and avoid unbounded state growth.

Three anomaly rules (any one triggers quarantine):
  1. HIGH_VELOCITY_NEGATIVE  — high review volume + low average stars
  2. COORDINATED_UNIFORM     — high review volume + suspiciously low star variance
  3. NEGATIVE_RATIO_SPIKE    — >80 % of reviews in window are 1- or 2-star

Outputs:
  - s3a://<bucket>/organic-reviews/     (Parquet, partitioned by business_id)
  - s3a://<bucket>/quarantined-reviews/ (Parquet, partitioned by business_id)
  - Console sink for quarantined rows (debug)

Usage (inside spark-master container):
    spark-submit \\
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,\\
                 org.apache.hadoop:hadoop-aws:3.3.4 \\
      /app/streaming_pipeline.py
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, udf, window, count, avg, stddev, lit, when,
    current_timestamp, expr,
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, DoubleType, BooleanType, TimestampType,
)
from pyspark.ml import PipelineModel

from config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, KAFKA_STARTING_OFFSETS,
    WINDOW_DURATION, WINDOW_SLIDE, WATERMARK_DELAY,
    REVIEW_COUNT_THRESHOLD, AVG_STARS_THRESHOLD, STD_THRESHOLD,
    COORDINATED_AVG_THRESHOLD,
    NEG_RATIO_THRESHOLD, NEG_COUNT_MIN_REVIEWS,
    SENTIMENT_MODEL_PATH,
    S3_ORGANIC_PATH, S3_QUARANTINED_PATH,
    S3_CHECKPOINT_ORGANIC, S3_CHECKPOINT_QUARANTINED,
    TRIGGER_INTERVAL, DEBUG_TRIGGER_INTERVAL,
)


# ══════════════════════════════════════════════════════════════
# SPARK SESSION
# ══════════════════════════════════════════════════════════════

spark = SparkSession.builder \
    .appName("ReviewBombingDetection") \
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
        "org.apache.hadoop:hadoop-aws:3.3.4",
    ) \
    .config("spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.amazonaws.auth.EnvironmentVariableCredentialsProvider",
    ) \
    .config(
        "spark.sql.streaming.stateStore.providerClass",
        "org.apache.spark.sql.execution.streaming.state."
        "HDFSBackedStateStoreProvider",
    ) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ══════════════════════════════════════════════════════════════
# SCHEMA  — is_injected is BooleanType (nullable; absent on organic rows)
# ══════════════════════════════════════════════════════════════

review_schema = StructType([
    StructField("review_id",   StringType(),  nullable=True),
    StructField("user_id",     StringType(),  nullable=True),
    StructField("business_id", StringType(),  nullable=True),
    StructField("stars",       IntegerType(), nullable=True),
    StructField("text",        StringType(),  nullable=True),
    StructField("date",        StringType(),  nullable=True),
    StructField("useful",      IntegerType(), nullable=True),
    StructField("funny",       IntegerType(), nullable=True),
    StructField("cool",        IntegerType(), nullable=True),
    StructField("is_injected", BooleanType(), nullable=True),
])


# ══════════════════════════════════════════════════════════════
# STEP 1: READ FROM KAFKA + PARSE
# ══════════════════════════════════════════════════════════════

raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", KAFKA_STARTING_OFFSETS) \
    .option("failOnDataLoss", "false") \
    .load()

reviews = (
    raw_stream
    .select(from_json(col("value").cast("string"), review_schema).alias("d"))
    .select("d.*")
    .withColumn("event_time", col("date").cast(TimestampType()))
    .withColumn("ingestion_time", current_timestamp())
    .filter(col("event_time").isNotNull())        # drop records without a parseable date
    .filter(col("business_id").isNotNull())
    .withWatermark("event_time", WATERMARK_DELAY)
)


# ══════════════════════════════════════════════════════════════
# STEP 2: STREAM A — SENTIMENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════

sentiment_model = PipelineModel.load(SENTIMENT_MODEL_PATH)

# Transform returns all original columns + prediction + probability
sentiment_df = (
    sentiment_model.transform(reviews)
    .withColumnRenamed("prediction", "sentiment_label")
    .select(
        "review_id", "user_id", "business_id", "stars", "text",
        "event_time", "ingestion_time", "is_injected",
        "sentiment_label",          # 0.0 = negative, 1.0 = positive
    )
)


# ══════════════════════════════════════════════════════════════
# STEP 3: STREAM B — SLIDING WINDOW ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════

# Aggregate per business × sliding window
window_stats = (
    reviews
    .groupBy(
        col("business_id"),
        window("event_time", WINDOW_DURATION, WINDOW_SLIDE),
    )
    .agg(
        count("*").alias("review_count"),
        avg("stars").alias("avg_stars"),
        stddev("stars").alias("std_stars"),
        count(when(col("stars") <= 2, True)).alias("neg_count"),
    )
)

# Apply three anomaly rules — any match → quarantine
anomaly_df = (
    window_stats
    .withColumn("is_anomaly",
        # Rule 1: High volume burst with low average rating
        when(
            (col("review_count") > REVIEW_COUNT_THRESHOLD) &
            (col("avg_stars") < AVG_STARS_THRESHOLD),
            lit(True),
        )
        # Rule 2: High volume with suspiciously uniform AND low ratings
        # The avg_stars guard prevents flagging consistently positive restaurants
        .when(
            (col("review_count") > REVIEW_COUNT_THRESHOLD) &
            (col("std_stars") < STD_THRESHOLD) &
            (col("avg_stars") < lit(COORDINATED_AVG_THRESHOLD)),
            lit(True),
        )
        # Rule 3: >80 % of reviews in window are negative
        .when(
            (col("review_count") >= NEG_COUNT_MIN_REVIEWS) &
            (col("neg_count").cast(DoubleType()) / col("review_count") > NEG_RATIO_THRESHOLD),
            lit(True),
        )
        .otherwise(lit(False))
    )
    .withColumn("anomaly_reason",
        when(
            (col("review_count") > REVIEW_COUNT_THRESHOLD) &
            (col("avg_stars") < AVG_STARS_THRESHOLD),
            lit("HIGH_VELOCITY_NEGATIVE"),
        )
        .when(
            (col("review_count") > REVIEW_COUNT_THRESHOLD) &
            (col("std_stars") < STD_THRESHOLD) &
            (col("avg_stars") < lit(COORDINATED_AVG_THRESHOLD)),
            lit("COORDINATED_UNIFORM_RATING"),
        )
        .when(
            (col("review_count") >= NEG_COUNT_MIN_REVIEWS) &
            (col("neg_count").cast(DoubleType()) / col("review_count") > NEG_RATIO_THRESHOLD),
            lit("NEGATIVE_RATIO_SPIKE"),
        )
        .otherwise(lit("NONE"))
    )
)

# Only emit rows where an anomaly was detected.
# Add a timestamp column so Spark can apply a watermark to this side of the join.
anomalous_windows = (
    anomaly_df
    .filter(col("is_anomaly") == True)
    .select(
        col("business_id").alias("anom_business_id"),
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        col("review_count").alias("window_review_count"),
        col("avg_stars").alias("window_avg_stars"),
        col("std_stars").alias("window_std_stars"),
        col("anomaly_reason"),
    )
    # Use window_start as the event-time anchor for the right side of the join.
    # This watermark tells Spark it can safely expire old anomaly state.
    .withColumn("anomaly_event_time", col("window_start"))
    .withWatermark("anomaly_event_time", WATERMARK_DELAY)
)


# ══════════════════════════════════════════════════════════════
# STEP 4: STREAM-STREAM JOIN WITH TIME-RANGE CONDITION
#
# Spark requires a time-range condition on stream-stream left outer joins
# so it can bound the state it needs to retain for both sides.
#
# Condition: a review is labeled quarantined if its event_time falls
# within an anomaly detection window for the same business.
# ══════════════════════════════════════════════════════════════

labeled = (
    sentiment_df.join(
        anomalous_windows,
        (sentiment_df.business_id == anomalous_windows.anom_business_id) &
        (sentiment_df.event_time >= anomalous_windows.window_start) &
        (sentiment_df.event_time <= anomalous_windows.window_end +
         expr("INTERVAL 2 MINUTES")),   # small buffer for clock skew
        how="left",
    )
    .withColumn("label",
        when(col("anom_business_id").isNotNull(), lit("quarantined"))
        .otherwise(lit("organic"))
    )
)


# ══════════════════════════════════════════════════════════════
# STEP 5: WRITE OUTPUTS
# ══════════════════════════════════════════════════════════════

# 5a. Organic reviews → S3 (drop anomaly columns which are null here)
organic_sink = (
    labeled
    .filter(col("label") == "organic")
    .select(
        "review_id", "user_id", "business_id", "stars", "text",
        "sentiment_label", "event_time", "ingestion_time", "is_injected",
    )
    .writeStream
    .format("parquet")
    .option("path", S3_ORGANIC_PATH)
    .option("checkpointLocation", S3_CHECKPOINT_ORGANIC)
    .partitionBy("business_id")
    .outputMode("append")
    .trigger(processingTime=TRIGGER_INTERVAL)
    .start()
)

# 5b. Quarantined reviews → S3 (keep anomaly metadata)
quarantine_sink = (
    labeled
    .filter(col("label") == "quarantined")
    .select(
        "review_id", "user_id", "business_id", "stars", "text",
        "sentiment_label", "anomaly_reason",
        "window_review_count", "window_avg_stars", "window_std_stars",
        "window_start", "window_end",
        "event_time", "ingestion_time", "is_injected",
    )
    .writeStream
    .format("parquet")
    .option("path", S3_QUARANTINED_PATH)
    .option("checkpointLocation", S3_CHECKPOINT_QUARANTINED)
    .partitionBy("business_id")
    .outputMode("append")
    .trigger(processingTime=TRIGGER_INTERVAL)
    .start()
)

# 5c. Console debug — quarantined alerts only
debug_sink = (
    labeled
    .filter(col("label") == "quarantined")
    .select(
        "business_id", "stars", "sentiment_label",
        "anomaly_reason", "window_review_count", "window_avg_stars",
    )
    .writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", "false")
    .trigger(processingTime=DEBUG_TRIGGER_INTERVAL)
    .start()
)

print("Pipeline started — waiting for termination (Ctrl+C to stop).")
spark.streams.awaitAnyTermination()
