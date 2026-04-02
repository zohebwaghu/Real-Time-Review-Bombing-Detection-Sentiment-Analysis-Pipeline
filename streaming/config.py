"""
Configuration for the Spark Structured Streaming pipeline.

Centralizes all thresholds, window sizes, S3 paths, and Kafka settings
so they can be tuned without touching the pipeline code.
"""

import os

# ── Kafka ──────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "yelp-reviews")
KAFKA_STARTING_OFFSETS  = "latest"

# ── Sliding Window ──────────────────────────────────────────────
WINDOW_DURATION  = "10 minutes"   # Size of each detection window
WINDOW_SLIDE     = "1 minute"     # How often a new window starts
WATERMARK_DELAY  = "15 minutes"   # Tolerate events arriving this late

# ── Anomaly Detection Thresholds ───────────────────────────────
# Rule 1 + Rule 2
REVIEW_COUNT_THRESHOLD = 8     # Window must contain > N reviews to trigger
AVG_STARS_THRESHOLD    = 2.5   # Average stars below this → suspicious (Rule 1)
STD_THRESHOLD          = 0.8   # Star std-dev below this → coordinated (Rule 2)

# Rule 2 guard — only flag if avg stars is also suspiciously low
# Prevents flagging high-quality restaurants with consistently 4-5 star reviews
COORDINATED_AVG_THRESHOLD = 3.5

# Rule 3 — negative ratio spike
NEG_RATIO_THRESHOLD   = 0.80   # >80 % of window reviews are 1–2 stars
NEG_COUNT_MIN_REVIEWS = 5      # Minimum reviews in window for Rule 3 to fire

# ── Sentiment Model ─────────────────────────────────────────────
SENTIMENT_MODEL_PATH = os.getenv(
    "SENTIMENT_MODEL_PATH", "/models/sentiment_pipeline"
)

# ── S3 Output Paths ─────────────────────────────────────────────
S3_BUCKET               = os.getenv("S3_BUCKET", "review-bombing-group09")
S3_ORGANIC_PATH         = f"s3a://{S3_BUCKET}/organic-reviews/"
S3_QUARANTINED_PATH     = f"s3a://{S3_BUCKET}/quarantined-reviews/"
S3_CHECKPOINT_ORGANIC   = f"s3a://{S3_BUCKET}/checkpoints/organic/"
S3_CHECKPOINT_QUARANTINED = f"s3a://{S3_BUCKET}/checkpoints/quarantined/"

# ── Processing Triggers ─────────────────────────────────────────
TRIGGER_INTERVAL       = "30 seconds"
DEBUG_TRIGGER_INTERVAL = "10 seconds"

# ── TF-IDF (must match training) ────────────────────────────────
NUM_FEATURES = 20000
