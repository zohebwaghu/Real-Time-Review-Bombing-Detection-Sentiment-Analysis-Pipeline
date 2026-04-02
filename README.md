# Real-Time Review Bombing Detection & Sentiment Analysis Pipeline

**DATA 228 — Big Data | Group 09**

A real-time streaming pipeline that detects coordinated fake-review attacks on Yelp businesses using Apache Kafka, Spark Structured Streaming, TF-IDF + Logistic Regression sentiment analysis, sliding-window anomaly detection, and SHAP explainability. Results are stored in AWS S3 (Parquet) and queried via Athena.

---

## Architecture

```
Yelp review.json ──┐
                    ├──► Kafka (yelp-reviews) ──► Spark Structured Streaming
Poison Injector ───┘                              │
                                                  ├─ Stream A: Sentiment (TF-IDF + LR)
                                                  ├─ Stream B: Anomaly (sliding window)
                                                  └─ Join ──► S3 organic / quarantined
                                                              │
                                                              ├─ SHAP XAI reports
                                                              └─ Athena evaluation
```

---

## Prerequisites

- Docker + Docker Compose
- Python 3.11+
- AWS account with S3 bucket and Athena enabled
- [Yelp Open Dataset](https://www.yelp.com/dataset) (`review.json`, `business.json`)

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials and S3 bucket name
```

### 2. Place Yelp data

```bash
mkdir -p data/
# Download from https://www.yelp.com/dataset and extract
cp /path/to/yelp_dataset/review.json  data/
cp /path/to/yelp_dataset/business.json data/
```

### 3. Start infrastructure

```bash
docker-compose up -d

# Wait ~20 seconds for Kafka to initialize, then verify
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

### 4. Create the Kafka topic

```bash
docker exec kafka kafka-topics --create \
  --topic yelp-reviews \
  --bootstrap-server localhost:9092 \
  --partitions 6 \
  --replication-factor 1
```

### 5. Train the sentiment model (one-time, ~10–15 min)

```bash
docker exec spark-master spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /models/train_sentiment_model.py \
  --input /data/review.json \
  --spark-output /models/sentiment_pipeline \
  --sklearn-output /models/sklearn_lr.pkl \
  --vectorizer-output /models/tfidf_vectorizer.pkl
```

Expected output:
```
  AUC:      0.9341  ✓
  Accuracy: 0.8912  ✓
  F1:       0.8889  ✓
```

### 6. Start the streaming pipeline

```bash
docker exec spark-master spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.apache.hadoop:hadoop-aws:3.3.4 \
  /app/streaming_pipeline.py
```

### 7. Start the organic producer

```bash
# Stream the first 50,000 reviews at 200/sec
docker exec producer python /app/producer.py \
  --file /data/review.json \
  --rate 200 \
  --max 50000
```

### 8. Launch an attack (in a separate terminal)

```bash
# Get a real business_id from the data
BIZ_ID=$(head -1 data/review.json | python3 -c "import json,sys; print(json.load(sys.stdin)['business_id'])")

# Blitz attack (100 reviews in 30s — easy to detect)
python poison_injector/poison_injector.py \
  --target "$BIZ_ID" \
  --scenario blitz \
  --bootstrap localhost:9092

# Sustained attack (200 reviews over 10 min — medium difficulty)
python poison_injector/poison_injector.py \
  --target "$BIZ_ID" \
  --scenario sustained \
  --bootstrap localhost:9092
```

### 9. Generate SHAP explanation reports

```bash
python xai/generate_xai_reports.py \
  --quarantine-dir s3://review-bombing-group09/quarantined-reviews/ \
  --output-dir ./xai_reports/ \
  --model models/sklearn_lr.pkl \
  --vectorizer models/tfidf_vectorizer.pkl
```

### 10. Run Athena evaluation queries

```bash
# Create tables
aws athena start-query-execution \
  --query-string file://evaluation/athena_queries.sql \
  --result-configuration OutputLocation=s3://review-bombing-group09/athena-results/
```

---

## Project Structure

```
review-bombing-detection/
├── docker-compose.yml          Infrastructure (Zookeeper, Kafka, Spark master+worker, Producer)
├── .env.example                AWS + Kafka config template
├── .gitignore
├── README.md
├── requirements.txt
│
├── streaming/
│   ├── streaming_pipeline.py   Core Spark Structured Streaming job
│   └── config.py               Thresholds, paths, window sizes
│
├── producer/
│   ├── producer.py             Kafka producer — streams Yelp JSON at target rate
│   ├── Dockerfile
│   └── requirements.txt
│
├── poison_injector/
│   ├── poison_injector.py      Attack simulator — 4 scenarios (blitz/sustained/stealth/multi)
│   └── requirements.txt
│
├── models/
│   └── train_sentiment_model.py  Offline training — Spark PipelineModel + sklearn mirror
│
├── xai/
│   ├── xai_module.py           SHAP LinearExplainer for flagged review batches
│   └── generate_xai_reports.py Batch XAI processing on quarantined Parquet data
│
├── evaluation/
│   ├── athena_queries.sql      Precision / Recall / F1 / Latency / Throughput queries
│   └── latency_profiler.py     Local profiling from Parquet files
│
├── tests/
│   ├── test_producer.py
│   ├── test_poison_injector.py
│   ├── test_sentiment_model.py
│   └── test_anomaly_rules.py
│
├── demo/
│   └── demo_script.md          5-minute demo walkthrough
│
├── data/                       Yelp dataset (gitignored — download separately)
├── models/                     Saved models (gitignored — created by training)
└── report/                     IEEE paper (LaTeX)
```

---

## Anomaly Detection Rules

Three rules fire independently; any one match quarantines the business window:

| Rule | Condition | Scenario detected |
|---|---|---|
| `HIGH_VELOCITY_NEGATIVE` | `review_count > 8` AND `avg_stars < 2.5` in 10-min window | Blitz / sustained attack |
| `COORDINATED_UNIFORM_RATING` | `review_count > 8` AND `std_stars < 0.8` | Bot farms with identical ratings |
| `NEGATIVE_RATIO_SPIKE` | `review_count ≥ 5` AND `neg_reviews / total > 80%` | Stealth attack with varied text |

All thresholds are configurable in `streaming/config.py`.

---

## Attack Scenarios

| Scenario | Reviews | Duration | Rate | Difficulty |
|---|---|---|---|---|
| `blitz` | 100 | 30 s | 3.3/sec | Easy |
| `sustained` | 200 | 600 s | 0.33/sec | Medium |
| `stealth` | 50 | 1800 s | 0.028/sec | Hard |
| `multi` | 30 × N targets | 120 s/target | 0.25/sec | Distributed |

---

## Evaluation Targets

| Metric | Target |
|---|---|
| Sentiment AUC | > 0.85 |
| Anomaly Precision | > 0.80 |
| Anomaly Recall | > 0.85 |
| Anomaly F1 | > 0.85 |
| End-to-end latency | < 60 s |
| Throughput | > 5,000 events/min |
| Blitz detection time | < 2 min |
| False positive rate | < 5% |

---

## Team Responsibilities

| Member | Components |
|---|---|
| Liza | `docker-compose.yml`, `producer.py`, Parquet sink, Athena DDL |
| Prakhar | `streaming_pipeline.py`, dual-stream join, checkpoint management |
| Sarvesh | `train_sentiment_model.py`, `poison_injector.py`, 4 attack scenarios |
| Shibin | Sliding window rules, threshold calibration, watermark tuning |
| Zoheb | `xai_module.py`, `athena_queries.sql`, F1 evaluation, IEEE paper, demo |

---

## Monitoring

- **Spark UI**: http://localhost:8080
- **Spark Job UI**: http://localhost:4040
- **Kafka topic**: `docker exec kafka kafka-console-consumer --topic yelp-reviews --bootstrap-server localhost:9092`
