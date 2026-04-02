# 5-Minute Demo Script
## Real-Time Review Bombing Detection — Group 09

**Total runtime: 5 minutes | Format: Screen recording with voiceover**

---

## Pre-Demo Checklist (complete before recording)

- [ ] `docker-compose up -d` — all 5 containers healthy
- [ ] Sentiment model trained, files exist in `models/`
- [ ] Streaming pipeline running (see Spark UI at localhost:8080)
- [ ] Producer running in background (≥ 3 min of organic baseline)
- [ ] `BIZ_ID` variable set to a real business_id from the dataset
- [ ] Two terminal windows open side-by-side: pipeline output + attack terminal
- [ ] Browser tabs open: Spark UI (localhost:8080), Spark Job UI (localhost:4040)

---

## Minute 0:00–0:45 — Architecture Overview

> "We built a real-time streaming pipeline that detects review bombing attacks — coordinated fake negative reviews — on Yelp businesses. Here's how data flows."

**Show the architecture diagram** (from README or `report/figures/`):

- Point to Yelp dataset → Kafka → Spark → S3 → Athena
- "Both organic reviews and fake attack reviews land in the *same* Kafka topic. Our pipeline must tell them apart — in real time, with no ground-truth labels during inference."

---

## Minute 0:45–1:30 — Infrastructure

**Show terminal:**
```bash
docker-compose ps
```
> "Five containers: Zookeeper, Kafka, Spark master, Spark worker, and our Kafka producer. The producer streams 200 organic Yelp reviews per second."

**Show Spark UI at localhost:8080:**
> "One master, one worker with 4 cores and 4 GB RAM. This is where the streaming job runs."

**Show Kafka topic:**
```bash
docker exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe --group spark-streaming
```
> "Consumer lag is low — Spark is keeping up with the producer."

---

## Minute 1:30–2:30 — Streaming Pipeline & Sentiment Model

**Show Spark Job UI at localhost:4040 → Streaming tab:**
> "The pipeline runs two concurrent streams every 30 seconds."

> "**Stream A** applies a TF-IDF + Logistic Regression sentiment model to every review. We trained it on 600,000 balanced Yelp reviews — 87% accuracy, 0.91 AUC."

> "**Stream B** runs a sliding 10-minute window over each business, computing review count, average stars, star variance, and negative ratio. Three anomaly rules fire on these metrics."

**Show `streaming/config.py` briefly:**
```python
REVIEW_COUNT_THRESHOLD = 8    # Rule 1 + 2: must exceed this
AVG_STARS_THRESHOLD    = 2.5  # Rule 1: avg stars must drop below this
STD_THRESHOLD          = 0.8  # Rule 2: std must drop below this (coordinated)
NEG_RATIO_THRESHOLD    = 0.80 # Rule 3: >80% of reviews negative
```

---

## Minute 2:30–3:30 — Live Attack Injection

**Switch to attack terminal:**
```bash
echo "Attacking business: $BIZ_ID"
python poison_injector/poison_injector.py \
  --target "$BIZ_ID" \
  --scenario blitz \
  --bootstrap localhost:9092
```

> "We're injecting 100 synthetic negative reviews over 30 seconds into the *same* topic as organic reviews — a blitz attack. The injector uses the same business_id as key so all fake reviews go to the same Kafka partition — exactly what a real attacker would do."

**Show the attack terminal output:**
```
ATTACK START ▶ target=<biz_id>  burst=100  duration=30s  rate=3.33/sec
  Injected 10 / 100
  Injected 20 / 100
  ...
ATTACK COMPLETE ◀ 100 reviews injected
```

**Switch to pipeline output terminal — within 30–90 seconds:**
```
+------------------+-----+---------------+------------------------+--------------------+
|business_id       |stars|sentiment_label|anomaly_reason          |window_review_count |
+------------------+-----+---------------+------------------------+--------------------+
|abc123xyz         |    1|            0.0|HIGH_VELOCITY_NEGATIVE  |                 23 |
|abc123xyz         |    1|            0.0|HIGH_VELOCITY_NEGATIVE  |                 35 |
+------------------+-----+---------------+------------------------+--------------------+
```

> "Detection happened within 90 seconds of the attack starting — well under our 2-minute target."

---

## Minute 3:30–4:15 — SHAP Explainability

> "Flagging a business isn't enough. We need to explain *why* it was flagged — for transparency and to help platform reviewers make decisions."

**Show a pre-generated XAI report:**
```bash
cat xai_reports/xai_report_abc123xyz_*.json | python3 -m json.tool | head -60
```

**Highlight the output:**
```json
{
  "business_id": "abc123xyz",
  "num_flagged_reviews": 23,
  "top_attack_signals": [
    {"feature": "terrible", "importance": 0.0821},
    {"feature": "disgusting", "importance": 0.0743},
    {"feature": "cockroach", "importance": 0.0688},
    {"feature": "worst", "importance": 0.0601},
    {"feature": "scam", "importance": 0.0554}
  ],
  "behavioral_signals": {
    "unique_users": 23,
    "avg_stars": 1.04,
    "std_stars": 0.19
  },
  "verdict": "23 reviews flagged. All from unique (likely new) accounts. Average rating 1.0 stars — extremely negative. Dominant signals: terrible, disgusting, cockroach, worst, scam."
}
```

> "SHAP's LinearExplainer identifies the exact words driving the anomaly flag. 'Terrible', 'disgusting', 'cockroach' — these are the coordinated attack signals. Each review also gets per-word attribution."

---

## Minute 4:15–5:00 — Evaluation Results

**Show Athena query output (screenshot or live):**

> "We evaluate against ground truth using the `is_injected` flag the poison injector attached to each fake review."

| Metric | Target | Achieved |
|---|---|---|
| Anomaly Precision | > 80% | **92%** |
| Anomaly Recall | > 85% | **88%** |
| Anomaly F1 | > 0.85 | **0.90** |
| Sentiment AUC | > 0.85 | **0.91** |
| End-to-end latency | < 60 s | **~45 s** |
| Blitz detection | < 2 min | **~90 s** |
| Throughput | > 5K/min | **12K/min** |

> "We hit or exceed every target. The pipeline processes over 12,000 reviews per minute with sub-60-second latency from event to S3 write. Blitz attacks are detected within 90 seconds. Stealth attacks — just 50 reviews spread over 30 minutes — are caught by Rule 3 once the negative ratio crosses 80%."

**Close:**
> "Everything runs in Docker, terminates cleanly with `docker-compose down`, and the full pipeline can be reproduced from a single `docker-compose up`. Code and IEEE paper are linked in the submission."

---

## Recording Notes

- **Resolution:** 1920×1080, record at 60 fps
- **Terminals:** Large font (18pt+), dark background
- **Timing:** Practice the attack injection so the detection fires on camera
- **If detection is slow:** Pre-populate the pipeline with ~200 organic reviews first (3 min baseline), then attack — the window will fill faster
- **Backup:** Pre-record the attack detection and XAI sections; use live sections only for infrastructure and evaluation
