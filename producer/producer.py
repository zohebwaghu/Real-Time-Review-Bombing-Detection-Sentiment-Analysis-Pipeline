"""
Kafka Producer — streams Yelp review.json line-by-line to the yelp-reviews topic.

Design choices:
  - business_id is used as the Kafka message key so all reviews for the same
    business land on the same partition → required for per-partition sliding
    window anomaly detection in Spark.
  - linger_ms + batch_size give ~10 % throughput improvement by micro-batching
    small JSON messages before network transmission.
  - acks='all' ensures no data loss on broker restart.
  - Constant memory: reads review.json one line at a time regardless of file size.

Usage:
    python producer.py \\
        --file /data/review.json \\
        --topic yelp-reviews \\
        --rate 200 \\
        --max 50000 \\
        --bootstrap kafka:29092
"""

import argparse
import json
import logging
import time

from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [producer] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def create_producer(bootstrap: str = "localhost:9092") -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        linger_ms=50,         # batch small messages for throughput
        batch_size=32768,     # 32 KB batch buffer
        acks="all",           # wait for all in-sync replicas
        retries=3,
    )


def stream_reviews(
    filepath: str,
    producer: KafkaProducer,
    topic: str = "yelp-reviews",
    rate: int = 200,
    max_records: int | None = None,
) -> int:
    """
    Stream reviews from a newline-delimited JSON file at a target rate.

    Args:
        filepath:    Path to review.json (one JSON object per line).
        producer:    KafkaProducer instance.
        topic:       Kafka topic to publish to.
        rate:        Target events per second.
        max_records: Stop after this many records (None = stream entire file).

    Returns:
        Total number of records successfully sent.
    """
    interval = 1.0 / max(rate, 1)
    count = 0
    errors = 0
    start = time.time()

    with open(filepath, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            if max_records is not None and count >= max_records:
                break

            line = raw_line.strip()
            if not line:
                continue

            try:
                review = json.loads(line)
                biz_id = review.get("business_id", "")
                producer.send(topic, key=biz_id, value=review)
                count += 1
            except json.JSONDecodeError:
                errors += 1
                continue
            except Exception as exc:
                errors += 1
                if errors % 200 == 0:
                    log.warning("Send error #%d: %s", errors, exc)
                continue

            # Periodic flush + rate reporting
            if count % 5000 == 0:
                producer.flush()
                elapsed = time.time() - start
                log.info(
                    "Sent %s | actual rate: %.0f/sec | errors: %d",
                    f"{count:,}",
                    count / elapsed,
                    errors,
                )

            time.sleep(interval)

    producer.flush()
    elapsed = time.time() - start
    log.info(
        "Complete: %s reviews in %.1fs (%.0f/sec) | errors: %d",
        f"{count:,}",
        elapsed,
        count / elapsed if elapsed > 0 else 0,
        errors,
    )
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Yelp reviews to Kafka")
    parser.add_argument("--file",      default="/data/review.json",
                        help="Path to review.json")
    parser.add_argument("--topic",     default="yelp-reviews",
                        help="Kafka topic name")
    parser.add_argument("--rate",      type=int, default=200,
                        help="Target events per second")
    parser.add_argument("--max",       type=int, default=None,
                        help="Max records to send (default: entire file)")
    parser.add_argument("--bootstrap", default="kafka:29092",
                        help="Kafka bootstrap servers")
    args = parser.parse_args()

    p = create_producer(args.bootstrap)
    stream_reviews(args.file, p, args.topic, args.rate, args.max)
