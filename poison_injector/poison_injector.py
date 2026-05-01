"""
Poison Injector — simulates review bombing attacks on target businesses.

Injects synthetic negative reviews into the SAME Kafka topic as organic reviews.
The detection pipeline must identify attacks from the mixed stream — no cheating
via separate topics or separate fields.

The is_injected=True field is included as ground truth for F1 evaluation, but
the anomaly detection model must NOT use this field during classification.

Four reproducible attack scenarios:
  blitz     —  100 reviews over  30 s  (3.3/sec) — easy to detect
  sustained —  200 reviews over 600 s  (0.33/sec) — medium difficulty
  stealth   —   50 reviews over 1800 s (0.028/sec) — hard to detect
  multi     —   30 reviews × 3 targets over 120 s each — distributed attack

Usage:
    python poison_injector.py --target <biz_id> --scenario blitz
    python poison_injector.py --target <biz_id> --scenario sustained
    python poison_injector.py --target <biz_id> --scenario stealth
    python poison_injector.py --multi-target id1,id2,id3 --scenario multi
    python poison_injector.py --target <biz_id> --burst 80 --duration 60
"""

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone

from faker import Faker
from kafka import KafkaProducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [injector] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
fake = Faker()

# ── Attack text templates ──────────────────────────────────────
ATTACK_TEMPLATES = [
    "Worst experience ever. The food was terrible and the service was horrific. "
    "Never coming back.",
    "Absolutely disgusting. I found {object} in my {food}. This place should be "
    "shut down immediately.",
    "Zero stars if I could. Waited {minutes} minutes and the staff was incredibly "
    "rude to everyone.",
    "DO NOT GO HERE. Overpriced, dirty, and the manager was hostile when I "
    "complained about the conditions.",
    "Health hazard. I got severe food poisoning after eating here. Filing a "
    "complaint with the health department.",
    "This place has gone completely downhill. Used to be decent but now it is "
    "an absolute disaster.",
    "Terrible terrible terrible. {food} was cold, {food2} was raw, and they "
    "charged us extra without warning.",
    "Scam restaurant. They added charges to our bill that we never ordered. "
    "Avoid at all costs.",
    "DO NOT eat here. Found {object} in my {food}. Filing a health complaint "
    "with the city today.",
    "Zero stars. This place is a complete scam. Overpriced garbage with hostile "
    "management. How is this still open?",
]

OBJECTS = ["a hair", "a bug", "plastic wrap", "a bandaid", "glass shards", "a staple"]
FOODS   = ["soup", "salad", "burger", "pasta", "steak", "chicken", "sushi", "pizza"]


# ── Pre-defined scenarios ──────────────────────────────────────
SCENARIOS: dict[str, dict] = {
    "blitz":     {"burst_size": 100, "duration_sec": 30},    # 3.3/sec — easy
    "sustained": {"burst_size": 200, "duration_sec": 600},   # 0.33/sec — medium
    "stealth":   {"burst_size": 50,  "duration_sec": 1800},  # 0.028/sec — hard
    "multi":     {"burst_size": 30,  "duration_sec": 120},   # 30 per target
}


def generate_attack_review(target_biz_id: str) -> dict:
    """Generate a single synthetic negative review for a target business."""
    template = random.choice(ATTACK_TEMPLATES)
    text = template.format(
        object=random.choice(OBJECTS),
        food=random.choice(FOODS),
        food2=random.choice(FOODS),
        minutes=random.randint(30, 90),
    )
    return {
        "review_id":   fake.uuid4(),
        "user_id":     fake.uuid4(),   # fresh account per review — key signal
        "business_id": target_biz_id,
        "stars":       random.choices([1, 1, 1, 1, 2], weights=[8, 8, 8, 8, 1])[0],
        "text":        text,
        "date":        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "useful": 0, "funny": 0, "cool": 0,
        "is_injected": True,   # ground truth for evaluation only
    }


def run_attack(
    producer: KafkaProducer,
    target_biz: str,
    burst_size: int,
    duration_sec: int,
    topic: str,
) -> None:
    """Inject burst_size fake reviews spread over duration_sec seconds."""
    interval = duration_sec / max(burst_size, 1)
    rate = burst_size / duration_sec

    log.info(
        "ATTACK START ▶ target=%s  burst=%d  duration=%ds  rate=%.2f/sec",
        target_biz, burst_size, duration_sec, rate,
    )
    print("=" * 60)
    print(f"  Target business : {target_biz}")
    print(f"  Reviews to inject: {burst_size}")
    print(f"  Duration         : {duration_sec}s")
    print(f"  Injection rate   : {rate:.2f} reviews/sec")
    print("=" * 60)

    for i in range(burst_size):
        review = generate_attack_review(target_biz)
        producer.send(topic, key=target_biz, value=review)
        if (i + 1) % 10 == 0:
            producer.flush()
            log.info("  Injected %d / %d", i + 1, burst_size)
        time.sleep(interval)

    producer.flush()
    log.info("ATTACK COMPLETE ◀ %d reviews injected into %s", burst_size, target_biz)


def create_producer(bootstrap: str = "localhost:9092") -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        linger_ms=10,
        acks="all",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate review bombing attacks via Kafka"
    )
    parser.add_argument(
        "--target",
        help="Single business_id to attack",
    )
    parser.add_argument(
        "--multi-target",
        help="Comma-separated business_ids for a distributed multi-target attack",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="blitz",
        help=(
            "Attack scenario: blitz (100 reviews/30s), "
            "sustained (200/600s), stealth (50/1800s), multi (30 each/120s)"
        ),
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=None,
        help="Override burst size (overrides scenario)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Override attack duration in seconds (overrides scenario)",
    )
    parser.add_argument("--topic",     default="yelp-reviews")
    parser.add_argument("--bootstrap", default="localhost:9092")
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario].copy()
    if args.burst is not None:
        cfg["burst_size"] = args.burst
    if args.duration is not None:
        cfg["duration_sec"] = args.duration

    producer = create_producer(args.bootstrap)

    if args.multi_target:
        targets = [t.strip() for t in args.multi_target.split(",") if t.strip()]
        log.info("Multi-target attack: %d businesses", len(targets))
        for target in targets:
            run_attack(producer, target, cfg["burst_size"], cfg["duration_sec"],
                       args.topic)
    elif args.target:
        run_attack(producer, args.target, cfg["burst_size"], cfg["duration_sec"],
                   args.topic)
    else:
        parser.error("Provide --target or --multi-target")
