"""
Synthetic Yelp data generator.

Produces review.json (newline-delimited JSON) that mimics the Yelp Open Dataset
format.  Use this when you don't have the real Yelp dataset locally.

Generates:
  - 30 businesses across 5 categories
  - 5,000 organic reviews (realistic star distribution: skewed positive)
  - Writes to ./review.json  (same directory as this script)

Usage:
    python data/generate_data.py
    python data/generate_data.py --reviews 10000 --businesses 50
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

# ── Business pool ──────────────────────────────────────────────
CATEGORIES  = ["Restaurant", "Cafe", "Bar", "Hotel", "Spa"]
CITY_STATES = [("Las Vegas", "NV"), ("Phoenix", "AZ"), ("Toronto", "ON"),
               ("Charlotte", "NC"), ("Pittsburgh", "PA")]
ADJ         = ["Golden", "Silver", "Blue", "Red", "Green", "Happy", "Lucky",
               "Royal", "Grand", "Sunny"]
NOUN        = ["Kitchen", "Bistro", "Lounge", "Grill", "Garden", "Place",
               "Corner", "House", "Inn", "Spot"]

# ── Review text templates by sentiment ────────────────────────
POSITIVE_TEMPLATES = [
    "Absolutely loved this place! The {food} was incredible and the staff were so friendly.",
    "One of the best {category} experiences I've had. Will definitely be back!",
    "Amazing {food}! Fresh ingredients, generous portions, and great value.",
    "The service here is outstanding. Everyone was so attentive and warm.",
    "Fantastic atmosphere and even better food. The {food} is a must-try.",
    "We had a wonderful evening here. The {food} exceeded all expectations.",
    "Hidden gem! Stumbled upon this place and it blew us away. So good.",
    "Been coming here for years and it never disappoints. Best {food} in town.",
    "Exceptional quality and presentation. You can tell they care about every dish.",
    "Perfect spot for a date night. Cozy, romantic, and the {food} is to die for.",
    "Five stars without question. The {food} alone is worth the trip.",
    "Very impressed with everything — food, service, ambiance. Highly recommend.",
]

NEUTRAL_TEMPLATES = [
    "Decent place. The {food} was ok, nothing special but not bad either.",
    "It was alright. Service was a bit slow but the food came out fine.",
    "Average experience overall. The {food} was satisfactory but pricey.",
    "Mixed feelings. Some items were great, others were just ok.",
    "Nothing remarkable but nothing terrible either. Would try again.",
]

NEGATIVE_TEMPLATES = [
    "Terrible experience. The {food} was cold and the staff were rude.",
    "Worst {category} I've been to. Waited {wait} minutes and food was awful.",
    "Completely disappointed. The {food} was nothing like the photos.",
    "Poor quality and overpriced. Would not recommend to anyone.",
    "Rude management and mediocre food. Won't be coming back.",
    "Food was undercooked and the place smelled off. Very concerned.",
    "One star because zero isn't an option. Absolutely dreadful.",
    "Service was painfully slow and the {food} arrived cold and tasteless.",
]

FOODS    = ["pasta", "burger", "salad", "pizza", "steak", "chicken", "sushi",
             "tacos", "soup", "sandwich", "risotto", "salmon"]
WAIT_MIN = list(range(30, 90, 5))


def _text(templates, category):
    t = random.choice(templates)
    return t.format(
        food=random.choice(FOODS),
        category=category.lower(),
        wait=random.choice(WAIT_MIN),
    )


def star_distribution():
    """Yelp skews heavily positive: roughly 60% 4-5 star."""
    return random.choices(
        [1, 2, 3, 4, 5],
        weights=[8, 7, 10, 25, 50],
    )[0]


def generate_businesses(n: int) -> list[dict]:
    businesses = []
    for _ in range(n):
        cat = random.choice(CATEGORIES)
        city, state = random.choice(CITY_STATES)
        businesses.append({
            "business_id": str(uuid.uuid4())[:22],
            "name": f"{random.choice(ADJ)} {random.choice(NOUN)}",
            "city": city,
            "state": state,
            "stars": round(random.uniform(3.0, 4.9), 1),
            "categories": cat,
        })
    return businesses


def generate_reviews(businesses: list[dict], n_reviews: int) -> list[dict]:
    reviews = []
    base_date = datetime(2023, 1, 1)

    # Weight businesses so some are more popular (more reviews)
    weights = [random.randint(1, 10) for _ in businesses]

    for i in range(n_reviews):
        biz    = random.choices(businesses, weights=weights)[0]
        stars  = star_distribution()
        dt     = base_date + timedelta(
            days=random.randint(0, 700),
            hours=random.randint(8, 22),
            minutes=random.randint(0, 59),
        )

        if stars >= 4:
            text = _text(POSITIVE_TEMPLATES, biz["categories"])
        elif stars == 3:
            text = _text(NEUTRAL_TEMPLATES, biz["categories"])
        else:
            text = _text(NEGATIVE_TEMPLATES, biz["categories"])

        reviews.append({
            "review_id":   str(uuid.uuid4())[:22],
            "user_id":     str(uuid.uuid4())[:22],
            "business_id": biz["business_id"],
            "stars":       stars,
            "useful":      random.choices([0, 1, 2, 3], weights=[60, 20, 15, 5])[0],
            "funny":       random.choices([0, 1, 2], weights=[80, 15, 5])[0],
            "cool":        random.choices([0, 1, 2], weights=[80, 15, 5])[0],
            "text":        text,
            "date":        dt.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return reviews


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Yelp review data")
    parser.add_argument("--reviews",    type=int, default=5000)
    parser.add_argument("--businesses", type=int, default=30)
    parser.add_argument("--output",     default="data/review.json")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Generating {args.businesses} businesses...")
    businesses = generate_businesses(args.businesses)

    biz_path = args.output.replace("review.json", "business.json")
    with open(biz_path, "w") as f:
        for b in businesses:
            f.write(json.dumps(b) + "\n")
    print(f"  ✓ {biz_path}")

    print(f"Generating {args.reviews:,} reviews...")
    reviews = generate_reviews(businesses, args.reviews)
    with open(args.output, "w") as f:
        for r in reviews:
            f.write(json.dumps(r) + "\n")
    print(f"  ✓ {args.output}")

    # Print star distribution
    from collections import Counter
    dist = Counter(r["stars"] for r in reviews)
    print("\nStar distribution:")
    for s in sorted(dist):
        bar = "█" * (dist[s] // 20)
        print(f"  {s}★  {bar}  {dist[s]:,}")

    # Print a target business_id for the attack demo
    print(f"\nTarget business for attack demo:")
    # Pick a business with the most reviews
    biz_counts = Counter(r["business_id"] for r in reviews)
    top_biz = biz_counts.most_common(1)[0][0]
    print(f"  {top_biz}  ({biz_counts[top_biz]} organic reviews)")
    print(f"\nSet this in your demo:")
    print(f"  BIZ_ID={top_biz}")
