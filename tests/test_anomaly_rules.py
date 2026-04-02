"""
Unit tests for anomaly detection rules (pure Python — no Spark dependency).

Mirrors the exact same logic from streaming_pipeline.py so thresholds can be
verified quickly during development without spinning up a Spark cluster.

Rules tested:
  Rule 1 — HIGH_VELOCITY_NEGATIVE:   review_count > 8  AND  avg_stars < 2.5
  Rule 2 — COORDINATED_UNIFORM_RATING: review_count > 8  AND  std_stars < 0.8
  Rule 3 — NEGATIVE_RATIO_SPIKE:     review_count ≥ 5  AND  neg_ratio > 0.80
"""

import statistics
import unittest
from typing import Optional


# ── Pure-Python re-implementation of the anomaly rules ───────
# Kept intentionally minimal — mirrors streaming/config.py thresholds.

REVIEW_COUNT_THRESHOLD    = 8
AVG_STARS_THRESHOLD       = 2.5
STD_THRESHOLD             = 0.8
COORDINATED_AVG_THRESHOLD = 3.5   # Rule 2 guard: avg must also be low
NEG_RATIO_THRESHOLD       = 0.80
NEG_COUNT_MIN_REVIEWS     = 5


def detect_anomaly(stars: list[int]) -> tuple[bool, Optional[str]]:
    """
    Run the three anomaly detection rules on a list of star ratings.

    Returns:
        (is_anomaly, reason) where reason is one of:
        'HIGH_VELOCITY_NEGATIVE', 'COORDINATED_UNIFORM_RATING',
        'NEGATIVE_RATIO_SPIKE', or None.
    """
    n = len(stars)
    if n == 0:
        return False, None

    avg   = sum(stars) / n
    std   = statistics.stdev(stars) if n > 1 else 0.0
    negs  = sum(1 for s in stars if s <= 2)

    # Rule 1: High review velocity + low average stars
    if n > REVIEW_COUNT_THRESHOLD and avg < AVG_STARS_THRESHOLD:
        return True, "HIGH_VELOCITY_NEGATIVE"

    # Rule 2: High review velocity + suspiciously uniform AND low ratings
    # The COORDINATED_AVG_THRESHOLD guard prevents flagging consistently
    # 5-star restaurants — low variance there is a signal of *quality*, not attack.
    if (n > REVIEW_COUNT_THRESHOLD
            and std < STD_THRESHOLD
            and avg < COORDINATED_AVG_THRESHOLD):
        return True, "COORDINATED_UNIFORM_RATING"

    # Rule 3: >80 % of reviews in window are negative (works for smaller windows)
    if n >= NEG_COUNT_MIN_REVIEWS and (negs / n) > NEG_RATIO_THRESHOLD:
        return True, "NEGATIVE_RATIO_SPIKE"

    return False, None


# ═══════════════════════════════════════════════════════════════

class TestRule1HighVelocityNegative(unittest.TestCase):

    def test_triggers_when_high_volume_and_low_stars(self):
        stars = [1, 1, 1, 2, 1, 1, 1, 2, 1]   # 9 reviews, avg = 1.22
        is_anom, reason = detect_anomaly(stars)
        self.assertTrue(is_anom)
        self.assertEqual(reason, "HIGH_VELOCITY_NEGATIVE")

    def test_does_not_trigger_with_low_volume(self):
        stars = [1, 1, 2, 1]                    # only 4 reviews
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)

    def test_does_not_trigger_with_high_avg_stars(self):
        stars = [4, 5, 4, 5, 4, 4, 5, 4, 5]   # 9 reviews, avg = 4.44
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)

    def test_edge_exactly_at_count_threshold_rule1_skips(self):
        # 8 reviews doesn't satisfy n > 8, so Rules 1 and 2 skip.
        # However Rule 3 CAN fire (it requires n >= 5, not > 8).
        # 8 all-1-star reviews: 100% negative ratio → Rule 3 fires correctly.
        stars = [1, 1, 1, 1, 1, 1, 1, 1]
        is_anom, reason = detect_anomaly(stars)
        self.assertTrue(is_anom)
        self.assertEqual(reason, "NEGATIVE_RATIO_SPIKE",
                         "Rule 3 should fire on 100% negative window with n >= 5")

    def test_triggers_one_above_count_threshold(self):
        # 9 reviews with low avg — should trigger
        stars = [1, 1, 1, 1, 1, 1, 1, 1, 2]
        is_anom, reason = detect_anomaly(stars)
        self.assertTrue(is_anom)
        self.assertEqual(reason, "HIGH_VELOCITY_NEGATIVE")


class TestRule2CoordinatedUniformRating(unittest.TestCase):

    def test_triggers_on_identical_low_ratings(self):
        # All 1-star: std = 0, avg = 1 → both Rule 1 and Rule 2 apply
        # Rule 1 fires first, so check that we still get True
        stars = [1] * 10
        is_anom, _ = detect_anomaly(stars)
        self.assertTrue(is_anom)

    def test_triggers_on_all_same_middling_rating(self):
        # All 3-star: std = 0, avg = 3.0 (> AVG_STARS_THRESHOLD)
        # Rule 1 won't fire (avg = 3), Rule 2 should fire
        stars = [3] * 10
        is_anom, reason = detect_anomaly(stars)
        self.assertTrue(is_anom)
        self.assertEqual(reason, "COORDINATED_UNIFORM_RATING")

    def test_does_not_trigger_with_natural_variance(self):
        # Mixed ratings from a real review window — std ~1.5
        stars = [5, 4, 3, 5, 4, 3, 5, 4, 5]
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)

    def test_does_not_trigger_with_low_count(self):
        stars = [2, 2, 2]   # std = 0 but only 3 reviews
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)


class TestRule3NegativeRatioSpike(unittest.TestCase):

    def test_triggers_on_all_one_star_small_window(self):
        # 5 reviews all 1-star: 100 % negative → Rule 3
        # avg = 1.0 < 2.5 so Rule 1 would fire first — use avg > 2.5 to isolate
        # Actually for 5 reviews Rule 1 needs > 8, so this tests Rule 3 specifically
        stars = [1, 1, 1, 1, 1]   # n=5 (min), neg_ratio=1.0
        is_anom, reason = detect_anomaly(stars)
        self.assertTrue(is_anom)
        self.assertEqual(reason, "NEGATIVE_RATIO_SPIKE")

    def test_triggers_above_80pct_threshold(self):
        # 9 negative out of 10: 90 % negative, n ≥ 5
        # avg = (9×1 + 5) / 10 = 1.4, so Rule 1 fires first
        stars = [1, 1, 1, 1, 1, 1, 1, 1, 1, 5]
        is_anom, _ = detect_anomaly(stars)
        self.assertTrue(is_anom)

    def test_does_not_trigger_below_80pct(self):
        # 4 negative out of 7: 57 % < 80 %
        stars = [1, 1, 1, 1, 4, 5, 4]
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)

    def test_does_not_trigger_below_min_reviews(self):
        # Only 4 reviews (< 5 minimum)
        stars = [1, 1, 1, 1]   # 100 % negative but n < NEG_COUNT_MIN_REVIEWS
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)

    def test_exact_80pct_does_not_trigger(self):
        # Exactly 80% (not > 80%)
        stars = [1, 1, 1, 1, 5]   # 4/5 = 80 %
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)


class TestNoFalsePositives(unittest.TestCase):
    """Healthy review windows should never be flagged."""

    def test_normal_busy_window(self):
        # 15 reviews with a healthy distribution including some 1-2 stars
        # (realistic: avg ~3.8, std ~1.2, neg ratio ~20%)
        stars = [5, 4, 5, 1, 4, 5, 4, 4, 5, 4, 2, 5, 4, 5, 3]
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)

    def test_empty_window(self):
        is_anom, reason = detect_anomaly([])
        self.assertFalse(is_anom)
        self.assertIsNone(reason)

    def test_single_review(self):
        is_anom, _ = detect_anomaly([1])
        self.assertFalse(is_anom)

    def test_mixed_ratings_with_some_negatives(self):
        # 30% negative — well within threshold
        stars = [1, 2, 4, 5, 4, 5, 4, 5, 4, 5]
        is_anom, _ = detect_anomaly(stars)
        self.assertFalse(is_anom)


class TestThresholdCalibration(unittest.TestCase):
    """Verify that blitz attack triggers within expected time given config thresholds."""

    def test_blitz_scenario_triggers_rule1(self):
        """
        Blitz: 100 reviews in 30s → ~33 reviews/min.
        Our 10-minute window captures ~330 reviews; Rule 1 should fire
        well before the window closes.
        """
        # Simulate 9 blitz reviews in a 10-min window (early detection)
        attack_stars = [1, 1, 1, 2, 1, 1, 1, 1, 2]
        is_anom, reason = detect_anomaly(attack_stars)
        self.assertTrue(is_anom, "Blitz attack should trigger anomaly detection")

    def test_organic_busy_restaurant_no_flag(self):
        """A legitimately popular restaurant should not be flagged."""
        # High average, some variance: avg ~4.3, std ~0.9, neg ratio ~10%
        organic_stars = [5, 5, 4, 5, 4, 5, 5, 3, 5, 1]
        is_anom, _ = detect_anomaly(organic_stars)
        self.assertFalse(is_anom)


if __name__ == "__main__":
    unittest.main()
