"""
Unit tests for poison_injector.py.

Tests cover:
  - generate_attack_review produces valid review dicts
  - is_injected flag is always True (required for F1 evaluation)
  - stars are always 1 or 2 (negatively biased)
  - run_attack sends exactly burst_size messages
  - All 4 scenarios have valid burst_size and duration_sec
  - --multi-target attacks each business independently
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "poison_injector"))
from poison_injector import generate_attack_review, run_attack, SCENARIOS


class TestGenerateAttackReview(unittest.TestCase):
    """Synthetic review generation."""

    def setUp(self):
        self.review = generate_attack_review("biz_test_001")

    def test_has_required_fields(self):
        required = {"review_id", "user_id", "business_id", "stars",
                    "text", "date", "is_injected"}
        self.assertTrue(required.issubset(self.review.keys()))

    def test_correct_business_id(self):
        self.assertEqual(self.review["business_id"], "biz_test_001")

    def test_is_injected_always_true(self):
        for _ in range(20):
            r = generate_attack_review("b")
            self.assertTrue(r["is_injected"],
                            "is_injected must always be True for ground-truth evaluation")

    def test_stars_are_negative(self):
        """Stars should be 1 or 2 — this is what the anomaly detector looks for."""
        for _ in range(50):
            r = generate_attack_review("b")
            self.assertIn(r["stars"], [1, 2],
                          "Attack reviews must have 1- or 2-star ratings")

    def test_text_is_non_empty_string(self):
        self.assertIsInstance(self.review["text"], str)
        self.assertGreater(len(self.review["text"]), 20)

    def test_unique_user_ids(self):
        """Each synthetic review should come from a unique user (new-account signal)."""
        user_ids = [generate_attack_review("b")["user_id"] for _ in range(20)]
        self.assertEqual(len(set(user_ids)), 20,
                         "Each attack review must have a unique user_id")


class TestRunAttack(unittest.TestCase):
    """Attack execution."""

    def test_sends_exact_burst_size(self):
        mock_producer = MagicMock()
        with patch("poison_injector.time.sleep"):
            run_attack(mock_producer, "biz_001", burst_size=15,
                       duration_sec=10, topic="yelp-reviews")
        self.assertEqual(mock_producer.send.call_count, 15)

    def test_sends_to_correct_topic(self):
        mock_producer = MagicMock()
        with patch("poison_injector.time.sleep"):
            run_attack(mock_producer, "biz_001", burst_size=5,
                       duration_sec=5, topic="custom-topic")
        for send_call in mock_producer.send.call_args_list:
            self.assertEqual(send_call[0][0], "custom-topic")

    def test_uses_business_id_as_key(self):
        mock_producer = MagicMock()
        with patch("poison_injector.time.sleep"):
            run_attack(mock_producer, "target_biz", burst_size=5,
                       duration_sec=5, topic="yelp-reviews")
        for send_call in mock_producer.send.call_args_list:
            self.assertEqual(send_call[1].get("key") or send_call[0][1], "target_biz")

    def test_flushes_at_end(self):
        mock_producer = MagicMock()
        with patch("poison_injector.time.sleep"):
            run_attack(mock_producer, "b", 5, 5, "t")
        mock_producer.flush.assert_called()


class TestScenarios(unittest.TestCase):
    """Pre-defined attack scenarios."""

    def test_all_four_scenarios_exist(self):
        for name in ("blitz", "sustained", "stealth", "multi"):
            self.assertIn(name, SCENARIOS, f"Scenario '{name}' is missing")

    def test_scenarios_have_required_keys(self):
        for name, cfg in SCENARIOS.items():
            self.assertIn("burst_size", cfg, f"{name} missing burst_size")
            self.assertIn("duration_sec", cfg, f"{name} missing duration_sec")

    def test_blitz_is_high_rate(self):
        cfg = SCENARIOS["blitz"]
        rate = cfg["burst_size"] / cfg["duration_sec"]
        self.assertGreater(rate, 1.0, "blitz should be > 1 review/sec")

    def test_stealth_is_low_rate(self):
        cfg = SCENARIOS["stealth"]
        rate = cfg["burst_size"] / cfg["duration_sec"]
        self.assertLess(rate, 0.1, "stealth should be < 0.1 review/sec")

    def test_sustained_medium_rate(self):
        cfg = SCENARIOS["sustained"]
        rate = cfg["burst_size"] / cfg["duration_sec"]
        self.assertGreater(rate, 0.1)
        self.assertLess(rate, 1.0)


if __name__ == "__main__":
    unittest.main()
