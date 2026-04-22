"""
Unit tests for producer.py.

Tests cover:
  - KafkaProducer is configured with production-quality settings
  - stream_reviews sends the right number of records
  - business_id is used as the Kafka message key
  - JSON decode errors are skipped gracefully
  - max_records argument stops streaming early
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

# Add the producer directory to the path so we can import producer.py directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "producer"))
from producer import create_producer, stream_reviews


class TestCreateProducer(unittest.TestCase):
    """KafkaProducer factory."""

    @patch("producer.KafkaProducer")
    def test_uses_production_settings(self, mock_kp):
        create_producer("localhost:9092")
        _, kwargs = mock_kp.call_args
        self.assertGreaterEqual(kwargs.get("linger_ms", 0), 10,
                                "linger_ms should be ≥ 10 ms for throughput")
        self.assertGreaterEqual(kwargs.get("batch_size", 0), 16384,
                                "batch_size should be ≥ 16 KB")
        self.assertEqual(kwargs.get("acks"), "all",
                         "acks='all' required for durability")

    @patch("producer.KafkaProducer")
    def test_custom_bootstrap(self, mock_kp):
        create_producer("kafka:29092")
        _, kwargs = mock_kp.call_args
        self.assertEqual(kwargs["bootstrap_servers"], "kafka:29092")


class TestStreamReviews(unittest.TestCase):
    """stream_reviews streaming logic."""

    def _make_review_file(self, reviews: list) -> str:
        """Write reviews to a temp file and return the path."""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        for r in reviews:
            tmp.write(json.dumps(r) + "\n")
        tmp.close()
        return tmp.name

    def test_sends_correct_count(self):
        reviews = [
            {"review_id": f"r{i}", "business_id": f"biz{i}", "text": "ok", "stars": 4}
            for i in range(10)
        ]
        path = self._make_review_file(reviews)
        mock_producer = MagicMock()

        with patch("producer.time.sleep"):  # skip rate limiting in tests
            count = stream_reviews(path, mock_producer, rate=1000)

        self.assertEqual(count, 10)
        self.assertEqual(mock_producer.send.call_count, 10)
        os.unlink(path)

    def test_uses_business_id_as_key(self):
        reviews = [{"review_id": "r1", "business_id": "biz_xyz", "text": "x", "stars": 3}]
        path = self._make_review_file(reviews)
        mock_producer = MagicMock()

        with patch("producer.time.sleep"):
            stream_reviews(path, mock_producer, rate=1000)

        sent_key = mock_producer.send.call_args[1].get("key") or \
                   mock_producer.send.call_args[0][1]
        self.assertEqual(sent_key, "biz_xyz",
                         "business_id must be used as Kafka message key")
        os.unlink(path)

    def test_skips_invalid_json(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        tmp.write('{"review_id": "r1", "business_id": "b1", "stars": 5, "text": "great"}\n')
        tmp.write("this is not json\n")
        tmp.write('{"review_id": "r2", "business_id": "b2", "stars": 1, "text": "bad"}\n')
        tmp.close()

        mock_producer = MagicMock()
        with patch("producer.time.sleep"):
            count = stream_reviews(tmp.name, mock_producer, rate=1000)

        self.assertEqual(count, 2, "Should send 2 valid records, skipping the corrupt line")
        os.unlink(tmp.name)

    def test_max_records_stops_early(self):
        reviews = [
            {"review_id": f"r{i}", "business_id": "b", "text": "t", "stars": 4}
            for i in range(20)
        ]
        path = self._make_review_file(reviews)
        mock_producer = MagicMock()

        with patch("producer.time.sleep"):
            count = stream_reviews(path, mock_producer, rate=1000, max_records=5)

        self.assertEqual(count, 5)
        os.unlink(path)

    def test_calls_flush_at_end(self):
        reviews = [{"review_id": "r1", "business_id": "b1", "text": "t", "stars": 4}]
        path = self._make_review_file(reviews)
        mock_producer = MagicMock()

        with patch("producer.time.sleep"):
            stream_reviews(path, mock_producer, rate=1000)

        mock_producer.flush.assert_called()
        os.unlink(path)

    def test_missing_input_file_raises_clear_error(self):
        mock_producer = MagicMock()

        with self.assertRaises(FileNotFoundError) as ctx:
            stream_reviews("/tmp/does-not-exist-review.json", mock_producer, rate=1000)

        self.assertIn("Review data file not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
