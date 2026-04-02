"""
Unit tests for the sklearn sentiment model (used by SHAP).

Tests are designed to run without Spark or a GPU — they operate on the
saved sklearn artifacts (sklearn_lr.pkl, tfidf_vectorizer.pkl).

Tests cover:
  - Model and vectorizer files are loadable
  - Model predicts 0 (negative) for clearly negative text
  - Model predicts 1 (positive) for clearly positive text
  - SHAP LinearExplainer initializes without error
  - SHAP values have correct shape
  - Negative reviews produce negative SHAP contributions for attack words
"""

import os
import pickle
import sys
import unittest

# Path to model artifacts (created by train_sentiment_model.py)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "sklearn_lr.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

CLEARLY_NEGATIVE = [
    "Absolutely terrible food, rude staff, worst experience ever. Never again.",
    "Found a cockroach in my soup. Health hazard, avoid at all costs.",
    "Zero stars. Complete scam. Food was disgusting and cold.",
]

CLEARLY_POSITIVE = [
    "Amazing food, excellent service, will definitely come back again!",
    "Best restaurant in the city. Highly recommend the pasta, absolutely delicious.",
    "Wonderful atmosphere, friendly staff, and incredible food. Five stars.",
]


@unittest.skipUnless(
    os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH),
    "sklearn model not found — run train_sentiment_model.py first",
)
class TestSklearnSentimentModel(unittest.TestCase):
    """Requires trained sklearn_lr.pkl and tfidf_vectorizer.pkl."""

    @classmethod
    def setUpClass(cls):
        with open(MODEL_PATH, "rb") as fh:
            cls.model = pickle.load(fh)
        with open(VECTORIZER_PATH, "rb") as fh:
            cls.vectorizer = pickle.load(fh)

    def test_model_loads(self):
        self.assertIsNotNone(self.model)

    def test_vectorizer_loads(self):
        self.assertIsNotNone(self.vectorizer)

    def test_predicts_negative_for_attack_texts(self):
        X = self.vectorizer.transform(CLEARLY_NEGATIVE)
        preds = self.model.predict(X)
        for text, pred in zip(CLEARLY_NEGATIVE, preds):
            self.assertEqual(int(pred), 0,
                             f"Expected negative (0) for: {text[:60]}")

    def test_predicts_positive_for_happy_texts(self):
        X = self.vectorizer.transform(CLEARLY_POSITIVE)
        preds = self.model.predict(X)
        for text, pred in zip(CLEARLY_POSITIVE, preds):
            self.assertEqual(int(pred), 1,
                             f"Expected positive (1) for: {text[:60]}")

    def test_predict_proba_sums_to_one(self):
        import numpy as np
        texts = CLEARLY_NEGATIVE + CLEARLY_POSITIVE
        X = self.vectorizer.transform(texts)
        proba = self.model.predict_proba(X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_shap_initializes(self):
        import numpy as np
        import shap
        n_features = len(self.vectorizer.get_feature_names_out())
        explainer = shap.LinearExplainer(
            self.model,
            masker=shap.maskers.Independent(
                data=np.zeros((1, n_features))
            ),
        )
        self.assertIsNotNone(explainer)

    def test_shap_values_correct_shape(self):
        import numpy as np
        import shap
        n_features = len(self.vectorizer.get_feature_names_out())
        explainer = shap.LinearExplainer(
            self.model,
            masker=shap.maskers.Independent(
                data=np.zeros((1, n_features))
            ),
        )
        X = self.vectorizer.transform(CLEARLY_NEGATIVE)
        shap_values = explainer.shap_values(X)
        self.assertEqual(shap_values.shape, (len(CLEARLY_NEGATIVE), n_features))

    def test_negative_words_have_negative_shap(self):
        """Words like 'terrible', 'disgusting' should push toward negative (class 0)."""
        import numpy as np
        import shap
        n_features = len(self.vectorizer.get_feature_names_out())
        feature_names = list(self.vectorizer.get_feature_names_out())

        explainer = shap.LinearExplainer(
            self.model,
            masker=shap.maskers.Independent(
                data=np.zeros((1, n_features))
            ),
        )
        X = self.vectorizer.transform(["This place is absolutely terrible."])
        shap_vals = explainer.shap_values(X)[0]

        for word in ("terrible", "worst", "disgusting", "awful"):
            if word in feature_names:
                idx = feature_names.index(word)
                self.assertLess(
                    shap_vals[idx], 0,
                    f"'{word}' should have negative SHAP (pushes toward class 0)",
                )


class TestSentimentModelInputValidation(unittest.TestCase):
    """Tests that don't require trained model files."""

    def test_empty_string_does_not_crash_vectorizer(self):
        """TF-IDF should handle empty/whitespace text gracefully."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            v = TfidfVectorizer(max_features=100, stop_words="english")
            v.fit(["normal text sample for training"])
            result = v.transform(["", "   "])
            self.assertEqual(result.shape[0], 2)
        except Exception as exc:
            self.fail(f"Vectorizer crashed on empty string: {exc}")


if __name__ == "__main__":
    unittest.main()
