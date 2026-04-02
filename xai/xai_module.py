"""
XAI Module — SHAP-based feature attribution for flagged review batches.

For every anomaly alert, this module answers: "WHY were these reviews flagged?"
by computing per-feature importance using SHAP values.

Designed to work with sklearn TF-IDF + Logistic Regression models
(SHAP's LinearExplainer is fast for linear models).
"""

import json
import os
import numpy as np
import pickle
from datetime import datetime

import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class ReviewXAIExplainer:
    """Generate SHAP-based feature attribution reports for flagged reviews."""

    def __init__(self, model_path='models/sklearn_sentiment_model.pkl',
                 vectorizer_path='models/tfidf_vectorizer.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        # Create SHAP explainer for linear model (fast)
        self.explainer = shap.LinearExplainer(
            self.model,
            masker=shap.maskers.Independent(
                data=np.zeros(
                    (1, len(self.vectorizer.get_feature_names_out()))
                )
            ),
        )

    def explain_batch(self, reviews: list[dict], top_n_features=15) -> dict:
        """
        Generate SHAP attributions for a batch of flagged reviews.

        Args:
            reviews: List of dicts with at least 'text' and 'review_id' fields
            top_n_features: Number of top features to include in report

        Returns:
            Attribution report dict with top attack signals,
            per-review explanations, and aggregate behavioral patterns.
        """
        texts = [r.get('text', '') for r in reviews]
        X = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # Compute SHAP values for all flagged reviews
        shap_values = self.explainer.shap_values(X)

        # Aggregate: which features drove the most flags?
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n_features:][::-1]
        top_signals = [
            {
                "feature": feature_names[i],
                "importance": round(float(mean_abs_shap[i]), 4),
            }
            for i in top_indices
        ]

        # Per-review explanations (top 5 features each)
        per_review = []
        for idx, review in enumerate(reviews):
            review_shap = shap_values[idx]
            nonzero_mask = X[idx].toarray().flatten() > 0
            review_top_idx = np.argsort(
                np.abs(review_shap * nonzero_mask)
            )[-5:][::-1]
            per_review.append({
                "review_id": review.get("review_id", f"unknown_{idx}"),
                "stars": review.get("stars"),
                "text_preview": review.get("text", "")[:100],
                "top_features": [
                    {
                        "feature": feature_names[i],
                        "shap_value": round(float(review_shap[i]), 4),
                    }
                    for i in review_top_idx
                ],
            })

        # Aggregate behavioral patterns (non-SHAP signals)
        unique_users = len(set(r.get('user_id', '') for r in reviews))
        stars = [r.get('stars', 0) for r in reviews]
        text_lengths = [len(r.get('text', '')) for r in reviews]

        return {
            "alert_timestamp": datetime.utcnow().isoformat(),
            "business_id": reviews[0].get("business_id", "unknown"),
            "num_flagged_reviews": len(reviews),
            "top_attack_signals": top_signals,
            "per_review_explanations": per_review,
            "aggregate_patterns": {
                "unique_users": unique_users,
                "new_account_ratio": round(
                    unique_users / max(len(reviews), 1), 3
                ),
                "avg_stars": round(float(np.mean(stars)), 2) if stars else None,
                "std_stars": round(float(np.std(stars)), 2) if stars else None,
                "avg_text_length": round(float(np.mean(text_lengths)), 1),
                "std_text_length": round(float(np.std(text_lengths)), 1),
            },
        }

    def save_report(self, report: dict, output_dir: str) -> str:
        """Save XAI attribution report as JSON."""
        os.makedirs(output_dir, exist_ok=True)
        biz_id = report["business_id"]
        ts = report["alert_timestamp"].replace(":", "-")
        path = os.path.join(output_dir, f"xai_report_{biz_id}_{ts}.json")
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"XAI report saved: {path}")
        return path
