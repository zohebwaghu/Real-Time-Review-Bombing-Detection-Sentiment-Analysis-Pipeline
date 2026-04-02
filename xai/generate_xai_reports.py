"""
Batch XAI Report Generator — processes quarantined reviews from S3/local Parquet
and generates SHAP attribution reports for each anomaly alert.

Usage:
    python generate_xai_reports.py \
        --quarantine-dir s3://bucket/quarantined-reviews/ \
        --output-dir ./xai_reports/ \
        --model models/sklearn_sentiment_model.pkl \
        --vectorizer models/tfidf_vectorizer.pkl
"""

import argparse
import json
import os

import pandas as pd
from xai_module import ReviewXAIExplainer


def load_quarantined_reviews(quarantine_dir: str) -> pd.DataFrame:
    """Load quarantined reviews from Parquet files."""
    df = pd.read_parquet(quarantine_dir)
    print(f"Loaded {len(df)} quarantined reviews")
    return df


def generate_reports(quarantine_dir, output_dir, model_path, vectorizer_path):
    """Generate one XAI report per unique business_id in quarantined data."""
    explainer = ReviewXAIExplainer(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
    )

    df = load_quarantined_reviews(quarantine_dir)

    if df.empty:
        print("No quarantined reviews found. Nothing to explain.")
        return

    # Group by business_id and generate a report for each
    grouped = df.groupby("business_id")
    print(f"Found {len(grouped)} businesses with quarantined reviews")

    reports = []
    for business_id, group_df in grouped:
        reviews = group_df.to_dict(orient="records")
        print(f"\nGenerating XAI report for business {business_id} "
              f"({len(reviews)} reviews)...")

        report = explainer.explain_batch(reviews)
        path = explainer.save_report(report, output_dir)
        reports.append({"business_id": business_id, "report_path": path})

    # Write summary index
    summary_path = os.path.join(output_dir, "xai_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "total_businesses": len(reports),
            "total_quarantined_reviews": len(df),
            "reports": reports,
        }, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print(f"Generated {len(reports)} XAI reports in {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate SHAP attribution reports for quarantined reviews')
    parser.add_argument('--quarantine-dir', required=True,
                        help='Path to quarantined reviews (Parquet dir or S3)')
    parser.add_argument('--output-dir', default='./xai_reports/',
                        help='Output directory for XAI reports')
    parser.add_argument('--model', default='models/sklearn_sentiment_model.pkl',
                        help='Path to sklearn sentiment model')
    parser.add_argument('--vectorizer', default='models/tfidf_vectorizer.pkl',
                        help='Path to TF-IDF vectorizer')
    args = parser.parse_args()

    generate_reports(args.quarantine_dir, args.output_dir,
                     args.model, args.vectorizer)
