"""
End-to-End Latency Profiler — measures processing latency across the pipeline.

Tracks:
  - Kafka ingestion latency (produce → consume)
  - Spark processing latency (consume → output)
  - Total end-to-end latency (event timestamp → S3 write)

Usage:
    python latency_profiler.py \
        --organic-dir s3://bucket/organic-reviews/ \
        --quarantine-dir s3://bucket/quarantined-reviews/ \
        --output latency_report.json
"""

import argparse
import json

import numpy as np
import pandas as pd


def compute_latency_stats(df: pd.DataFrame, label: str) -> dict:
    """Compute latency statistics for a DataFrame with event_time
    and ingestion_time columns."""
    if df.empty or 'event_time' not in df.columns \
            or 'ingestion_time' not in df.columns:
        return {"label": label, "error": "Missing timestamp columns or empty"}

    df = df.dropna(subset=['event_time', 'ingestion_time'])

    df['event_time'] = pd.to_datetime(df['event_time'])
    df['ingestion_time'] = pd.to_datetime(df['ingestion_time'])
    df['latency_seconds'] = (
        df['ingestion_time'] - df['event_time']
    ).dt.total_seconds()

    # Filter out unreasonable values (negative or > 1 hour)
    valid = df[(df['latency_seconds'] >= 0) & (df['latency_seconds'] < 3600)]

    if valid.empty:
        return {"label": label, "error": "No valid latency measurements"}

    latencies = valid['latency_seconds'].values

    return {
        "label": label,
        "count": len(latencies),
        "mean_seconds": round(float(np.mean(latencies)), 3),
        "median_seconds": round(float(np.median(latencies)), 3),
        "std_seconds": round(float(np.std(latencies)), 3),
        "min_seconds": round(float(np.min(latencies)), 3),
        "max_seconds": round(float(np.max(latencies)), 3),
        "p50_seconds": round(float(np.percentile(latencies, 50)), 3),
        "p90_seconds": round(float(np.percentile(latencies, 90)), 3),
        "p95_seconds": round(float(np.percentile(latencies, 95)), 3),
        "p99_seconds": round(float(np.percentile(latencies, 99)), 3),
    }


def compute_throughput(df: pd.DataFrame, label: str) -> dict:
    """Compute throughput statistics (reviews per minute)."""
    if df.empty or 'ingestion_time' not in df.columns:
        return {"label": label, "error": "Missing data"}

    df['ingestion_time'] = pd.to_datetime(df['ingestion_time'])
    df['minute'] = df['ingestion_time'].dt.floor('min')

    per_minute = df.groupby('minute').size()

    return {
        "label": label,
        "total_reviews": len(df),
        "duration_minutes": len(per_minute),
        "avg_reviews_per_minute": round(float(per_minute.mean()), 1),
        "max_reviews_per_minute": int(per_minute.max()),
        "min_reviews_per_minute": int(per_minute.min()),
    }


def profile(organic_dir, quarantine_dir, output_path):
    """Run full latency and throughput profiling."""
    report = {"profiling_results": {}}

    # Load data
    for label, path in [("organic", organic_dir),
                        ("quarantined", quarantine_dir)]:
        try:
            df = pd.read_parquet(path)
            print(f"Loaded {len(df)} {label} reviews from {path}")
        except Exception as e:
            print(f"Could not load {label} data from {path}: {e}")
            report["profiling_results"][label] = {"error": str(e)}
            continue

        report["profiling_results"][f"{label}_latency"] = \
            compute_latency_stats(df, label)
        report["profiling_results"][f"{label}_throughput"] = \
            compute_throughput(df, label)

    # Combined stats
    try:
        organic_df = pd.read_parquet(organic_dir)
        quarantine_df = pd.read_parquet(quarantine_dir)
        combined = pd.concat([organic_df, quarantine_df], ignore_index=True)
        report["profiling_results"]["combined_latency"] = \
            compute_latency_stats(combined, "combined")
        report["profiling_results"]["combined_throughput"] = \
            compute_throughput(combined, "combined")
    except Exception:
        pass

    # Write report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nLatency report saved: {output_path}")

    # Print summary
    for key, stats in report["profiling_results"].items():
        if "error" not in stats:
            print(f"\n  {key}:")
            for k, v in stats.items():
                print(f"    {k}: {v}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Profile end-to-end pipeline latency')
    parser.add_argument('--organic-dir', required=True,
                        help='Path to organic reviews Parquet')
    parser.add_argument('--quarantine-dir', required=True,
                        help='Path to quarantined reviews Parquet')
    parser.add_argument('--output', default='latency_report.json',
                        help='Output path for latency report')
    args = parser.parse_args()

    profile(args.organic_dir, args.quarantine_dir, args.output)
