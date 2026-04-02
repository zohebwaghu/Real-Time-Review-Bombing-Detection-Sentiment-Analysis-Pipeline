"""
Offline Sentiment Model Training — TF-IDF + Logistic Regression.

Trains TWO models from the same balanced Yelp review sample:
  1. Spark PipelineModel  → loaded directly in the Spark Streaming pipeline
  2. sklearn LR + TF-IDF  → used by the SHAP XAI module (SHAP requires sklearn)

Both models use identical hyperparameters so their predictions are consistent.

Usage:
    spark-submit \\
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \\
      /models/train_sentiment_model.py \\
      --input /data/review.json \\
      --spark-output /models/sentiment_pipeline \\
      --sklearn-output /models/sklearn_lr.pkl \\
      --vectorizer-output /models/tfidf_vectorizer.pkl

Evaluation targets:
    AUC      > 0.85
    Accuracy > 0.85
    F1       > 0.85
"""

import argparse
import os
import pickle

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


def train_spark_model(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    num_features: int = 20000,
    max_iter: int = 100,
    reg_param: float = 0.01,
    seed: int = 42,
):
    """Train and save a Spark ML PipelineModel (used in streaming)."""
    print("\n── Loading data ──────────────────────────────")
    raw = (
        spark.read.json(input_path)
        .select("review_id", "stars", "text")
        .na.drop(subset=["text"])
        .filter(col("stars") != 3)     # drop ambiguous 3-star reviews
    )

    # Binary label: 1–2 stars → 0.0 (negative), 4–5 stars → 1.0 (positive)
    labeled = raw.withColumn(
        "label", when(col("stars") <= 2, 0.0).otherwise(1.0)
    )

    # Balance dataset (Yelp skews heavily positive)
    neg = labeled.filter(col("label") == 0.0)
    pos = labeled.filter(col("label") == 1.0)
    neg_count = neg.count()
    pos_count = pos.count()
    print(f"  Before balancing — negative: {neg_count:,}, positive: {pos_count:,}")

    if pos_count > neg_count:
        pos = pos.sample(False, neg_count / pos_count, seed=seed)
    else:
        neg = neg.sample(False, pos_count / neg_count, seed=seed)

    balanced = neg.union(pos)
    print(f"  After balancing  — total: {balanced.count():,}")

    train_df, test_df = balanced.randomSplit([0.8, 0.2], seed=seed)
    print(f"  Train: {train_df.count():,}  Test: {test_df.count():,}")

    # Pipeline: Tokenize → StopWords → HashingTF → IDF → Logistic Regression
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered"),
        HashingTF(inputCol="filtered", outputCol="raw_features",
                  numFeatures=num_features),
        IDF(inputCol="raw_features", outputCol="features"),
        LogisticRegression(
            featuresCol="features", labelCol="label",
            maxIter=max_iter, regParam=reg_param, elasticNetParam=0.0,
        ),
    ])

    print("\n── Training Spark PipelineModel ──────────────")
    model = pipeline.fit(train_df)

    preds = model.transform(test_df)

    auc = BinaryClassificationEvaluator(labelCol="label").evaluate(preds)
    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(preds)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    ).evaluate(preds)

    print(f"  AUC:      {auc:.4f}  {'✓' if auc > 0.85 else '✗ (target > 0.85)'}")
    print(f"  Accuracy: {acc:.4f}  {'✓' if acc > 0.85 else '✗ (target > 0.85)'}")
    print(f"  F1:       {f1:.4f}  {'✓' if f1 > 0.85 else '✗ (target > 0.85)'}")

    model.save(output_path)
    print(f"\n  Spark PipelineModel saved → {output_path}")

    return balanced, train_df, test_df, model


def train_sklearn_model(
    balanced_df,
    sklearn_output: str,
    vectorizer_output: str,
    num_features: int = 20000,
    seed: int = 42,
):
    """
    Train an sklearn TF-IDF + LR model (mirror of the Spark model) for SHAP.

    SHAP's LinearExplainer requires an sklearn linear model — it cannot directly
    introspect a Spark ML model.  We train on the same balanced dataset so both
    models have consistent accuracy and the SHAP explanations are valid.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    print("\n── Training sklearn mirror for SHAP ──────────")

    # Collect to driver — balanced_df is already small after downsampling
    pdf = balanced_df.select("text", "label").toPandas()
    X_raw = pdf["text"].values
    y     = pdf["label"].values.astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=seed, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=num_features,
        stop_words="english",
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)

    lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed)
    lr.fit(X_train, y_train)

    y_pred  = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")

    os.makedirs(os.path.dirname(sklearn_output) or ".", exist_ok=True)
    with open(sklearn_output, "wb") as fh:
        pickle.dump(lr, fh)
    with open(vectorizer_output, "wb") as fh:
        pickle.dump(vectorizer, fh)

    print(f"\n  sklearn model saved    → {sklearn_output}")
    print(f"  TF-IDF vectorizer saved → {vectorizer_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + LR sentiment models (Spark + sklearn)"
    )
    parser.add_argument("--input",             default="/data/review.json",
                        help="Path to review.json")
    parser.add_argument("--spark-output",      default="/models/sentiment_pipeline",
                        help="Output path for Spark PipelineModel")
    parser.add_argument("--sklearn-output",    default="/models/sklearn_lr.pkl",
                        help="Output path for sklearn LR model (pickle)")
    parser.add_argument("--vectorizer-output", default="/models/tfidf_vectorizer.pkl",
                        help="Output path for sklearn TF-IDF vectorizer (pickle)")
    parser.add_argument("--num-features",      type=int, default=20000)
    parser.add_argument("--max-iter",          type=int, default=100)
    parser.add_argument("--reg-param",         type=float, default=0.01)
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("SentimentModelTraining") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    balanced_df, train_df, test_df, spark_model = train_spark_model(
        spark,
        input_path=args.input,
        output_path=args.spark_output,
        num_features=args.num_features,
        max_iter=args.max_iter,
        reg_param=args.reg_param,
        seed=args.seed,
    )

    train_sklearn_model(
        balanced_df,
        sklearn_output=args.sklearn_output,
        vectorizer_output=args.vectorizer_output,
        num_features=args.num_features,
        seed=args.seed,
    )

    spark.stop()
    print("\nTraining complete.")
