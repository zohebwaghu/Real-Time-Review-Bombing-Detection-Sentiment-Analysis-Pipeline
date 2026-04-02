-- ═══════════════════════════════════════════════════════════════
-- Athena Evaluation Queries — Review Bombing Detection Pipeline
-- ═══════════════════════════════════════════════════════════════

-- ── TABLE DEFINITIONS ──

CREATE EXTERNAL TABLE IF NOT EXISTS organic_reviews (
    review_id STRING,
    business_id STRING,
    user_id STRING,
    stars INT,
    text STRING,
    sentiment_label DOUBLE,
    event_time TIMESTAMP,
    ingestion_time TIMESTAMP
)
STORED AS PARQUET
LOCATION 's3://your-bucket/organic-reviews/'
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');


CREATE EXTERNAL TABLE IF NOT EXISTS quarantined_reviews (
    review_id STRING,
    business_id STRING,
    user_id STRING,
    stars INT,
    text STRING,
    sentiment_label DOUBLE,
    anomaly_reason STRING,
    review_count INT,
    avg_stars DOUBLE,
    is_injected STRING,
    event_time TIMESTAMP,
    ingestion_time TIMESTAMP
)
STORED AS PARQUET
LOCATION 's3://your-bucket/quarantined-reviews/'
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');


-- ═══════════════════════════════════════════════════════════════
-- 1. PRECISION: Of reviews flagged as quarantined, how many
--    were actually injected?
-- ═══════════════════════════════════════════════════════════════

SELECT
    COUNT(CASE WHEN is_injected = 'true' THEN 1 END) AS true_positives,
    COUNT(CASE WHEN is_injected IS NULL
               OR is_injected != 'true' THEN 1 END) AS false_positives,
    COUNT(*) AS total_quarantined,
    ROUND(
        CAST(COUNT(CASE WHEN is_injected = 'true' THEN 1 END) AS DOUBLE)
        / NULLIF(COUNT(*), 0), 4
    ) AS precision
FROM quarantined_reviews;


-- ═══════════════════════════════════════════════════════════════
-- 2. RECALL: Of all injected reviews, how many were caught?
-- ═══════════════════════════════════════════════════════════════

WITH all_injected AS (
    SELECT COUNT(*) AS cnt
    FROM organic_reviews WHERE is_injected = 'true'
    UNION ALL
    SELECT COUNT(*) AS cnt
    FROM quarantined_reviews WHERE is_injected = 'true'
),
caught AS (
    SELECT COUNT(*) AS caught_count
    FROM quarantined_reviews WHERE is_injected = 'true'
)
SELECT
    c.caught_count,
    (SELECT SUM(cnt) FROM all_injected) AS total_injected,
    ROUND(
        CAST(c.caught_count AS DOUBLE)
        / NULLIF((SELECT SUM(cnt) FROM all_injected), 0), 4
    ) AS recall
FROM caught c;


-- ═══════════════════════════════════════════════════════════════
-- 3. F1 SCORE (combined precision + recall)
-- ═══════════════════════════════════════════════════════════════

WITH metrics AS (
    SELECT
        -- Precision
        CAST(COUNT(CASE WHEN q.is_injected = 'true' THEN 1 END) AS DOUBLE)
            / NULLIF(COUNT(*), 0) AS precision,
        -- Recall numerator
        CAST(COUNT(CASE WHEN q.is_injected = 'true' THEN 1 END) AS DOUBLE)
            AS tp
    FROM quarantined_reviews q
),
total_injected AS (
    SELECT
        (SELECT COUNT(*) FROM organic_reviews
         WHERE is_injected = 'true')
      + (SELECT COUNT(*) FROM quarantined_reviews
         WHERE is_injected = 'true') AS total
),
scores AS (
    SELECT
        m.precision,
        m.tp / NULLIF(t.total, 0) AS recall
    FROM metrics m, total_injected t
)
SELECT
    ROUND(precision, 4) AS precision,
    ROUND(recall, 4) AS recall,
    ROUND(
        2.0 * precision * recall / NULLIF(precision + recall, 0), 4
    ) AS f1_score
FROM scores;


-- ═══════════════════════════════════════════════════════════════
-- 4. SENTIMENT ACCURACY on organic data
-- ═══════════════════════════════════════════════════════════════

SELECT
    COUNT(CASE
        WHEN (stars <= 2 AND sentiment_label = 0)
          OR (stars >= 4 AND sentiment_label = 1) THEN 1
    END) AS correct,
    COUNT(*) AS total,
    ROUND(
        CAST(COUNT(CASE
            WHEN (stars <= 2 AND sentiment_label = 0)
              OR (stars >= 4 AND sentiment_label = 1) THEN 1
        END) AS DOUBLE) / NULLIF(COUNT(*), 0), 4
    ) AS sentiment_accuracy
FROM organic_reviews
WHERE stars != 3;


-- ═══════════════════════════════════════════════════════════════
-- 5. ATTACK TIMELINE: when did the anomaly detector fire?
-- ═══════════════════════════════════════════════════════════════

SELECT
    business_id,
    anomaly_reason,
    MIN(event_time) AS attack_start,
    MAX(event_time) AS attack_end,
    COUNT(*) AS flagged_count,
    ROUND(AVG(stars), 2) AS avg_stars,
    COUNT(DISTINCT user_id) AS unique_attackers
FROM quarantined_reviews
GROUP BY business_id, anomaly_reason
ORDER BY attack_start DESC;


-- ═══════════════════════════════════════════════════════════════
-- 6. END-TO-END LATENCY: time from event to ingestion
-- ═══════════════════════════════════════════════════════════════

SELECT
    ROUND(AVG(
        EXTRACT(EPOCH FROM ingestion_time)
        - EXTRACT(EPOCH FROM event_time)
    ), 2) AS avg_latency_seconds,
    ROUND(
        APPROX_PERCENTILE(
            EXTRACT(EPOCH FROM ingestion_time)
            - EXTRACT(EPOCH FROM event_time),
            0.50
        ), 2
    ) AS p50_latency_seconds,
    ROUND(
        APPROX_PERCENTILE(
            EXTRACT(EPOCH FROM ingestion_time)
            - EXTRACT(EPOCH FROM event_time),
            0.95
        ), 2
    ) AS p95_latency_seconds,
    ROUND(
        APPROX_PERCENTILE(
            EXTRACT(EPOCH FROM ingestion_time)
            - EXTRACT(EPOCH FROM event_time),
            0.99
        ), 2
    ) AS p99_latency_seconds
FROM quarantined_reviews;


-- ═══════════════════════════════════════════════════════════════
-- 7. THROUGHPUT: reviews processed per minute
-- ═══════════════════════════════════════════════════════════════

SELECT
    DATE_TRUNC('minute', ingestion_time) AS minute_bucket,
    COUNT(*) AS reviews_per_minute
FROM (
    SELECT ingestion_time FROM organic_reviews
    UNION ALL
    SELECT ingestion_time FROM quarantined_reviews
)
GROUP BY DATE_TRUNC('minute', ingestion_time)
ORDER BY minute_bucket;


-- ═══════════════════════════════════════════════════════════════
-- 8. CONFUSION MATRIX (detailed breakdown)
-- ═══════════════════════════════════════════════════════════════

SELECT
    'quarantined' AS predicted,
    CASE WHEN is_injected = 'true' THEN 'injected'
         ELSE 'organic' END AS actual,
    COUNT(*) AS cnt
FROM quarantined_reviews
GROUP BY CASE WHEN is_injected = 'true' THEN 'injected'
              ELSE 'organic' END

UNION ALL

SELECT
    'organic' AS predicted,
    CASE WHEN is_injected = 'true' THEN 'injected'
         ELSE 'organic' END AS actual,
    COUNT(*) AS cnt
FROM organic_reviews
GROUP BY CASE WHEN is_injected = 'true' THEN 'injected'
              ELSE 'organic' END
ORDER BY predicted, actual;
