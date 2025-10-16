-- (optional) quick peek
SELECT * FROM nvda_price
ORDER BY date
LIMIT 5;

-- Build aligned view in one statement
WITH trading_days AS (
  SELECT DISTINCT date::date AS dt
  FROM nvda_price
),
sentiment_aligned AS (
  SELECT
    td.dt AS date,
    COALESCE(d.daily_sentiment, 0.0)::double precision AS daily_sentiment
  FROM trading_days td
  LEFT JOIN nvda_daily_sentiment d
    ON d.date::date = td.dt           -- ensure type match
),
spike_aligned AS (
  SELECT
    td.dt AS date,
    COALESCE(s.post_count, 0)::int                AS post_count,
    COALESCE(s.spike_presence, 0)::int            AS spike_presence,
    COALESCE(s.spike_intensity, 0.0)::double precision AS spike_intensity
  FROM trading_days td
  LEFT JOIN nvda_spike s
    ON s.impact_trading_day::date = td.dt         -- ensure type match
)
SELECT
  td.dt                                                   AS date,
  p.close,                                               -- or adj_close if needed
  sa.daily_sentiment,
  sp.post_count,
  sp.spike_presence,
  sp.spike_intensity
FROM trading_days td
LEFT JOIN nvda_price p      ON p.date::date = td.dt       -- avoid USING; cast to date
LEFT JOIN sentiment_aligned sa ON sa.date = td.dt
LEFT JOIN spike_aligned sp     ON sp.date = td.dt
ORDER BY td.dt;


select * from nvda_combined_range


CREATE TABLE IF NOT EXISTS nvda_combined_range AS
WITH bounds AS (
  SELECT DATE '2024-06-03' AS dmin, DATE '2025-07-21' AS dmax
),
base AS (
  SELECT
    f.date::date                           AS date,
    /* drop: open/high/low/return_1d/sentiment_full */
    f."Close"::double precision            AS close,
    f."Volume"::bigint                     AS volume,
    f.last_earnings_date::timestamptz      AS last_earnings_date,
    f.days_since_earning::double precision AS days_since_earning,
    f.month::int                           AS month,
    f.day_of_week::int                     AS day_of_week,
    f.quarter::int                         AS quarter,
    f.year::int                            AS year,
    f.is_month_end::smallint               AS is_month_end,
    f.is_month_start::smallint             AS is_month_start,
    f.rolling_volatility::double precision AS rolling_volatility,
    f.cumulative_return::double precision  AS cumulative_return,
    f.time_idx::int                        AS time_idx,
    f.nvda_close::double precision         AS nvda_close,
    f.unique_id                            AS unique_id
  FROM NVDA_full_features f
  JOIN bounds b
    ON f.date::date BETWEEN b.dmin AND b.dmax
),
sentiment_aligned AS (
  SELECT b.date, COALESCE(d.daily_sentiment::double precision, 0.0) AS daily_sentiment
  FROM base b
  LEFT JOIN nvda_daily_sentiment d ON d.date::date = b.date
),
spike_aligned AS (
  SELECT b.date,
         COALESCE(s.post_count::int, 0)                     AS post_count,
         COALESCE(s.spike_presence::int, 0)                 AS spike_presence,
         COALESCE(s.spike_intensity::double precision, 0.0) AS spike_intensity
  FROM base b
  LEFT JOIN nvda_spike s ON s.impact_trading_day::date = b.date
)
SELECT
  b.*,
  sa.daily_sentiment,
  sp.post_count,
  sp.spike_presence,
  sp.spike_intensity
FROM base b
LEFT JOIN sentiment_aligned sa ON sa.date = b.date
LEFT JOIN spike_aligned     sp ON sp.date = b.date
ORDER BY b.date;

WITH ordered AS (
  SELECT
    ctid,  -- physical row id
    ROW_NUMBER() OVER (ORDER BY date) - 1 AS new_idx
  FROM nvda_combined_range
)
UPDATE nvda_combined_range t
SET time_idx = o.new_idx
FROM ordered o
WHERE t.ctid = o.ctid;


