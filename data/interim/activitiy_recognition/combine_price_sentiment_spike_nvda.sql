alter table nvda_price add daily_sentiment float

UPDATE nvda_price p
SET daily_sentiment = d.daily_sentiment
FROM nvda_daily_sentiment d
WHERE p.date = d.date;

-- CREATE TABLE nvda_price_sentiment AS
-- SELECT *
-- FROM nvda_price
-- WHERE daily_sentiment IS NOT NULL;

select * from  nvda_price_sentiment ;

-- CREATE TABLE nvda_spike(
--     impact_trading_day DATE PRIMARY KEY,
--     post_count INTEGER NOT NULL,
--     smoothed DOUBLE PRECISION,
--     spike_presence SMALLINT NOT NULL CHECK (spike_presence IN (0, 1)),
--     spike_intensity DOUBLE PRECISION,
--     loess_upper DOUBLE PRECISION
-- );

select * from nvda_spike limit 20 ; 
select * from nvda_price_sentiment limit 20 ; 

alter table nvda_price_sentiment add column post_count int
alter table nvda_price_sentiment add column spike_presence INT CHECK (spike_presence IN (0, 1))
alter table nvda_price_sentiment add column spike_intensity float

update nvda_price_sentiment p
set post_count = s.post_count, spike_presence = s.spike_presence, spike_intensity = s.spike_intensity 
from nvda_spike s 
where p.date = s.impact_trading_day

select * from nvda_price_sentiment order by date ;

UPDATE nvda_price_sentiment
SET
    post_count = COALESCE(post_count, 0),
    spike_presence = COALESCE(spike_presence, 0),
    spike_intensity = COALESCE(spike_intensity, 0);

-- time index alignment 
UPDATE nvda_price_sentiment p
SET time_idx = new_idx
FROM (
    SELECT
        ctid,
        ROW_NUMBER() OVER (ORDER BY date) - 1 AS new_idx
    FROM nvda_price_sentiment
) sub
WHERE p.ctid = sub.ctid;


