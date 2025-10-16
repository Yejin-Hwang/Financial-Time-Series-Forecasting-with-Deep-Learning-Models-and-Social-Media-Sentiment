
alter table tsla_price add daily_sentiment float
alter table tsla_price add daily_sentiment float

update tsla_price p
set daily_sentiment = d.daily_sentiment 
from tsla_daily_sentiment d 
where p.date = d.date ; 

CREATE TABLE tsla_price_sentiment AS
SELECT *
FROM tsla_price
WHERE daily_sentiment IS NOT NULL;

select * from  tsla_price_sentiment ;


alter table tsla_price_sentiment add column post_count int
alter table tsla_price_sentiment add column spike_presence INT CHECK (spike_presence IN (0, 1))
alter table tsla_price_sentiment add column spike_intensity float

update tsla_price_sentiment p
set post_count = s.post_count, spike_presence = s.spike_presence, spike_intensity = s.spike_intensity 
from tsla_spike s 
where p.date = s.impact_trading_day

select * from tsla_price_sentiment order by date ;

UPDATE tsla_price_sentiment
SET
    post_count = COALESCE(post_count, 0),
    spike_presence = COALESCE(spike_presence, 0),
    spike_intensity = COALESCE(spike_intensity, 0);

UPDATE tsla_price_sentiment p
SET time_idx = new_idx
FROM (
    SELECT
        ctid,
        ROW_NUMBER() OVER (ORDER BY date) - 1 AS new_idx
    FROM tsla_price_sentiment
) sub
WHERE p.ctid = sub.ctid;



