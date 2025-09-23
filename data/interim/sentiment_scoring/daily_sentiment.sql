select * from tesla_sentiment limit 20;

select date,sentiment,sentiment_score,new_score from tesla_sentiment;

alter table tesla_sentiment Add column new_score float;

update tesla_sentiment
set new_score = 
    case when sentiment = 'positive' then 1
         when sentiment = 'negative' then -1
         else sentiment_score
    end;

select date, avg(new_score) as daily_sentiment from tesla_sentiment group by date order by date desc;