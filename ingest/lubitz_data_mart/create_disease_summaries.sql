--this script creates a summary of hf/stroke incidence, by age, with counts of pre-history and follow-up

--stroke: per patient with stroke, get dates
drop table datamart_schema.pb_summary_stroke;
select
DATEDIFF(year, birth_date, first_stroke) as age_diagnosis,
DATEDIFF(year, start_fu, first_stroke) as prehistory,
DATEDIFF(year, first_stroke, last_enc) as posthistory,
DATEDIFF(year, start_fu, last_enc) as total_history
into datamart_schema.pb_summary_stroke
from datamart_schema.sk_ehr_cohort
where first_stroke <> 'NA'
and first_stroke > start_fu;

drop table datamart_schema.pb_summary_stats_stroke;
--summary of dates per age_diagnosis
select count(*) as total_people, age_diagnosis,
avg(total_history*1.0) as avg_total_history,
avg(prehistory*1.0) as avg_prehistory,
avg(posthistory*1.0) as avg_posthistory
into datamart_schema.pb_summary_stats_stroke
from datamart_schema.pb_summary_Stroke
group by age_diagnosis
having count(*) >= 10
order by age_diagnosis asc;







--hf: per patient with hf, get dates

drop table datamart_schema.pb_summary_hf;
select
DATEDIFF(year, birth_date, first_hf) as age_diagnosis,
DATEDIFF(year, start_fu, first_hf) as prehistory,
DATEDIFF(year, first_hf, last_enc) as posthistory,
DATEDIFF(year, start_fu, last_enc) as total_history
into datamart_schema.pb_summary_hf
from datamart_schema.sk_ehr_cohort
where first_hf <> 'NA'
and first_hf > start_fu;


--summary of dates per age_diagnosis
drop table datamart_schema.pb_summary_stats_hf;

select count(*) as total_people, age_diagnosis,
avg(total_history*1.0) as avg_total_history,
avg(prehistory*1.0) as avg_prehistory,
avg(posthistory*1.0) as avg_posthistory
into datamart_schema.pb_summary_stats_hf
from datamart_schema.pb_summary_hf
group by age_diagnosis
having count(*) >= 10
order by age_diagnosis asc;


select age_diagnosis,total_people, avg_total_history, avg_prehistory, avg_posthistory 
from datamart_schema.pb_summary_stats_stroke order by age_diagnosis;
select age_diagnosis,total_people, avg_total_history, avg_prehistory, avg_posthistory
from datamart_schema.pb_summary_stats_hf order by age_diagnosis;
