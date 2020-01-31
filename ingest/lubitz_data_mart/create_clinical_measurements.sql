--ECGS
-- query to obtain ECGs in loyalty cohort
select a.patient_num, b.encounter_num, b.concept_cd as ecg_concept, b.start_date as ecg_date
into datamart_schema.sk_ehr_cohort_ecgs
from datamart_schema.sk_ehr_cohort as a
inner join Observation_Fact as b
on a.patient_num=b.patient_num
where b.concept_cd in ('C93000', 'C93005', 'C93010')
order by a.patient_num, b.encounter_num

-- summary stats
--- number of ECGs total
select count(*) from datamart_schema.sk_ehr_cohort_ecgs
--- number of individuals with an ECG
select count(distinct(patient_num)) from datamart_schema.sk_ehr_cohort_ecgs

--- ECGs per individual with an ECG
select count(distinct(encounter_num)) as ecgs_per_individual
into #temp1
from datamart_schema.sk_ehr_cohort_ecgs
group by patient_num;
select avg(ecgs_per_individual) as avg_ecgs_per_individual, stdev(ecgs_per_individual) as std_dev from #temp1

--ECHOS
-- query to obtain echos in loyalty cohort
select a.patient_num, b.encounter_num, b.concept_cd as tte_concept, b.start_date as tte_date
into datamart_schema.sk_ehr_cohort_echos
from datamart_schema.sk_ehr_cohort as a
inner join Observation_Fact as b
on a.patient_num=b.patient_num
where b.concept_cd in ('C93303', 'C93304', 'C93306','C93307','C93308')
order by a.patient_num, b.encounter_num asc
-- summary stats
--- number of echos total
select count(*) from datamart_schema.sk_ehr_cohort_echos
--- number of individuals with a TTE
select count(distinct(patient_num)) from datamart_schema.sk_ehr_cohort_echos
--- TTEs per individual with a TTE
select count(distinct(encounter_num)) as ttes_per_individual
into #temp1
from datamart_schema.sk_ehr_cohort_echos
group by patient_num;
select avg(ttes_per_individual) as avg_ttes_per_individual, stdev(ttes_per_individual) as std_dev from #temp1


