--This is a script to add a new pheno to existing EHR cohort table
--Relies on existing table where the pheno's concepts are stored
--You can just replace all instances of 'phenotype' with the name of your new field and the query will run
--PLEASE DO NOT RUN THE ENTIRE SCRIPT AT ONCE - SEE NOTE BELOW RE: CHECKING YOUR OUTPUT BEFORE YOU PERFORM THE FINAL JOIN

-------------------------------------------------------Pheno
--gather all diagnosis codes for the new pheno
select *
into #temp_phenotype
from Observation_Fact
where start_date >= '20010101'
and concept_cd in (
	select CONCEPT_CD
	from --insert phenotype table here--
)

--gather all unique individuals with a diagnosis code for the new pheno
select patient_num,MIN(start_date) AS first_phenotype
into #temp_phenotype2
from #temp_phenotype
group by patient_num

--join pheno dates
select a.*,first_phenotype, 
CASE WHEN ((first_phenotype is not null) AND (datediff(day,start_fu,first_phenotype) > 0)) THEN 1 ELSE 0 END as incd_phenotype,
CASE WHEN ((first_phenotype is not null) AND (datediff(day,start_fu,first_phenotype) <= 0)) THEN 1 ELSE 0 END as prev_phenotype,
datediff(day,start_fu,first_phenotype) as time_to_phenotype
into #temp_output
from datamart_schema.sk_ehr_cohort as a
left join #temp_phenotype2 as b
ON a.patient_num = b.patient_num;

-- PLEASE CHECK TEMP_OUTPUT TO MAKE SURE YOU ARE HAPPY FIRST --
-- IT SHOULD CONTAIN EVERYTHING WE HAD BEFORE + YOUR NEW FIELD --
select top 500 * from #temp_output
-- BECAUSE WE WILL NOW DELETE AND REPLACE EXISTING EHR_COHORT TABLE (SQL PAIN) --

--save new ehr_cohort table 
drop table datamart_schema.sk_ehr_cohort
select * 
into datamart_schema.sk_ehr_cohort
from #temp_output 

