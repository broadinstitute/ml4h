--This is a script to curate the preprocessed pcp cohort
-------------------------------------------------------Obtain birthdate for age
select a.*,b.birth_date as birth_date, b.sex_cd as sex, b.death_date as death_date
into #temp1
from datamart_schema.sk_pcp_13 as a
inner join Patient_Dimension as b on
a.patient_num  = b.patient_num;

--last encounter
select patient_num,MAX(start_date) AS last_enc
into #temp2
from Observation_Fact
where patient_num in (
	SELECT patient_num
	from #temp1)
group by patient_num;

select #temp1.*,last_enc
into #temp3
from #temp2
inner join #temp1 on
#temp2.patient_num = #temp1.patient_num

-------------------------------------------------------AF  
--gather all diagnosis codes for AF
select *
into #temp_af
from Observation_Fact
where start_date >= '20010101'
and concept_cd in (
	select CONCEPT_CD --18 codes in total
	from Concept_Dimension
	where NAME_CHAR like '%atrial fib%'
	OR name_char like '%atrial flu%'
)

--gather all unique individuals with a diagnosis code for AF
select patient_num,MIN(start_date) AS first_af
into #temp_af2
from #temp_af
group by patient_num

-------------------------------------------------------Stroke
--gather all diagnosis codes for stroke
select *
into #temp_stroke
from Observation_Fact
where start_date >= '20010101'
and concept_cd in (
	select CONCEPT_CD --15 codes in total
	from Concept_Dimension
	where (NAME_CHAR like '%stroke%'
	OR concept_cd in ('36231','36232','36233','36234',--eye strokes
	'38802',--transient ischemic deafness
	'43301','43311','43321','43331','43381','43391', --precerebral arterial disease w infarct
	'434','4340','43400','43401','4341','43410','43411','4349','43490','43491',--cerebral thrombosis
	'435','4350','4351','4352','4353','4358','4359',--transient cerebral ischemia
	'4371','4377','4379',--ischemic cerebrovascular disease
	'438','4380','4381','43810','43811','43812','43813','43819','4382','43820','43821',
	'43822','4383','43830','43831','43832','4384','43840','43841','43842','4385','43850',
	'43851','43852','43853','4388','43881','43882','43883','43884','43885','43889','4389',--late effects of cerebrovascular disease
	'99702','V1254',--personal history of TIA
	'ICD10:G45','ICD10:G45.0','ICD10:G45.1','ICD10:G45.2','ICD10:G45.3','ICD10:G45.4',
	'ICD10:G45.8','ICD10:G45.9',--TIA syndromes
	'ICD10:G46','ICD10:G46.0','ICD10:G46.1','ICD10:G46.2','ICD10:G46.3','ICD10:G46.4',
	'ICD10:G46.5','ICD10:G46.6','ICD10:G46.7','ICD10:G46.8',--Lacunar syndromes
	'ICD10:H34','ICD10:H34.0','ICD10:H34.00','ICD10:H34.01','ICD10:H34.02','ICD10:H34.03',
	'ICD10:H34.1','ICD10:H34.10','ICD10:H34.11','ICD10:H34.12','ICD10:H34.13',
	'ICD10:H34.2','ICD10:H34.21','ICD10:H34.211','ICD10:H34.212','ICD10:H34.213','ICD10:H34.219',
	'ICD10:H34.23','ICD10:H34.231','ICD10:H34.232','ICD10:H34.233','ICD10:H34.239',--retinal occlusions
	'ICD10:H93.01','ICD10:H93.011','ICD10:H93.012','ICD10:H93.013','ICD10:H93.019',--transient ischemic deafness
	'ICD10:I63%',--cerebral infarction
	'ICD10:I66%',--occlusion/stenosis of cerebral arteries
	'ICD10:I67.81','ICD10:I67.82','ICD10:I67',--other cerebrovascular disease
	'ICD10:I69.3%','ICD10.I69.8%','ICD10.I69.9%',--sequelae of cerebrovascular disease
	'ICD10:I97.81','ICD10:I97.810','ICD10:I97.811','ICD10:I97.82','ICD10:I97.820','ICD10:I97.821',--procedural stroke
	'ICD10:Z86.73',--personal history of TIA
	'BRRADV2413','C37195','C61645','EPRC:76001137'))--cerebral thrombolysis
	AND concept_cd not in ('9920','MX257')
	AND concept_cd not like 'EDEPT%'
	AND name_char not like '%sunstroke%'
	AND name_char not like '%family history%');

--gather all unique individuals with a diagnosis code for stroke
select patient_num,MIN(start_date) AS first_stroke
into #temp_stroke2
from #temp_stroke
group by patient_num

-------------------------------------------------------HTN
--gather all diagnosis codes for HTN
select *
into #temp_htn
from Observation_Fact
where start_date >= '20010101'
and concept_cd in (
	select concept_cd
	from Concept_Dimension
	where ((NAME_CHAR like '%hypertension%')
	OR (concept_cd in ('4372','7962','36211')) --hypertensive encephalopathy, elevated BP, hypertensive retinopathy
	OR (concept_cd like '402%') --hypertensive heart disease
	OR (concept_cd like '403%') --hypertensive renal disease
	OR (concept_cd like '404%') --hypertensive heart and renal disease
	OR (concept_cd like 'ICD10:I11%') --hypertensive heart disease
	OR (concept_cd like 'ICD10:I12%') --hypertensive renal disease
	OR (concept_cd like 'ICD10:I13%') --hypertensive heart and renal disease
	OR (concept_cd like 'ICD10:I15%')) --hypertensive heart and renal disease
	AND (name_char not like '%venous hypertension%')
	AND (name_char not like '%pulmonary hypertension%')
	AND (name_char not like '%portal hypertension%')
	AND (name_char not like '%intracranial hypertension%')
	AND (name_char not like '%ocular hypertension%')
	AND (concept_cd not in ('6461','6462','V811'))); --'without mention of hypertension','screening for hypertension'

--gather all unique individuals with a diagnosis code for HTN
select patient_num,MIN(start_date) AS first_htn
into #temp_htn2
from #temp_htn
group by patient_num

-------------------------------------------------------DM
--gather all diagnosis codes for DM
select *
into #temp_dm
from Observation_Fact
where start_date >= '20010101'
and concept_cd in (
	select concept_cd
	from Concept_Dimension
	where ((NAME_CHAR like '%diabetes%')
	OR (concept_cd like '249%')
	OR (concept_cd in ('3620','36201','36202','36203','36204','36205','36206','36207','36641','7916','R824'))
	OR (concept_cd like ('ICD10:E08%')) --diabetes mellitus
	OR (concept_cd like ('ICD10:E09%')) --drug or chemical induced diabetes
	OR (concept_cd like ('ICD10:E10%')) --type 1 diabetes
	OR (concept_cd like ('ICD10:E11%')) --type 2 diabetes
	OR (concept_cd like ('ICD10:E13%'))) --other diabetes
	AND (name_char not like '%insipidus%')
	AND (concept_cd not in ('7916','C81403','C81404','C99500','EPRC:31000682','ERX:128248','ICD10:P70.0','ICD10:R73.03',
	'ICD10:Z13.1','ICD10:Z83.3','LMA7627','MCSQ-DHBA1C','MCSQ-DIACOM','MCSQ-DIAMOD','NHINSDD','ODA:ELAC2','V180','V771'))) --acetonuria, family history, testing, etc

--gather all unique individuals with a diagnosis code for DM
select patient_num,MIN(start_date) AS first_dm
into #temp_dm2
from #temp_dm
group by patient_num

-------------------------------------------------------HF
--gather all diagnosis codes for HF
select *
into #temp_hf
from Observation_Fact
where start_date >= '20010101'
and concept_cd in (
	select concept_cd
	from Concept_Dimension
	where (((name_char like '%heart failure%')
	OR concept_cd in ('ICD10:I50.1'))
	AND (concept_cd not in ('40200','40210','40290','40400','40410','40490','APRDRG:170',
	'APRDRG:171','BRRADV984','DRG:115','ICD10:I11.9','ICD10:I13.1','ICD10:I13.10','ICD10:I13.11'))));

--gather all unique individuals with a diagnosis code for HF
select patient_num,MIN(start_date) AS first_hf
into #temp_hf2
from #temp_hf
group by patient_num

--join phenotype dates
select a.*,first_af,first_stroke,first_htn,first_dm,first_hf
into #temp4
from #temp3 as a
left join #temp_stroke2 as b --stroke
ON a.patient_num = b.patient_num
left join #temp_af2 as c --AF
ON a.patient_num = c.patient_num
left join #temp_htn2 as d --HTN
ON a.patient_num = d.patient_num
left join #temp_dm2 as e --DM
ON a.patient_num = e.patient_num
left join #temp_hf2 as f --HF
ON a.patient_num = f.patient_num;

--save output as table
drop table datamart_schema.sk_pcp_13;
 
select *
into datamart_schema.sk_pcp_13
from #temp4;

select distinct(patient_num) 
into #temp1
from datamart_schema.sk_enc_ov
