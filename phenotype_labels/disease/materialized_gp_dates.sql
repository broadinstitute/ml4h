WITH read_2_only AS (
  SELECT gpc.eid sample_id, 41202 FieldID, cv2.icd10_code value, gpc.event_dt vdate,
  FROM `ukbb-analyses.ukbb7089_202109.gp_clinical` gpc
  INNER JOIN `ukbb-analyses.ukbb7089_202109.map_read_v2_icd10` cv2 on gpc.read_2 = cv2.read_code
  WHERE gpc.read_2 is not null and cv2.icd10_code_def=1
    and cv2.icd10_code not like '%-%' and cv2.icd10_code not like '%,%'
    and cv2.icd10_code not like '%+%' and cv2.icd10_code not like '% %'
    and cv2.icd10_code not like '%X' 
), result_read_2 AS (
  SELECT sample_id, FieldID, value, MIN(vdate) first_date from read_2_only
  GROUP BY sample_id, FieldID, value
), read_3_only AS (
  SELECT gpc.eid sample_id, 41202 FieldID, cv3.icd10_code value, gpc.event_dt vdate,
  FROM `ukbb-analyses.ukbb7089_202109.gp_clinical` gpc
  INNER JOIN `ukbb-analyses.ukbb7089_202109.map_read_v3_icd10` cv3 on gpc.read_3 = cv3.read_code
  WHERE gpc.read_3 is not null
    and cv3.icd10_code not like '%-%' and cv3.icd10_code not like '%,%'
    and cv3.icd10_code not like '%+%' and cv3.icd10_code not like '% %'
    and cv3.icd10_code not like '%X' and  cv3.icd10_code not like '%D'
    and cv3.icd10_code not like '%A'
  and ((cv3.mapping_status='E' and cv3.refine_flag != 'M')  or (cv3.mapping_status='D' and cv3.refine_flag in ('C','P') and cv3.add_code_flag in ('C', 'P', 'M')))
), result_read_3 AS (
  SELECT sample_id, FieldID, value, MIN(vdate) first_date from read_3_only
  GROUP BY sample_id, FieldID, value
)

SELECT all_results.sample_id, all_results.FieldID, all_results.value, MIN(all_results.first_date) first_date
FROM (
  SELECT * FROM result_read_2
  UNION DISTINCT
  SELECT * FROM result_read_3
) all_results
GROUP BY all_results.sample_id, all_results.FieldID, all_results.value
