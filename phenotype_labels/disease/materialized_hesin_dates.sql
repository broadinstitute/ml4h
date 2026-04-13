WITH oper4 AS (
  SELECT 41200 FieldID, oper.eid, oper4 code,
    CASE 
      WHEN oper.opdate IS NOT NULL THEN oper.opdate 
      WHEN h.admidate IS NOT NULL THEN h.admidate
      ELSE h.epistart
    END vdate
  FROM `broad-ml4cvd.ukbb7089_2024_01_20.hesin_oper` oper
  LEFT JOIN `broad-ml4cvd.ukbb7089_2024_01_20.hesin` h ON oper.eid=h.eid AND oper.ins_index = h.ins_index
  WHERE oper4 IS NOT NULL AND oper.level=1
), diag_icd10 AS (
  SELECT 41202 FieldID, hd.eid, diag_icd10 code,
    CASE 
      WHEN h.admidate IS NOT NULL THEN h.admidate
      ELSE h.epistart
    END vdate
  FROM `broad-ml4cvd.ukbb7089_2024_01_20.hesin_diag` hd
  LEFT JOIN `broad-ml4cvd.ukbb7089_2024_01_20.hesin` h ON h.eid = hd.eid and h.ins_index = hd.ins_index
  WHERE diag_icd10 IS NOT NULL and hd.level=1
), diag_icd9 AS (
  SELECT 41203 FieldID, hd.eid, diag_icd9 code,
    CASE 
      WHEN h.admidate IS NOT NULL THEN h.admidate
      ELSE h.epistart
    END vdate
  FROM `broad-ml4cvd.ukbb7089_2024_01_20.hesin_diag` hd
  LEFT JOIN `broad-ml4cvd.ukbb7089_2024_01_20.hesin` h ON h.eid = hd.eid and h.ins_index = hd.ins_index
  WHERE diag_icd9 IS NOT NULL and hd.level=1
), oper4secondary AS (
  SELECT 41210 FieldID, sec_oper.eid, sec_oper.oper4 code, 
    CASE 
      WHEN sec_oper.opdate IS NOT NULL THEN sec_oper.opdate 
      WHEN h.admidate IS NOT NULL THEN h.admidate
      ELSE h.epistart
    END vdate
  FROM `broad-ml4cvd.ukbb7089_2024_01_20.hesin_oper` sec_oper
  LEFT JOIN `broad-ml4cvd.ukbb7089_2024_01_20.hesin` h ON sec_oper.eid=h.eid AND sec_oper.ins_index = h.ins_index
  WHERE sec_oper.oper4 IS NOT NULL AND sec_oper.level=2
), diag_icd10_secondary AS (
  SELECT 41204 FieldID, h.eid, sec.diag_icd10 code, 
    CASE
      WHEN h.admidate IS NOT NULL THEN h.admidate
      ELSE h.epistart
    END vdate
  FROM `broad-ml4cvd.ukbb7089_2024_01_20.hesin_diag` sec
  LEFT JOIN `broad-ml4cvd.ukbb7089_2024_01_20.hesin` h ON sec.eid=h.eid AND sec.ins_index = h.ins_index
  WHERE sec.diag_icd10 IS NOT NULL and sec.level=2
), diag_icd9_secondary AS (
  SELECT 41205 FieldID, h.eid, sec.diag_icd9 code, 
    CASE 
      WHEN h.admidate IS NOT NULL THEN h.admidate
      ELSE h.epistart
    END vdate
  FROM `broad-ml4cvd.ukbb7089_2024_01_20.hesin_diag` sec
  LEFT JOIN `broad-ml4cvd.ukbb7089_2024_01_20.hesin` h ON sec.eid=h.eid AND sec.ins_index = h.ins_index
  WHERE sec.diag_icd9 IS NOT NULL and sec.level=2
)

SELECT 
  diagnostics.eid sample_id, diagnostics.FieldID, diagnostics.code value, 
  CASE
    WHEN MIN(PARSE_DATE("%d/%m/%E4Y", vdate)) IS NULL THEN MIN(PARSE_DATE("%E4Y-%m-%d", p.value))
    ELSE MIN(PARSE_DATE("%d/%m/%E4Y", vdate))
  END first_date
FROM (
  SELECT * FROM oper4
  UNION DISTINCT
  SELECT * FROM diag_icd10
  UNION DISTINCT
  SELECT * FROM diag_icd9
  UNION DISTINCT
  SELECT * FROM oper4secondary
  UNION DISTINCT
  SELECT * FROM diag_icd10_secondary
  UNION DISTINCT
  SELECT * FROM diag_icd9_secondary
) diagnostics
JOIN `broad-ml4cvd.ukbb7089_2024_01_20.phenotype` p ON p.sample_id = diagnostics.eid AND p.array_idx=0 AND p.instance=0 AND p.FieldID=53
GROUP BY diagnostics.eid, diagnostics.FieldID, diagnostics.code, diagnostics.vdate
ORDER BY first_date ASC

