# Exploring data



## Examples

```py
import glob
import tensormap.explain.explain

# Retrieve the path to a file
files = glob.glob('ML4CVD-muse_ecg_deidentified_1-ae78c9790e42.h5')[0]
# New Explain class
e = tensormap.explain.explain.Explain()
e.explain(files)
print(e)
```

will produce the following output:

```bash
...
/ecg/instance0/RestingECG/Waveform/SampleType-2: STRING CONTINUOUS_SAMPLES 1
/ecg/instance0/RestingECG/Waveform/WaveformStartTime: STRING 0 1
/ecg/instance0/RestingECG/Waveform/WaveformStartTime-2: STRING 0 1
/ecg/instance0/RestingECG/Waveform/WaveformType: STRING Median 1
/ecg/instance0/RestingECG/Waveform/WaveformType-2: STRING Rhythm 1
/ecg/instance0/data/WaveFormData: int16 300, 2754.0, 110308.0, -63, 95, 9.18
/ecg/instance0/data/WaveFormData-10: int16 2500, -12372.0, 5247416.0, -214, 392, -4.9488
/ecg/instance0/data/WaveFormData-11: int16 2500, -12556.0, 1453372.0, -178, 90, -5.0224
/ecg/instance0/data/WaveFormData-12: int16 2500, -15349.0, 3881205.0, -209, 56, -6.1396
/ecg/instance0/data/WaveFormData-13: int16 2500, -15912.0, 4676278.0, -219, 102, -6.3648
/ecg/instance0/data/WaveFormData-14: int16 2500, -8584.0, 3251864.0, -166, 204, -3.4336
/ecg/instance0/data/WaveFormData-15: int16 2500, 1078.0, 3428232.0, -194, 240, 0.4312
/ecg/instance0/data/WaveFormData-16: int16 2500, -2518.0, 1892204.0, -152, 270, -1.0072
/ecg/instance0/data/WaveFormData-2: int16 300, 4707.0, 225239.0, -87, 78, 15.69
/ecg/instance0/data/WaveFormData-3: int16 300, -1706.0, 59416.0, -101, 5, -5.6866666666666665
/ecg/instance0/data/WaveFormData-4: int16 300, 1754.0, 248226.0, -181, 37, 5.846666666666667
/ecg/instance0/data/WaveFormData-5: int16 300, 3174.0, 327886.0, -178, 49, 10.58
...
```

Numerical values are reported as: primitive type, number of observations, sum, sum of squares, min, max, mean. For strings, it is possible to compute statistics for tokenized strings using some character delimiter:

```py
e._tokenize_strings = True
e._tokenizer = " " # Default
```

resulting in statistics for the tokens and the full-length string

```
...
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING LOW VOLTAGE QRS, CONSIDER PULMONARY DISEASE, PERICARDIAL EFFUSION, OR NORMAL VARIANT 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING LOW 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING VOLTAGE 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING QRS, 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING CONSIDER 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING PULMONARY 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING DISEASE, 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING PERICARDIAL 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING EFFUSION, 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING OR 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING NORMAL 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING VARIANT 1
...
```

Triggering `e._tokenize_only = False` ignores full-length strings and stores the tokens only

```
...
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING LOW 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING VOLTAGE 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING QRS, 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING CONSIDER 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING PULMONARY 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING DISEASE, 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING PERICARDIAL 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING EFFUSION, 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING OR 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING NORMAL 1
/ecg/instance0/RestingECG/OriginalDiagnosis/DiagnosisStatement/StmtText-5: STRING VARIANT 1
...
```
