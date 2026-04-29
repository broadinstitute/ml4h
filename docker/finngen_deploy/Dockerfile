#FROM ghcr.io/broadinstitute/ml4h:tf2.19-latest-gpu
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY encoder_ecg_4096_std.keras /app/encoder_ecg_4096_std.keras
COPY transformer_ecg_only_v2026_03_04.keras /app/transformer_ecg_only_v2026_03_04.keras
COPY transformer_ecg_only_512d_v2026_04_01.keras /app/transformer_ecg_only_512d_v2026_04_01.keras
COPY encoder_ecg_5000_std.keras /app/encoder_ecg_5000_std.keras
COPY ecgs2age_v2026_03_31.keras /app/ecgs2age_v2026_03_31.keras
COPY ecgs2age_v2026_04_27.keras /app/ecgs2age_v2026_04_27.keras

COPY process_files_parameterized.py .

ENTRYPOINT ["python", "process_files_parameterized.py"]
