FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY encoder_ecg_4096_std.keras /app/encoder_ecg_4096_std.keras
COPY transformer_ecg_only_v2026_02_10.keras /app/transformer_ecg_only_v2026_02_10.keras

COPY process_files_parameterized.py .

ENTRYPOINT ["python", "process_files_parameterized.py"]
