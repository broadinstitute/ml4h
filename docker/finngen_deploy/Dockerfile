FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and model files
COPY process_files.py .
COPY ecg2age_v2025_09_11.keras .
COPY longitudinal_ecg2age_performerage.pt .
COPY Tranformer.py .

#CMD ["python", "process_files.py", "--path", "/data", "--metadata_path", "/metadata"]
ENTRYPOINT ["python", "process_files.py"]

CMD ["--path", "/data"]

