FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy script and bundled models
COPY process_files_parameterized.py .

ENTRYPOINT ["python", "process_files_parameterized.py"]