FROM python:3.8-slim

# displays logs imidietly to the stream, they wont be part of the buffer
ENV PYTHONUNBUFFERED True
ENV GCP_PROJECT="783282864357"

WORKDIR /app

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY get_secret.py /app/get_secret.py
COPY entry-point.sh /app/entry-point.sh
COPY mlflow_auth.py /app/mlflow_auth.py

ENTRYPOINT ["/usr/bin/env", "bash", "/app/entry-point.sh"]

EXPOSE 8080
EXPOSE 8000/tcp
CMD [ "--backend-store-uri", "/tmp"]
