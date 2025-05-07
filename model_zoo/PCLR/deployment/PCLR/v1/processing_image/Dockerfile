FROM python:3.9-slim
WORKDIR /app
COPY prepare.py /app/
COPY finalize.py /app/
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
ENTRYPOINT ["python"]