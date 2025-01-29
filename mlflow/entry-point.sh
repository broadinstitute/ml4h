#!/usr/bin/env bash

set -e

# Verify that all required variables are set
if [[ -z "${GCP_PROJECT}" ]]; then
    echo "Error: GCP_PROJECT not set"
    exit 1
fi

# Fetch secrets from Secret Manager in CGP
export MLFLOW_TRACKING_USERNAME="$(python3 /app/get_secret.py --project="${GCP_PROJECT}" --secret=mlflow_tracking_username)"
export MLFLOW_TRACKING_PASSWORD="$(python3 /app/get_secret.py --project="${GCP_PROJECT}" --secret=mlflow_tracking_password)"
export ARTIFACT_URL="$(python3 /app/get_secret.py --project="${GCP_PROJECT}" --secret=mlflow_artifact_url)"
if [[ -z "${DATABASE_URL}" ]]; then # Allow overriding for local deployment
    export DATABASE_URL="$(python3 /app/get_secret.py --project="${GCP_PROJECT}" --secret=mlflow_database_url)"
fi

# Verify that all required variables are set
if [[ -z "${MLFLOW_TRACKING_USERNAME}" ]]; then
    echo "Error: MLFLOW_TRACKING_USERNAME not set"
    exit 1
fi

if [[ -z "${MLFLOW_TRACKING_PASSWORD}" ]]; then
    echo "Error: MLFLOW_TRACKING_PASSWORD not set"
    exit 1
fi

if [[ -z "${ARTIFACT_URL}" ]]; then
    echo "Error: ARTIFACT_URL not set"
    exit 1
fi

if [[ -z "${DATABASE_URL}" ]]; then
    echo "Error: DATABASE_URL not set"
    exit 1
fi

if [[ -z "${PORT}" ]]; then
    export PORT=8080
fi

export WSGI_AUTH_CREDENTIALS="${MLFLOW_TRACKING_USERNAME}:${MLFLOW_TRACKING_PASSWORD}"
export _MLFLOW_SERVER_ARTIFACT_ROOT="${ARTIFACT_URL}"
export _MLFLOW_SERVER_FILE_STORE="${DATABASE_URL}"

echo "hello its me im the problem its me"
echo ${_MLFLOW_SERVER_FILE_STORE}
echo "or me"
echo ${_MLFLOW_SERVER_ARTIFACT_ROOT}

# Start MLflow and ngingx using supervisor
exec gunicorn -b "${HOST}:${PORT}" -w 4 --log-level debug --access-logfile=- --error-logfile=- --log-level=debug mlflow_auth:app
