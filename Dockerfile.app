# Fabric QC Dashboard — Dockerfile
# Lightweight: this container only renders UI and calls the api
# service over HTTP. No model weight, no ultralytics/torch/opencv —
# all inference lives in the api container (single source of truth).
FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8501

# headless=true: required inside Docker, otherwise Streamlit waits
# for an interactive welcome-screen input on first run and the
# container appears to hang.
CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
