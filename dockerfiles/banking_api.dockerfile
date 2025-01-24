FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/src/cleaninbox
RUN mkdir -p /app/reports
WORKDIR /app

# Add required source files:
COPY src/cleaninbox/api/api_backend.py /app/src/cleaninbox/api/api_backend.py
COPY src/cleaninbox/model.py /app/src/cleaninbox/model.py
COPY src/cleaninbox/data.py /app/src/cleaninbox/data.py
COPY src/cleaninbox/evaluate.py /app/src/cleaninbox/evaluate.py
COPY src/cleaninbox/prediction.py /app/src/cleaninbox/prediction.py

# Add everything else:
COPY requirements_backend.txt /app/requirements_backend.txt
COPY pyproject.toml /app/pyproject.toml
COPY configs/ /app/configs/

# Install dependencies:
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements_backend.txt

# Only used for local fast docker build:
RUN --mount=type=cache,target=/root/.cache/pip pip install torch~=2.5.1 -i https://download.pytorch.org/whl/cpu

# Build the package:
RUN pip install -e . --no-cache-dir --verbose

EXPOSE 8080
CMD exec uvicorn --port 8080 --host 0.0.0.0 cleaninbox.api.api_backend:app
