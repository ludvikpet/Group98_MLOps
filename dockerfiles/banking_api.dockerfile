FROM python:3.11-slim

RUN mkdir -p src/cleaninbox
WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# Add required source files:
COPY src/cleaninbox/api/api_backend.py src/cleaninbox/api_backend.py
# COPY src/cleaninbox/model.py src/cleaninbox/model.py
# COPY src/cleaninbox/data.py src/cleaninbox/data.py
# COPY src/cleaninbox/evaluate.py src/cleaninbox/evaluate.py
# COPY src/cleaninbox/prediction.py src/cleaninbox/prediction.py

# Add everything else:
COPY requirements_backend.txt requirements_backend.txt
COPY pyproject.toml pyproject.toml
COPY configs configs

# Install dependencies:
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_backend.txt

# Only used for local fast docker build:
RUN pip install torch~=2.5.1 -i https://download.pytorch.org/whl/cpu
 
# Build the package:
RUN pip install . --no-cache-dir --verbose

EXPOSE 8080
CMD exec uvicorn --port 8080 --host 0.0.0.0 cleaninbox.api_backend:app