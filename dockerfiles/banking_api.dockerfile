FROM python:3.11-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add required source files:
COPY src/cleaninbox/api/__init__.py src/cleaninbox/__init__.py
COPY src/cleaninbox/api/api_backend.py src/cleaninbox/api/api_backend.py
COPY src/cleaninbox/model.py src/cleaninbox/model.py
COPY src/cleaninbox/data.py src/cleaninbox/data.py
COPY src/cleaninbox/evaluate.py src/cleaninbox/evaluate.py
COPY src/cleaninbox/prediction.py src/cleaninbox/prediction.py

# Add everything else:
COPY requirements.txt requirements.txt
COPY requirements_backend.txt requirements_backend.txt
COPY pyproject.toml pyproject.toml
COPY configs configs

# Install dependencies:
RUN pip install -r requirements.txt --no-cache-dir --verbose 
RUN pip install -r requirements_backend.txt --no-cache-dir --verbose 

# Only used for local fast docker build:
RUN pip install torch~=2.5.1 -i https://download.pytorch.org/whl/cpu

# Build the package:
RUN pip install . --no-deps --no-cache-dir --verbose

# 
RUN --mount=type=cache, target=/root/.cache/pip pip install . --no-deps --no-cache-dir --verbose

# CMD exec uvicorn api_backend:app --port $PORT --host 0.0.0.0 --workers 1
ENTRYPOINT ["streamlit", "src/cleaninbox/api/api_backend:app", "api_backend.py", "--server.port=$PORT", "--server.address=0.0.0.0"]