FROM python:3.11-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

    
COPY src/cleaninbox/api src/
COPY requirements_backend.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY configs configs

RUN pip install -r requirements.txt --no-cache-dir --verbose 

# Only used for local fast docker build:
RUN pip install torch~=2.5.1 -i https://download.pytorch.org/whl/cpu

RUN pip install . --no-deps --no-cache-dir --verbose

RUN --mount=type=cache, target=/root/.cache/pip pip install . --no-deps --no-cache-dir --verbose

# CMD exec uvicorn api_backend:app --port $PORT --host 0.0.0.0 --workers 1
ENTRYPOINT ["streamlit", "src/cleaninbox/api/api_backend:app", "api_backend.py", "--server.port=$PORT", "--server.address=0.0.0.0"]