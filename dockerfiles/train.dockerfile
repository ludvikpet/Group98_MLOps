# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY configs/ configs/
COPY data/ data/
COPY models /models
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

#RUN pip install torch~=2.5.1 -i https://download.pytorch.org/whl/cpu #only used for local fast docker build
RUN pip install google-cloud-storage google-cloud-aiplatform
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/cleaninbox/train.py"]
