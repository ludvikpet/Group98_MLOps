FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/src/cleaninbox
RUN mkdir -p /app/reports
WORKDIR /app


WORKDIR /app

COPY requirements_frontend.txt /app/src/cleaninbox/api/requirements_frontend.txt
COPY src/cleaninbox/api/multipagefrontend.py /app/src/cleaninbox/api/multipagefrontend.py
COPY src/cleaninbox/api/welcome.py /app/src/cleaninbox/api/welcome.py
COPY src/cleaninbox/api/monitoring/data_drifting.py /app/src/cleaninbox/api/monitoring/data_drifting.py
COPY src/cleaninbox/api/monitoring/user_statistics.py /app/src/cleaninbox/api/monitoring/user_statistics.py
COPY src/cleaninbox/api/monitoring/metrics.py /app/src/cleaninbox/api/monitoring/metrics.py
COPY src/cleaninbox/api/prediction/predict.py /app/src/cleaninbox/api/prediction/predict.py
COPY configs/ /app/configs/


RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt
RUN pip install -e . --no-cache-dir --verbose
#EXPOSE $PORT
#try and hardcode to see if this solves it 
EXPOSE 8080 

ENTRYPOINT ["streamlit", "run", "src/cleaninbox/api/multipagefrontend.py", "--server.port=8080", "--server.address=0.0.0.0"]
