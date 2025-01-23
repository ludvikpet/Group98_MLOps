FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_frontend.txt /app/requirements_frontend.txt
COPY src/cleaninbox/api/multipagefrontend.py /app/multipagefrontend.py
COPY src/cleaninbox/api/welcome.py /app/welcome.py
COPY src/cleaninbox/api/monitoring/data_drifting.py /app/monitoring/data_drifting.py
COPY src/cleaninbox/api/monitoring/user_statistics.py /app/monitoring/user_statistics.py

#COPY src/cleaninbox/api/monitoring/reports/report.html /app/monitoring/reports/report.html
COPY src/cleaninbox/api/prediction/predict.py /app/prediction/predict.py


RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

#EXPOSE $PORT
#try and hardcode to see if this solves it 
EXPOSE 8080 

ENTRYPOINT ["streamlit", "run", "multipagefrontend.py", "--server.port=8080", "--server.address=0.0.0.0"]
