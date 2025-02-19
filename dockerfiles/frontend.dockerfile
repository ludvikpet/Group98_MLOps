FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /app

WORKDIR /app

COPY requirements_frontend.txt /app/requirements_frontend.txt
COPY src/cleaninbox/api/api_frontend.py /app/frontend.py

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements_frontend.txt

#EXPOSE $PORT
#try and hardcode to see if this solves it 
EXPOSE 8080 

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port=8080", "--server.address=0.0.0.0"]
