import streamlit as st
import time
import numpy as np
from google.cloud import storage
from hydra import initialize, compose
import io 
import pandas as pd 


st.set_page_config(page_title="Data drifting", page_icon="📈", layout="wide")

st.markdown("# Data drifting monitoring")
st.sidebar.header("Data drifting monitoring")

#generate a report: needs to be added here
with initialize(config_path="../../../../configs", version_base="1.1"):
    cfg = compose(config_name="config")
    
# Get bucket and relevant blobs:
storage_client = storage.Client()
bucket = storage_client.bucket(cfg.gs.bucket)
# Fetch request-database:
user_data_blob = bucket.get_blob("data/monitoring/newdata_predictions_db.csv")
st.write(f"Trying to fetch request data from {user_data_blob}")
user_data_bytes = user_data_blob.download_as_bytes()
user_data = io.BytesIO(user_data_bytes)

"""
from here, it should simply load user data csv, 
handle it with evidently logic and compare to training data 
- except if we want backend to handle it directly. 
If handled in backend, we simply need to fetch html report from directory
Add Adams + Ludviks progress here, and we should be good"""
df = pd.read_csv(user_data)

###HANDLE DATA AND CAST TO DATAFRAME
###GENERATE HTML WITH EVIDENTLY
###SAVE TO REPORT.HTML 
path_to_html = "./monitoring/reports/report.html" 


# Read file and keep in variable
with open(path_to_html,'r') as f: 
    html_data = f.read()

## Show in webpage
st.header("Show an external HTML")
st.components.v1.html(html_data, height=400, scrolling=True)
