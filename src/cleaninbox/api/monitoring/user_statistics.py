import io
import time
import streamlit as st
import numpy as np
from google.cloud import storage
from hydra import initialize, compose


st.set_page_config(page_title="Data drifting", page_icon="📈", layout="wide")

st.markdown("# User statistics")
st.sidebar.header("User statistics")

with initialize(config_path="../../../../configs", version_base="1.1"):
    cfg = compose(config_name="config")
    
# Get bucket and relevant blobs:
storage_client = storage.Client()
bucket = storage_client.bucket(cfg.gs.bucket)
# Fetch request-database:
user_data_blob = bucket.get_blob("data/monitoring/newdata_predictions_db.csv")
user_data_bytes = user_data_blob.download_as_bytes()
user_data = io.BytesIO(user_data_bytes)

# Read file and keep in variable

