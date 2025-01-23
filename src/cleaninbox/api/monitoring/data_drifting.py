import streamlit as st
import time
import numpy as np
from google.cloud import storage
from hydra import initialize, compose


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
user_data_bytes = user_data_blob.download_as_bytes()
user_data = io.BytesIO(user_data_bytes)


path_to_html = "./monitoring/reports/report.html" 


# Read file and keep in variable
with open(path_to_html,'r') as f: 
    html_data = f.read()

## Show in webpage
st.header("Show an external HTML")
st.components.v1.html(html_data, height=400, scrolling=True)

#old
# st.write(
#     """This demo illustrates a combination of plotting and animation with
# Streamlit. We're generating a bunch of random numbers in a loop for around
# 5 seconds. Enjoy!"""
# )

# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()
# last_rows = np.random.randn(1, 1)
# chart = st.line_chart(last_rows)

# for i in range(1, 101):
#     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#     status_text.text("%i%% Complete" % i)
#     chart.add_rows(new_rows)
#     progress_bar.progress(i)
#     last_rows = new_rows
#     time.sleep(0.05)

# progress_bar.empty()

# # Streamlit widgets automatically run the script from top to bottom. Since
# # this button is not connected to any other logic, it just causes a plain
# # rerun.
# st.button("Re-run")