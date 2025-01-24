import io
import time
import streamlit as st
import numpy as np
from google.cloud import storage
from hydra import initialize, compose
import pandas as pd
import matplotlib.pyplot as plt 

st.set_page_config(page_title="User statistics", page_icon="📈")

st.markdown("# User statistics")
st.sidebar.header("User statistics")

with initialize(config_path="./configs", version_base="1.1"):
    cfg = compose(config_name="config")
    
# Get bucket and relevant blobs:
storage_client = storage.Client()
bucket = storage_client.bucket(cfg.gs.bucket)
# Fetch request-database:
user_data_blob = bucket.get_blob("data/monitoring/newdata_predictions_db.csv")
user_data_bytes = user_data_blob.download_as_bytes()
user_data = io.BytesIO(user_data_bytes)

# Read file and keep in variable
df = pd.read_csv(user_data,skipinitialspace=True)
st.write(f"Currently we have served {df.shape[0]} user requests in total across all models!")
df.columns = df.columns.str.strip().str.lower() #some column names seem to have trailing start space 
#allow model choice for group-by-functionality
model_name = st.selectbox('Model type',["All","MediumFit","LargeFit","OverFit"])
model_name_dict = {"All":"All","LargeFit":"fullfit8k32b","MediumFit":"Semifit4k","OverFit":"model_current"}
model_name_choice = model_name_dict[model_name]
column_names = df.columns
if model_name!="All":
    df_tmp = df.loc[df['model_name'] == model_name_choice]
else:
    df_tmp = df.copy()

st.write(f"Overall statistics for {model_name_choice}:")
st.dataframe(df_tmp.describe())

with st.expander("Inference time (s)"):
    inference_data = df_tmp["prediction_time"]
    st.dataframe(inference_data.describe())

    # Create a histogram using Matplotlib
    fig, ax = plt.subplots()
    ax.hist(inference_data, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Interactive Histogram with Streamlit")
    ax.set_xlabel("Inference Time")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with st.expander("Input string lengths"):
    string_lengths = df_tmp["input_length"]
    st.dataframe(string_lengths.describe())
    fig, ax = plt.subplots()
    ax.hist(string_lengths, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Interactive Histogram with Streamlit")
    ax.set_xlabel("Inference Time")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)



