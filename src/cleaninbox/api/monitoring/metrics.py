import io
import time
import streamlit as st
import numpy as np
from google.cloud import storage
from hydra import initialize, compose
import pandas as pd
import requests

st.set_page_config(page_title="Internal Metrics", page_icon="📈")

with initialize(config_path="../../../../configs", version_base="1.1"):
    cfg = compose(config_name="config")
    
# Get metrics text:
def get_metrics():
    try:
        metrics_page = requests.get(f"{cfg.gs.backend_url}/metrics")
        metrics_page.raise_for_status()
        return metrics_page.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching metrics page: {e}"

def parse_histogram(line: str, hists: dict, metric_name: str = "request_duration_seconds"):
    if "bucket" in line:
        # Extract endpoint, bucket, and value
        parts = line.split("{")
        labels = parts[1].split("}")[0]  # e.g., endpoint="prediction_time",le="0.1"
        labels_dict = dict(label.split("=") for label in labels.replace('"', '').split(","))
        endpoint = labels_dict["endpoint"]
        bucket = labels_dict["le"]
        value = float(line.split()[-1])

        if endpoint not in hists:
            hists[endpoint] = {}
        hists[endpoint][bucket] = value
    
    elif "sum" in line:
        endpoint = line.split("{")[1].split("}")[0].split("=")[1].strip('"')
        value = float(line.split()[-1])
        hists.setdefault(endpoint, {})["_count"] = value

    elif metric_name in line and "_sum" in line:
            endpoint = line.split("{")[1].split("}")[0].split("=")[1].strip('"')
            value = float(line.split()[-1])
            hists.setdefault(endpoint, {})["_sum"] = value
    
    return hists

def parse_metrics(metrics_text: str, metric_names: list):
    
    errors = []
    hists = {}
    for line in metrics_text.splitlines():

        if any(name in line for name in metric_names) and not line.startswith("#"):
            
            if "le=" in line:
                parse_histogram(line, hists)
                continue

            st.write(f"In errors_total with line: {line}")
            key = (line.split('='))[1][1:].split('"')[0]
            value = float(line.split("}")[1].strip())
            st.write(f"Stripped line: {key}, {value}")
            errors.append((key, value))

        elif line.startswith("lifetime_errors_total"):
            st.write(f"In lifetime_errors_total with line: {line}")
            key, value = line.split(" ")[0], float(line.split(" ")[1])
            st.write(f"Stripped line: {key}, {value}")
            errors.append((key, value))
    
    return errors

def load_metrics_to_html():
    # Backend connection:
    metrics_text = get_metrics()
    if "Error" in metrics_text:
        return st.error(metrics_text)
    
    errors, hists = parse_metrics(metrics_text, ["lifetime_errors_total", "errors_total", "request_duration_seconds"])
    
    st.markdown("## Errors")
    error_df = pd.DataFrame(errors, columns=[key for key, value in errors])
    st.bar_chart(error_df)
    st.write("Above, you'll find a table with the different errors that have occured throughout deployment. The x-axis represents the different errors, while the y-axis represents the number of times the error has occured. For transparency, the number of errors are listed below:")

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    for key, value in errors:
        st.write(f"{key} has occurred {value} times.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Request duration")
    
    # Prepare bucket data:
    bucket_data = {k: v for k, v in hists.items() if k not in ["_count", "_sum"]}
    bucket_labels = list(bucket_data.keys())
    bucket_values = list(bucket_data.values())
    bucket_df = pd.DataFrame(bucket_values, index=bucket_labels)
    st.bar_chart(bucket_df)

    


        

# Base page:
st.markdown("# Internal Application Metrics")
st.text("Below, you'll find different metrics, that describe internal application behaviour.")
st.sidebar.header("Metrics")

load_metrics_to_html()

st.button("Reload metrics", on_click= load_metrics_to_html)