import io
import time
import streamlit as st
import numpy as np
from google.cloud import storage
from hydra import initialize, compose
import pandas as pd
import requests

st.set_page_config(page_title="Internal Metrics", page_icon="📈")

with initialize(config_path="./configs", version_base="1.1"):
    cfg = compose(config_name="config")
    
# Get metrics text:
def get_metrics():
    try:
        metrics_page = requests.get(f"{cfg.gs.backend_url}/metrics")
        metrics_page.raise_for_status()
        return metrics_page.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching metrics page: {e}"

def parse_histogram(line: str, hists: dict, sum_counts, metric_name: str = "request_duration_seconds"):
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
    
    elif metric_name in line and "count" in line:
        endpoint = (line.split('='))[1][1:].split('"')[0]
        value = float(line.split()[-1])
        hists.setdefault(endpoint, {})["_count"] = value
        sum_counts[endpoint] = [0.0, 0.0]
        sum_counts[endpoint][1] = value

    elif metric_name in line and "_sum" in line:
            endpoint = (line.split('='))[1][1:].split('"')[0]
            value = float(line.split()[-1])
            sum_counts[endpoint][0] = value
    
    return hists, sum_counts

def parse_metrics(metrics_text: str, metric_names: list):
    
    errors = []
    hists = {}
    sum_counts = {}
    for line in metrics_text.splitlines():

        if any(name in line for name in metric_names) and not line.startswith("#"):
            
            if "request" in line:
                hists, sum_counts = parse_histogram(line, hists, sum_counts)
                continue
            
            elif line.startswith("lifetime_errors_total"):
                key, value = line.split(" ")[0], float(line.split(" ")[1])
                errors.append((key, value))
                continue

            key = (line.split('='))[1][1:].split('"')[0]
            value = float(line.split("}")[1].strip())
            errors.append((key, value))
    
    return errors, hists, sum_counts

def load_metrics_to_html():
    # Backend connection:
    metrics_text = get_metrics()
    if "Error" in metrics_text:
        return st.error(metrics_text)
    
    errors, hists, sum_counts = parse_metrics(metrics_text, ["lifetime_errors_total", "errors_total", "request_duration_seconds"])
    
    st.markdown("## Errors")
    if not errors or (len(errors) == 1 and errors[0][1] == 0):
        st.write("No errors have occured during deployment.")
    else:
        if len(errors) > 1:
            st.write("Overview over the set of errors for our application throughout deployment. The bar-chart describes both the total number of errors for all API functions, but also errors for each individual function. The x-axis represents the different errors, while the y-axis represents the number of times the error has occured. For transparency, the number of errors are listed beneath the chart!")
            error_df = pd.DataFrame(errors, columns=["Error", "Count"])
            st.bar_chart(error_df.set_index("Error"))

        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        for key, value in errors:
            st.write(f"{key} has occurred {value} times.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Request duration calls")
    if not hists:
        st.write("No requests have been recorded during deployment.")
    else:
        st.write("Below, you'll find a set of histograms that describe the duration of different user requests, their total call counts as well as statistics for the sum of request durations.")

        for endpoint, hist in hists.items():
            st.markdown(f"### {endpoint}")
            
            # Prepare bucket data:
            bucket_data = {k: v for k, v in hist.items() if k not in ["_count", "_sum"]}
            bucket_labels = list(bucket_data.keys())
            bucket_values = list(bucket_data.values())
            bucket_df = pd.DataFrame(bucket_values, index=bucket_labels)
            st.bar_chart(bucket_df)

            # Prepare count and sum data:
            count = sum_counts[endpoint][1]
            sum_value = sum_counts[endpoint][0]
            st.write(f"*Total number of requests:* **{count}**")
            st.write(f"*Total sum of request durations:* **{sum_value}s**")
            st.write(f"*Average request duration:* **{sum_value / count}s**")


# Define a session state to trigger reloading
if "reload_trigger" not in st.session_state:
    st.session_state.reload_trigger = False

def trigger_reload():
    # Toggle the reload trigger
    st.session_state.reload_trigger = True
    # Use experimental set query params to force a page reload
    st.experimental_set_query_params(reload=True)

st.markdown("# Internal Application Metrics")
st.text("Below, you'll find different metrics that describe internal application behavior. These are primarily used for debugging and monitoring purposes, however, here they are for your viewing pleasure!")
st.sidebar.header("Metrics")

# Button to reload metrics
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.button("Reload metrics", on_click=trigger_reload)
st.markdown("</div>", unsafe_allow_html=True)

# Reload metrics when the trigger toggles
if st.session_state.reload_trigger:
    # Clear the page and reload metrics
    st.session_state.reload_trigger = False
    load_metrics_to_html()
else:
    load_metrics_to_html()
