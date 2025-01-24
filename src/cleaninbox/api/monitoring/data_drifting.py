import io 
import requests
import time

from google.cloud import storage
from hydra import initialize, compose
import numpy as np
import streamlit as st


st.set_page_config(page_title="Data drifting", page_icon="📈", layout="wide")
@st.cache_resource  
def get_backend_url():
    return "https://backend-170780472924.europe-west1.run.app"

st.markdown("# Data drifting monitoring")
st.sidebar.header("Data drifting monitoring")

@st.cache_data
def load_html():    
    with initialize(config_path="../../../../configs", version_base="1.1"):
        cfg = compose(config_name="config")
    # Get bucket and relevant blobs:
    storage_client = storage.Client()
    bucket = storage_client.bucket(cfg.gs.bucket)
    # Fetch request-database:
    report_blob = bucket.get_blob("data/monitoring/reports/reports.html")
    report_bytes = report_blob.download_as_bytes()
    #html_report = io.BytesIO(report_bytes)
    html_report = report_bytes.decode("utf-8")  # Decode the bytes to a string
    return html_report

#for disabling button after click - only allow users to generate one button per session
if 'generate_button' in st.session_state and st.session_state.generate_button == True:
    st.session_state.running = True
else:
    st.session_state.running = False

#if st.button('Process', disabled=st.session_state.running, key='run_button'):
if st.button("Generate report!",disabled=st.session_state.running,key="generate_button"):
    backend_url = get_backend_url()
    with st.spinner("generating report..."):
        try:
            response = requests.get(url=f"{backend_url}/data-drift/", timeout=120) #await handled in backend
            response.raise_for_status()
            st.success("Report generated successfully")
            html_data = response.text
        except requests.RequestException as e:
            st.error("Failed to generate report: {e}")
    #html_data = load_html()
    if html_data:
        st.components.v1.html(html_data, height=550, scrolling=True)
        #path_to_html = "./monitoring/reports/report.html" 
else:
    st.write("No report to display. Click the button the generate one!")






#path_to_html = "./monitoring/reports/report.html" 

# # Read file and keep in variable
# with open(path_to_html,'r') as f: 
#     html_data = f.read()

# ## Show in webpage
# st.header("Show an external HTML")
# st.components.v1.html(html_data, height=400, scrolling=True)

# #old
# # st.write(
# #     """This demo illustrates a combination of plotting and animation with
# # Streamlit. We're generating a bunch of random numbers in a loop for around
# # 5 seconds. Enjoy!"""
# # )

# # progress_bar = st.sidebar.progress(0)
# # status_text = st.sidebar.empty()
# # last_rows = np.random.randn(1, 1)
# # chart = st.line_chart(last_rows)

# # for i in range(1, 101):
# #     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
# #     status_text.text("%i%% Complete" % i)
# #     chart.add_rows(new_rows)
# #     progress_bar.progress(i)
# #     last_rows = new_rows
# #     time.sleep(0.05)

# # progress_bar.empty()

# # # Streamlit widgets automatically run the script from top to bottom. Since
# # # this button is not connected to any other logic, it just causes a plain
# # # rerun.
# # st.button("Re-run")
