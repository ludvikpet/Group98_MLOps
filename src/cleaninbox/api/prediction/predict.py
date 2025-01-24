import os
import pandas as pd
from pydantic import BaseModel
import requests
import streamlit as st
from hydra import initialize, compose

st.set_page_config(page_title="Cleaninbox", page_icon=":mailbox_with_no_mail:")


# Prediction endpoint
class PredictRequest(BaseModel):
    prompt: str
    model_name: str

with initialize(config_path="../../../configs", version_base="1.1"):
    cfg = compose(config_name="config")

@st.cache_resource  
def get_backend_url():
    return cfg.gs.backend_url

    
def classify_email(request: PredictRequest, backend: str):
    """Send the email struct to the backend for classification."""
    predict_url = f"{backend}/predict/"
    #st.write(f"Predict-url: {predict_url}")
    #response = requests.post(predict_url, files={"image": image}, timeout=60)
    #files = {"file": ("image.jpg", mail_str, "image/jpeg")}
    response = requests.post(url=predict_url, json=request.model_dump(), timeout=60)
    #response returns a dictionary :
    #{"predicted_label": label_map[predicted_label], "topk_labels": label_tups}
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error from backend: {response.status_code} - {response.text}")
    return None



#"""Main function of the Streamlit frontend."""
#st.set_page_config(layout="wide")
st.markdown("# Let's classify your e-mail!")
st.sidebar.header("Predict e-mail classification")
st.text("Want to ensure that your mail reaches the right person such that you get the fastest possible response? Insert your e-mail string below, and press the button")


#st.sidebar.success("Select a demo above.")
###processing of inputs
#dashboard = st.Page(
#    "monitoring/data_drifting.py", title="Monitoring Dashboard", icon=":material/dashboard:", default=False
#)
backend = get_backend_url()
if backend is None:
    msg = "Backend service not found"
    raise ValueError(msg)

#old way - enter, but no model drop-down
#input_str = st.text_input("Your e-mail for classification: ","")
#request = PredictRequest(prompt=input_str)
#logic for handling inputs
model_name = st.selectbox('Model type',["MediumFit","LargeFit","OverFit"])
with st.expander("Model explanation"):
    #make a pandas dataframe to explain model types 
    modelDict = {"Model Name":["MediumFit","LargeFit","OverFit"],
                 "Size of training set": [4000,8000,1000],
                 "Batch-size during training": [64,16,64],
                 "Dataset": ["PolyAI/banking77","PolyAI/banking77","PolyAI/banking77"],
                 "Aliases": ["Semifit4k","fullfit8k32b","model_current"]
                }
    model_df = pd.DataFrame(data=modelDict)
    st.text("We provide three different model types, and you can choose which one you like:")
    st.dataframe(model_df)

string_mail_form = st.form("predict-form")
email_string = string_mail_form.text_input("Your e-mail string for classification","")
submit = string_mail_form.form_submit_button("Classify my e-mail!")
model_name_dict = {"LargeFit":"fullfit8k32b","MediumFit":"Semifit4k","OverFit":"model_current"}
if submit: 
    model_name_choice = model_name_dict[model_name]
    request = PredictRequest(prompt=email_string,model_name=model_name_choice)
    st.write(request)
    if len(request.prompt)>0:
        with st.spinner("classifying... this could take time..."):
            result = classify_email(request=request,backend=backend)

        if result is not None:
            prediction = result["predicted_label"]
            topk_tups = result["topk_labels"] 
            topk_labels = [x[1] for x in topk_tups]
            topk_probs = [x[0] for x in topk_tups] 

            #has been made as: 
            #label_tups = list(zip(topk_values, topk_labels)) # Tuple list (value, label)
            
            st.write("Predicted class:", prediction)

            # make a nice bar chart
            data = {
                "Class": topk_labels,
                "Probability": topk_probs
            }
            df = pd.DataFrame(data)
            st.dataframe(df)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability",horizontal=True)
    else:
        st.write("We can't classify an empty e-mail.")

