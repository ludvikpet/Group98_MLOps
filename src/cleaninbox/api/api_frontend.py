import os
import pandas as pd
from pydantic import BaseModel
import requests
import streamlit as st
from google.cloud import run_v2

# Prediction endpoint
class PredictRequest(BaseModel):
    prompt: str


# @st.cache_resource  
# def get_backend_url():
#     """Get the URL of the service automatically."""
#     projectID = "cleaninbox-448011"
#     REGION = "europe-west1"
#     parent = f"projects/{projectID}/locations/{REGION}"
#     client = run_v2.ServicesClient()
#     services = client.list_services(parent=parent)
#     for service in services:
#         if service.name.split("/")[-1] == "backend":
#             st.write(service.uri) #debugging
#             return service.uri
#     name = os.environ.get("BACKEND", None)
#     st.write(name) #debugging
#     return name

def get_backend_url():
    return "http://127.0.0.1:8000"
    
def classify_email(request: PredictRequest, backend):
    """Send the email struct to the backend for classification."""
    predict_url = f"{backend}/predict/"
    st.write(f"Predict-url: {predict_url}")
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


def main() -> None:
    """Main function of the Streamlit frontend."""
    #st.set_page_config(layout="wide")
    st.title("Welcome to :mailbox_with_no_mail: cleaninbox!")
    st.text("We help banks by automatically flagging e-mails such that they are sorted into the correct mail-inbox.")
    st.text("Want to ensure that your mail reaches the right person such that you get the fastest possible response? Insert your e-mail below!")
    
    ###processing of inputs
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    input_str = st.text_input("Your e-mail for classification: ","")
    request = PredictRequest(prompt=input_str)
    #uploaded_file = st.file_uploader("Enter a text-string", type=["jpg", "jpeg", "png"])

    if len(request.prompt)>0:
        result = classify_email(request, backend=backend)
        st.write(result)

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
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
    #port = int(os.environ.get("PORT", 8080))  # Default Streamlit port is 8501
    #host = "0.0.0.0" # Bind to all interfaces, hard-coded, a bit dump
    #st.web.bootstrap.run(main, f"--server.port={port} --server.address={host}", [])