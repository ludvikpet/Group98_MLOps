import json
from contextlib import asynccontextmanager
import os

import time
import datetime
import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from hydra import initialize, compose
from typing import Optional
import anyio
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from pydantic import BaseModel
from google.cloud import storage
import io
from pathlib import Path
from loguru import logger
import sys
from datasets import load_dataset
import zipfile
import uvicorn
import evidently
from prometheus_client import Counter, make_asgi_app, Histogram
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

from cleaninbox.model import BertTypeClassification  # Ensure this points to the correct module
from cleaninbox.data import text_dataset, tokenize_data
from cleaninbox.evaluate import eval
from cleaninbox.prediction import pred
# from cleaninbox.data_drift import data_drift

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{message}</green> | {level} | {time:HH:mm:ss}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, test_data, label_names, cfg, DEVICE, storage_client, bucket, tokenizer, error_counter, hist_tracker, reference_data, newdata_blob, lifetime_error_counter

    # Initialize Hydra configuration
    with initialize(config_path="../../../configs", version_base="1.1"):
        cfg = compose(config_name="config")

    # Get bucket and relevant blobs:
    storage_client = storage.Client()
    logger.info(f"Fetching bucket: {cfg.gs.bucket}")
    bucket = storage_client.bucket(cfg.gs.bucket)

    # Fetch model:
    model_blob = bucket.get_blob(cfg.gs.model_ckpt)
    model_bytes = model_blob.download_as_bytes()
    model_ckpt = io.BytesIO(model_bytes)
    logger.info(f"Fetching model checkpoint: {cfg.mount_gs.model_ckpt}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertTypeClassification(cfg.model.name, cfg.dataset.num_labels).to(DEVICE)

    model.load_state_dict(torch.load(model_ckpt, map_location=DEVICE))
    model.eval()

    # Fetch data:
    logger.info(f"Fetching test data with val_size: {cfg.dataset.val_size}, proc_data: {cfg.mount_gs.proc_data}, seed: {cfg.experiment.hyperparameters.seed}, bucket: {bucket}")
    _, _, test_data = text_dataset(cfg.dataset.val_size, cfg.gs.proc_data, "", cfg.experiment.hyperparameters.seed, bucket=bucket)
    dataset = load_dataset(cfg.dataset.name, trust_remote_code=True)
    label_names = dataset["train"].features["label"].names

    logger.info(f"Fetching tokenizer: {cfg.model.name}")
    tokenizer = BertTokenizer.from_pretrained(cfg.model.name)

    # Prometheus metrics:
    error_counter = Counter("errors", "Number of application errors", ["endpoint"])
    model_performance = Histogram("model_performance", "Model performance metrics", ["metric"])

    # New data blob for monitoring:
    # Load processed data from GCS:
    logger.info(f"Loading reference data from GCS: {cfg.gs.monitoring_ref_data}")
    referencedata_blob = bucket.get_blob(cfg.gs.monitoring_ref_data)
    logger.info(f"Downloading blob: {referencedata_blob}")
    data_bytes = referencedata_blob.download_as_bytes()
    reference_data = pd.read_pickle(io.BytesIO(data_bytes))

    # newdata blob for monitoring
    newdata_blob = bucket.blob(cfg.gs.monitoring_db)

    # Prometheus metrics:
    lifetime_error_counter = Counter("lifetime_errors", "Total number of application errors")
    error_counter = Counter("function_errors", "Number of application errors", ["endpoint"]) # Remember to add labels to all errors
    hist_tracker = Histogram("request_duration_seconds", "Request duration in seconds", ["endpoint"]) # Remember to add labels to all requests

    try:
        logger.info("Starting application...")
        yield

    finally:
        logger.info("Shutting down application...")
        del model, test_data, label_names, cfg, DEVICE, storage_client, bucket, tokenizer, newdata_blob, reference_data, error_counter, hist_tracker, lifetime_error_counter

app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app()) # Expose the application metrics at the mounted endpoint /metrics.

@app.get("/")
async def root():
    """ Application root """
    return {"message": "Msg from backend"}

@app.post("/readfiletest/")
async def read_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "contents": contents}

# @app.post("/preprocess/")
# async def preprocess_data(files: List[UploadFile] = File(...)):

#     try:
#         file_details = []
#         for file in files:
#             content = await file.read()
#             async with await anyio.open_file(file.filename, "wb") as f:
#                 f.write(content)
#             file_details.append(file.filename)

#             dataset = MyDataset() # Change needed: pass list of files instead of path
#             preprocess()

#     catch Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

def retrieve_model(model_name: str) -> BertTypeClassification:
    """
    Retrieve a model from Google Cloud Storage.

    Args:
        model_name (str): Name of the model to retrieve.
    """
    # Return model if already loaded:
    if model_name == Path(cfg.gs.model_ckpt).stem:
        return model

    # Download model checkpoint and return model:
    model_ckpt = bucket.get_blob(cfg.gs.model + "/" + model_name + ".pth")
    logger.info(f"Downloading model checkpoint: {model_ckpt.name}")
    file_data = model_ckpt.download_as_bytes()
    ckpt = io.BytesIO(file_data)
    _model = BertTypeClassification(cfg.model.name, cfg.dataset.num_labels).to(DEVICE)
    _model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    _model.eval()

    return _model


@app.get("/download-pretrained-model/")
def download_pretrained(model_name: str="model_current") -> StreamingResponse:
    """
    Download a pretrained model from Google Cloud Storage.

    Args:
        model_name (str): Name of the model to download - defaults to "model_current", used by backend service.
    """
    try:
        # Download model checkpoint:
        model_ckpt = bucket.get_blob(cfg.gs.model + "/" + model_name + ".pth")
        logger.info(f"Downloading model checkpoint: {model_ckpt.name}")
        file_data = model_ckpt.download_as_bytes()
        ckpt = io.BytesIO(file_data)


        # Return model checkpoint:
        filename = os.path.basename(model_ckpt.name)
        logger.info(f"Downloading model checkpoint: {filename}")
        return StreamingResponse(ckpt,
                                 media_type="application/octet-stream",
                                 headers={"Content-Disposition": "attachment; filename={filename}"}
        )

    except Exception as e:
        lifetime_error_counter.inc()
        error_counter.labels("err_download_pretrained_model").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-banking-data/")
def download_banking_data(raw: bool = False) -> StreamingResponse:
    """
    Download banking data to local machine from Google Cloud Storage.

    Args:
        raw (bool): If True, download raw data. If False, download processed data.
    """
    try:
        # Download data:
        logger.info(f"raw path: {cfg.gs.raw_data}, proc path: {cfg.gs.proc_data}")
        data_blobs = bucket.list_blobs(prefix=cfg.gs.raw_data) if raw else bucket.list_blobs(prefix=cfg.gs.proc_data)
        logger.info(f"Fetched data blobs. Raw: {raw}, blobs: {data_blobs}")

        # Create a zip file with the data:
        zip_file = io.BytesIO()
        with zipfile.ZipFile(zip_file, mode="w") as z:
            for blob in data_blobs:
                if blob.name.endswith(".pt"):
                    logger.info(f"Downloading blob: {blob.name}")
                    z.writestr(blob.name, blob.download_as_string())

        zip_file.seek(0)

        # Return the zip file:
        return StreamingResponse(zip_file,
                                    media_type="application/octet-stream",
                                    headers={"Content-Disposition": "attachment; filename={cfg.gs.raw_data.split('/')[-1]}"}
            )

    except Exception as e:
        lifetime_error_counter.inc()
        error_counter.labels("err_download_banking_data").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/")
async def eval_data(texts: Optional[UploadFile]=None, labels: Optional[UploadFile]=None, model_name: str="model_current") -> dict:
    """
    Evaluate model performance on dataset.

    Args:
        texts (UploadFile): File containing input data.
        labels (UploadFile): File containing labels.
        model_name (str): Name of the model to evaluate.
    """
    try:
        init = time.time()

        if texts and labels:

            text = await texts.read()
            labels = await labels.read()

            async with await anyio.open_file(text.filename, "wb") as f:
                f.write(text)
            async with await anyio.open_file(labels.filename, "wb") as f:
                f.write(labels)

            # Load data:
            text = torch.load(text.filename)
            labels = torch.load(labels.filename)

            # Tokenize input data and create dataset:
            tokens = tokenize_data(text, tokenizer)
            data = TensorDataset(tokens["input_ids"], tokens["token_type_ids"], tokens["attention_mask"], labels)

        else:
            data = test_data

        dataset_name = Path(text.filename).stem if texts and labels else Path(cfg.dataset.name).stem

        test_dataloader = torch.utils.data.DataLoader(data, batch_size=cfg.experiment.hyperparameters.batch_size)

        _model = retrieve_model(model_name)
        start = time.time()
        correct, total, accuracy = eval(test_dataloader, _model, DEVICE)
        end = time.time()

        # Save metrics to Prometheus:
        hist_tracker.labels("eval_time").observe(end - start)
        hist_tracker.labels("eval_request_time").observe(end - init)

        return {"dataset": dataset_name, "correct": correct, "total": total, "accuracy": accuracy}

    except Exception as e:
        lifetime_error_counter.inc()
        error_counter.labels("err_evaluate").inc()
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoint
class PredictRequest(BaseModel):
    prompt: str
    model_name: str = "model_current"

@app.post("/predict/")
async def predict(request: PredictRequest,
                  background_tasks: BackgroundTasks) -> dict:
    """
    Predict the label for a given prompt.
    """
    try:
        init = time.time()
        prompt = request.prompt
        _model = retrieve_model(request.model_name)
        logger.info(f"Received prompt: {prompt}")
        start = time.time()
        prediction_result = pred(tokenizer, _model, prompt, label_names, DEVICE)
        end = time.time()

        # Save metrics to Prometheus:
        hist_tracker.labels("prediction_time").observe(end - start)
        hist_tracker.labels("pred_request_time").observe(end - init)

        background_tasks.add_task(save_prediction_to_gcp,
                                  prompt,
                                  request.model_name,
                                  label_names.index(prediction_result['predicted_label']),
                                  end - start)
        return prediction_result

    except Exception as e:
        lifetime_error_counter.inc()
        error_counter.labels("err_predict").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-drift/", response_class=HTMLResponse)
async def data_drift():
    """Use Evidently to generate data drift report"""
    try:
        newdata = load_and_process_newdata()

        await run_data_drift_analysis(reference_data, newdata)

        async with await anyio.open_file("reports/report.html") as f:
            html_content = await f.read()

        return HTMLResponse(content=html_content, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Save prediction results to GCP
def save_prediction_to_gcp(prompt: str, model_name: str, prediction: int, prediction_time: float):
    """Save the prediction results to GCP bucket. Used for monitoring after deployment."""

    logger.info(f"Writing prediction to CSV file: {cfg.gs.monitoring_db}")

    csv_data = newdata_blob.download_as_string().decode('utf-8')

    # Append new data to the CSV
    timestamp = datetime.datetime.now(tz=datetime.UTC)
    csv_data += f"\n{timestamp},{model_name},{len(prompt)},{prediction},{prediction_time}"

    # Upload the updated CSV back to GCP
    newdata_blob.upload_from_string(csv_data)

    logger.info(f"Data written to CSV file: {cfg.gs.monitoring_db}")

async def run_data_drift_analysis(reference_data: pd.DataFrame, new_data: pd.DataFrame):
    """Run the data drift analysis with Evidently and return report. Used for monitoring after deployment."""

    logger.info("Running Evidently analysis...")
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference_data, current_data=new_data)
    report.save_html("reports/report.html")
    logger.info("Saved locally to reports.html. Trying to save to GCS...")
    blob = bucket.blob(f"{cfg.gs.monitoring}report.html")
    logger.info("Fetched blob.")
    with open("reports/report.html","rb") as f:
        blob.upload_from_file(f)
    logger.info("Done running Evidently, report saved to GCS.")


def load_and_process_newdata() -> pd.DataFrame:
    """Download and process the new data from the GCS bucket for monitoring."""

    logger.info(f"Downloading blob: {newdata_blob.name}")
    data_bytes = newdata_blob.download_as_bytes()
    new_data = pd.read_csv(io.BytesIO(data_bytes))

    logger.info("Done retrieving data, preparing data for Evidently...")

    new_data = new_data.drop(columns=['time', 'model_name', 'prediction_time'])
    new_data = new_data.rename(
        columns={
            'prediction': 'target'
        }
    )

    return new_data

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('PORT', 8080)))


