import json
from contextlib import asynccontextmanager
import os 

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
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


from cleaninbox.model import BertTypeClassification  # Ensure this points to the correct module
from cleaninbox.data import text_dataset, tokenize_data
from cleaninbox.evaluate import eval
from cleaninbox.prediction import pred

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{message}</green> | {level} | {time:HH:mm:ss}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, test_data, label_names, cfg, DEVICE, storage_client, bucket, tokenizer

    # Initialize Hydra configuration
    with initialize(config_path="../../../configs", version_base="1.1"):
        cfg = compose(config_name="config")
    
    # Get bucket and relevant blobs:
    storage_client = storage.Client()
    logger.info(f"Fetching bucket: {cfg.gs.bucket}")
    bucket = storage_client.bucket(cfg.gs.bucket)

    # Fetch model:
    logger.info(f"Fetching model checkpoint: {cfg.mount_gs.model_ckpt}")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertTypeClassification(cfg.model.name, cfg.dataset.num_labels).to(DEVICE)
    model.load_state_dict(torch.load(cfg.mount_gs.model_ckpt, map_location=DEVICE))
    model.eval()

    # Fetch data:
    logger.info(f"Fetching test data with val_size: {cfg.dataset.val_size}, proc_data: {cfg.mount_gs.proc_data}, seed: {cfg.experiment.hyperparameters.seed}, bucket: {bucket}")
    _, _, test_data = text_dataset(cfg.dataset.val_size, cfg.mount_gs.proc_data, "", cfg.experiment.hyperparameters.seed, bucket=bucket)
    dataset = load_dataset(cfg.dataset.name, trust_remote_code=True)
    label_names = dataset["train"].features["label"].names

    logger.info(f"Fetching tokenizer: {cfg.model.name}")
    tokenizer = BertTokenizer.from_pretrained(cfg.model.name)

    try:
        logger.info("Starting application...")
        yield

    finally:
        logger.info("Shutting down application...")
        del model, test_data, label_names, cfg, DEVICE, storage_client, bucket, tokenizer

app = FastAPI(lifespan=lifespan)

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

@app.get("/download-pretrained-model/")
def download_pretrained(model_name: str="model_current"):

    try:
        # Download model checkpoint:
        model_ckpt = bucket.get_blob(cfg.gs.models + "/" + model_name + ".pth")    
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-banking-data/")
def download_banking_data(raw: bool = False):
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
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/classify/")
# async def classify_text(file: Optional[UploadFile] = None, text: Optional[str] = None):
#     if not (file or text) or (file and text):
#         out_str = "" if (file and text) else "No text specified. "
#         return f"{out_str}Must include either raw text- or file input"
    
#     try:
        
#         if file:
#             contents = await file.read()
#             async with await anyio.open_file(file.filename, "wb") as f:
#                 f.write(contents)
            
#             async with await anyio.open_file(file.filename, "r") as f:
#                 samples = f.readlines().split("\n")
#         else:
#             samples = text.split("\n")
        
#         return "Probably worked, but not implemented yet :)"
#         # predictions = predict(samples)

#         # output_dir = {f"sample_{i}": {"text": sample, "label": text_classes[pred]} for i, (sample, pred) in enumerate(zip(samples, predictions))}
#         # return output_dir

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/")
async def eval_data(texts: Optional[UploadFile]=None, labels: Optional[UploadFile]=None):

    try:

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
        correct, total, accuracy = eval(test_dataloader, model, DEVICE)

        return {"dataset": dataset_name, "correct": correct, "total": total, "accuracy": accuracy}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoint
class PredictRequest(BaseModel):
    prompt: str

@app.post("/predict/")
async def predict(request: PredictRequest):
    """
    Predict the label for a given prompt.
    """
    try:
        prompt = request.prompt
        logger.info(f"Received prompt: {prompt}")
        return pred(tokenizer, model, prompt, label_names, DEVICE)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080))) 
    #remember to add uvicorn to requirements 
    #docker format using this approach is: 
    #EXPOSE $PORT
    #CMD exec uvicorn --port $PORT --host 0.0.0.0 api_backend:app