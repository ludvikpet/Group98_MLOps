import json
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
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


from cleaninbox.model import BertTypeClassification  # Ensure this points to the correct module
from cleaninbox.data import text_dataset, MyDataset, data_split, tokenize_data
from cleaninbox.evaluate import eval
from cleaninbox.train import train

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{message}</green> | {level} | {time:HH:mm:ss}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, test_data, labels, cfg, DEVICE, storage_client, bucket, model_ckpt, tokenizer


    # Initialize Hydra configuration
    with initialize(config_path="../../../configs", version_base="1.1"):
        cfg = compose(config_name="config")
    
    # Get bucket and relevant blobs:
    storage_client = storage.Client()
    bucket = storage_client.bucket(cfg.gs.bucket)
    model_ckpt = bucket.get_blob(cfg.gs.model_ckpt)
    logger.info(f"Model checkpoint: {model_ckpt}, type: {type(model_ckpt)}, location: {cfg.gs.model_ckpt}")
    logger.info(f"proc data location: {cfg.gs.proc_data}")
    # Fetch model:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertTypeClassification(cfg.model.name, cfg.dataset.num_labels).to(DEVICE)
    model.load_state_dict(torch.load(model_ckpt.name, map_location=DEVICE))
    model.eval()

    # Fetch data:
    _, _, test_data = text_dataset(cfg.dataset.val_size, Path(cfg.gs.proc_data), "", cfg.experiment.hyperparameters.seed, bucket=bucket)
    # text_classes = test_data.tensors[3].unique().tolist() # test_data.tensors[3] -> labels tensor
    labels = cfg.dataset.num_labels # test_data.tensors[3] -> labels tensor

    tokenizer = BertTokenizer.from_pretrained(cfg.model.name)

    try:
        yield

    finally:
        del model, test_data, labels, cfg, DEVICE, storage_client, bucket, model_ckpt, tokenizer

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
def download_pretrained():

    try:

        # Download model checkpoint:
        file_data = model.download_as_bytes()
        model_ckpt = io.BytesIO(file_data)

        # Return model checkpoint:
        return StreamingResponse(model_ckpt,
                                 media_type="application/octet-stream",
                                 headers={"Content-Disposition": "attachment; filename={cfg.gs.model_ckpt.split('/')[-1]}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/download-banking-data/")
# def download_banking_data(raw: bool = False):
#     try:
#         # Download raw data:
#         file_data = raw_data.download_as_bytes() if raw else proc_data.download_as_bytes()
#         data = io.BytesIO(file_data)

#         # Return raw data:
#         return StreamingResponse(data,
#                                  media_type="application/octet-stream",
#                                  headers={"Content-Disposition": "attachment; filename={cfg.gs.raw_data.split('/')[-1]}"}
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/")
async def classify_text(file: Optional[UploadFile] = None, text: Optional[str] = None):
    if not (file or text) or (file and text):
        out_str = "" if (file and text) else "No text specified. "
        return f"{out_str}Must include either raw text- or file input"
    
    try:
        
        if file:
            contents = await file.read()
            async with await anyio.open_file(file.filename, "wb") as f:
                f.write(contents)
            
            async with await anyio.open_file(file.filename, "r") as f:
                samples = f.readlines().split("\n")
        else:
            samples = text.split("\n")
        
        return "Probably worked, but not implemented yet :)"
        # predictions = predict(samples)

        # output_dir = {f"sample_{i}": {"text": sample, "label": text_classes[pred]} for i, (sample, pred) in enumerate(zip(samples, predictions))}
        # return output_dir

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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


class PromptRequest(BaseModel):
    prompt: str

# Prediction endpoint
@app.post("/predict")
async def predict(request: PromptRequest):
    """
    Predict the label for a given prompt.
    """
    try:
        prompt = request.prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is empty")

        print(f"Received prompt: {prompt}")

        # Tokenize input
        encoding = tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        token_type_ids = encoding.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        # Generate prediction
        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
            predicted_label = torch.argmax(logits, dim=1).item()
            class_name = labels[predicted_label]

        return {
            "prompt": prompt,
            "predicted_label": predicted_label,
            "class_name": class_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
