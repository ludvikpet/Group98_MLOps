import json
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from model import BertTypeClassification
from hydra import initialize, compose
from typing import Optional, List
from cleaninbox.data import text_dataset
import anyio
from torch.utils.data import DataLoader

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, test_data, text_classes, cfg, DEVICE

    # Initialize Hydra configuration
    with initialize(config_path="../../configs", version_base="1.1"):
        cfg = compose(config_name="config")
    
    # Fetch model:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertTypeClassification(cfg.model.name).to(DEVICE)
    model.load_state_dict(torch.load(cfg.basic.model_ckpt, map_location=DEVICE))
    model.eval()

    # Fetch data:
    _, _, test_data = text_dataset(cfg.basic.proc_banking_data)
    text_classes = test_data.tensors[3].unique().tolist() # test_data.tensors[3] -> labels tensor
    

    try:
        yield

    finally:
        del model, test_data, text_classes, cfg, DEVICE

app = FastAPI(lifespan=lifespan)

def predict():
    test_dataloader = DataLoader(test_data, batch_size = cfg.experiment.overfit.batch_size, shuffle=True)

    correct = 0
    total = 0

    # Disable gradient calculations for evaluation
    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, labels in test_dataloader:
            # Move inputs and labels to the computation device
            input_ids = input_ids.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass to compute logits
            logits = model(input_ids, attention_mask, token_type_ids)

            # Predicted labels are the indices of the maximum logit
            predictions = torch.argmax(logits, dim=1)

            # Update the counters
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Compute and print accuracy
    accuracy = correct / total


@app.get("/")
async def root():
    """ Application root """
    return {"message": "Msg from backend"}

@app.post("/classify/")
async def classify_text(file: Optional[UploadFile] = None, text: Optional[str] = None):
    """ Get model to classify text """
    if not (file or text) or (file and text):
        out_str = "" if (file and text) else "No text specified. "
        return f"{out_str}Must include either raw text- or file input"
    
    try:
        contents = await file.read() if file else text
        if file:
            async with await anyio.open_file(file.filename, "wb") as f:
                f.write(contents)
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/")
async def eval_data(file: Optional[UploadFile] = None):

    try:
        contents = await file.read()
        

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
