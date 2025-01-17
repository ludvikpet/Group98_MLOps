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

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, test_data, text_classes

    # Initialize Hydra configuration
    with initialize(config_path="../../configs", version_base="1.1"):
        cfg = compose(config_name="config")
    
    # Fetch model:
    model = BertTypeClassification(cfg.model.name)
    model.load_state_dict(torch.load(cfg.basic.current_model))
    model.eval()

    # Fetch data:
    _, _, test_data = text_dataset(cfg.basic.proc_banking_data)
    text_classes = test_data.tensors[3].unique().tolist() # test_data.tensors[3] -> labels tensor
    

    try:
        yield

    finally:
        del model, test_data, text_classes

app = FastAPI(lifespan=lifespan)



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
