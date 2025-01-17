import json
from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from model import BertTypeClassification
from hydra import initialize, compose
from typing import Optional, List


app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, data

    # Initialize Hydra configuration
    with initialize(config_path="../../configs", version_base="1.1"):
        cfg = compose(config_name="config")
    
    model = BertTypeClassification(cfg.model.name)

    try:
        yield

    finally:
        del model, data

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """ Application root """
    return {"message": "Msg from backend"}

@app.post("/classify/")
async def classify_text(file: UploadFile = File(...)):
    """ Get model to classify text """
