import torch
from transformers import BertTokenizer
from .model import BertTypeClassification  # Ensure this points to the correct module
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from omegaconf import OmegaConf
import os

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize FastAPI app
app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "API is working!"}

# Load configuration
CONFIG_PATH = "configs/config.yaml"
cfg = OmegaConf.load(CONFIG_PATH)

# Load model and tokenizer
print("Loading model and tokenizer...")
model_name = cfg.model.name
num_classes = cfg.dataset.num_labels
model = BertTypeClassification(model_name, num_classes).to(DEVICE)
model.load_state_dict(torch.load(cfg.basic.model_ckpt, map_location=DEVICE))
model.eval()
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load label mapping
print("Loading label mapping...")
dataset = load_dataset(cfg.dataset.name, trust_remote_code=True)
label_map = dataset["train"].features["label"].names

# Input data schema
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
            class_name = label_map[predicted_label]

        return {
            "prompt": prompt,
            "predicted_label": predicted_label,
            "class_name": class_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
