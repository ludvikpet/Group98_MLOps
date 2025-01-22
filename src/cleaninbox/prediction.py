import torch
from transformers import BertTokenizer
from cleaninbox.model import BertTypeClassification, top_k_logits  # Ensure this points to the correct module
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from loguru import logger
import hydra
from datasets import load_dataset
import sys

def pred(tokenizer: BertTokenizer, model: BertTypeClassification, prompt: str, label_map: list, DEVICE: torch.device) -> dict:
    
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{message}</green> | {level} | {time:HH:mm:ss}")
    
    encoding = tokenizer(
            [prompt],
            padding=True,       # Pad to the longest sequence in the batch
            truncation=False,   # Do not truncate
            return_tensors="pt"
        ).to(DEVICE)

    # Generate prediction
    logger.info("Generating prediction...")
    with torch.no_grad():
        logits = model(encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"])
        predicted_label = torch.argmax(logits, dim=1).item()
    
        # Top-k predictions:
        topk_values, topk_indices = top_k_logits(logits=logits, k=10)
        topk_labels = [label_map[idx] for idx in topk_indices]
        label_tups = list(zip(topk_values, topk_labels)) # Tuple list (value, label)
    
    return {"predicted_label": label_map[predicted_label], "topk_labels": label_tups}

@hydra.main(config_path=to_absolute_path("configs"), config_name="config", version_base="1.1")
def predict(cfg: DictConfig) -> None:
    """
    Predict the label for a given prompt.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    try:
        print("Starting prediction...")
        prompt = cfg.prompt
        print(f"Prompt: {prompt}")

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        print("Initializing model...")
        model_name = cfg.model.name
        num_classes = cfg.dataset.num_labels
        model = BertTypeClassification(model_name, num_classes).to(DEVICE)

        # Load model weights
        print("Loading model weights...")
        model.load_state_dict(torch.load(to_absolute_path(cfg.basic.model_ckpt), map_location=DEVICE))
        model.eval()

        # Load label mapping dynamically from dataset
        print("Loading label mapping...")
        dataset = load_dataset(cfg.dataset.name, trust_remote_code=True)
        label_map = dataset["train"].features["label"].names

        # Tokenize input
        print("Tokenizing input...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        encoding = tokenizer(
            [prompt],
            padding=True,       # Pad to the longest sequence in the batch
            truncation=False,   # Do not truncate
            return_tensors="pt"
        ).to(DEVICE)

        # Move inputs to device
        # input_ids = encoding["input_ids"].to(DEVICE)
        # attention_mask = encoding["attention_mask"].to(DEVICE)
        # token_type_ids = encoding.get("token_type_ids", None)
        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.to(DEVICE)

        # Generate prediction
        print("Generating prediction...")
        with torch.no_grad():
            logits = model(encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"])
            predicted_label = torch.argmax(logits, dim=1).item()

            # Top-k predictions:
            k = 10
            topk_values, topk_indices = torch.topk(logits, k, dim=1)
            topk_values = topk_values.cpu().numpy().tolist()[0]
            topk_indices = topk_indices.cpu().numpy().tolist()[0]
            topk_labels = [label_map[idx] for idx in topk_indices]

        # Map label to class name
        class_name = label_map[predicted_label]
        print(f"Predicted Label: {predicted_label} ({class_name})")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    predict()
