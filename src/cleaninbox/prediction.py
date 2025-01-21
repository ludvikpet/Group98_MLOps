import torch
from transformers import BertTokenizer
from model import BertTypeClassification  # Ensure this points to the correct module
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import hydra
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        )

        # Move inputs to device
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        token_type_ids = encoding.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        # Generate prediction
        print("Generating prediction...")
        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)
            predicted_label = torch.argmax(logits, dim=1).item()

        # Map label to class name
        class_name = label_map[predicted_label]
        print(f"Predicted Label: {predicted_label} ({class_name})")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    predict()
