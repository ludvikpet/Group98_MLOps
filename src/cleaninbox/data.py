from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import sys
import hydra
from omegaconf import DictConfig


# import typer
from torch.utils.data import Dataset
from loguru import logger

# Initialize the logger:
logger.remove()
logger.add(sys.stderr, format="<green>{time}</green> | <level>{level}</level> | {message}")
# logger.add("reports/logs/data.log", rotation="10 MB", format="<green>{time}</green> | <level>{level}</level> | {message}")

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

@hydra.main(config_path="configs", config_name="data.yaml", version_base="1.1")
def preprocess(cfg: DictConfig) -> None:
    logger.info("Preprocessing data...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf.model_name)

    def tokenization(data):
        return tokenizer(data["text"], return_tensors="pt")
    
    # model = AutoModel.from_pretrained(cfg.hf.model_name)
    
    # Load raw data and create train/val/test splits:
    dataset = load_dataset(cfg.hf.dataset_path, split="train+test")
    split = dataset.train_test_split(test_size=0.2, seed=42)
    split2 = split["test"].train_test_split(test_size=0.5, seed=42)
    
    # Get the train, validation, and test data:
    train_data = split["train"]
    val_data = split2["train"]
    test_data = split2["test"]

    logger.info("Train example: {}".format(train_data[0]))
    logger.info(f"Train size: {len(train_data)}, Validation size: {len(val_data)}, Test size: {len(test_data)}")

    # Tokenize the data:
    token_train_data, token_val_data, token_test_data = train_data.map(tokenization, batched=True), val_data.map(tokenization, batched=True), test_data.map(tokenization, batched=True)
    logger.info("Tokenized train example: {}".format(tokenizer(train_data[0]["text"])))

    train_data.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
    
    # Create the output folder:
    output_folder = Path("data/processed") / Path(cfg.hf.dataset_path).stem

    # Save the processed data:
    train_data.save_to_disk(output_folder / "train")
    val_data.save_to_disk(output_folder / "val")
    test_data.save_to_disk(output_folder / "test")
    

if __name__ == "__main__":
    preprocess()
    # typer.run(preprocess)
