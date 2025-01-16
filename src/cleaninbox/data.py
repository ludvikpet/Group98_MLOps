from pathlib import Path
import json
from datasets import load_dataset, dataset_dict
from transformers import AutoTokenizer, AutoModel
import sys
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path #for resolving paths as originally for loading data
import torch
from torch.utils.data import TensorDataset


# import typer
from torch.utils.data import Dataset
from loguru import logger

# Initialize the logger:
logger.remove()
logger.add(sys.stderr, format="<green>{time}</green> | <level>{level}</level> | {message}")
# logger.add("reports/logs/data.log", rotation="10 MB", format="<green>{time}</green> | <level>{level}</level> | {message}")

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path, model_name: str) -> None:
        self.data_path = raw_data_path
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def tokenize_data(self, data) -> torch.Tensor:
            encoding = self.tokenizer(data["text"],  # List of input texts
            padding=True,                       # Pad to the maximum sequence length
            truncation=False,                   # Truncate to the maximum sequence length if necessary
            return_tensors='pt',                # Return PyTorch tensors
            add_special_tokens=True             # Add special tokens CLS and SEP <- possibly uneeded 
            )
            return encoding["input_ids"]
    
    @logger.catch(level="ERROR")
    def download_data(self, dset_name: str) -> dataset_dict.DatasetDict:
        logger.info(f"Collecting and unpacking dataset {dset_name}.")
        dataset = load_dataset(dset_name, split="train+test", trust_remote_code=True)
        return dataset

    @logger.catch(level="ERROR")
    def preprocess(self, output_folder: Path) -> None:

        # Load raw data and create train/val/test splits:
        dataset = self.download_data(str(self.data_path))
        split = dataset.train_test_split(test_size=0.2, seed=42)
        split2 = split["test"].train_test_split(test_size=0.5, seed=42)
        
        # Get the train, validation, and test data:
        train_data = split["train"]
        val_data = split2["train"]
        test_data = split2["test"]
        
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
        token_train_data_l = train_data.map(self.tokenize_data, batched=True)  # N x MaxSeqLen
        token_val_data_l = val_data.map(self.tokenize_data, batched=True)      # N x MaxSeqLen
        token_test_data_l = test_data.map(self.tokenize_data, batched=True)    # N x MaxSeqLen

        logger.info("Tokenized train example: {}".format(self.tokenizer(train_data[0]["text"])))
        
        # Transform labels to tensors:
        train_data["label"] = torch.tensor(train_data["label"])
        val_data["label"] = torch.tensor(val_data["label"])
        test_data["label"] = torch.tensor(test_data["label"])
        
        # Save the processed data:
        output_folder.mkdir(parents=True, exist_ok=True)
        torch.save(token_train_data_l["input_ids"], output_folder / "train_text.pt")
        torch.save(train_data["label"], output_folder / "train_labels.pt")
        torch.save(token_val_data_l["input_ids"], output_folder / "val_text.pt")
        torch.save(val_data["label"], output_folder / "val_labels.pt")
        torch.save(token_test_data_l["input_ids"], output_folder / "test_text.pt")
        torch.save(test_data["label"], output_folder / "test_labels.pt")

@hydra.main(config_path="configs", config_name="data.yaml", version_base="1.1")
def preprocess(cfg: DictConfig) -> None:

    logger.info("Preprocessing data...")
    
    dataset = MyDataset(Path(cfg.hf.dataset_path), cfg.hf.model_name)
    output_folder = Path("data/processed") / Path(cfg.hf.dataset_path).stem

    dataset.preprocess(output_folder=output_folder)

def produce_raw_data(raw_dir, train_data, val_data, test_data):

    with open(raw_dir/"train_text.json", 'w') as f: 
        json.dump(train_data["text"],f)
    
    with open(raw_dir/"train_labels.json", 'w') as f: 
        json.dump(train_data["label"],f)
    
    with open(raw_dir/"val_text.json", 'w') as f:
        json.dump(val_data["text"],f)
    
    with open(raw_dir/"val_labels.json", 'w') as f:
        json.dump(val_data["label"],f)
    
    with open(raw_dir/"test_text.json", 'w') as f:
        json.dump(test_data["text"],f)
    
    with open(raw_dir/"test_labels.json", 'w') as f:
        json.dump(test_data["label"],f)

# Get processed data:
def text_dataset():
    proc_path = to_absolute_path("data/processed/")+"/"

    train_text = torch.load(proc_path + "train_text.pt")
    train_labels = torch.load(proc_path + "train_labels.pt")
    test_text = torch.load(proc_path + "test_text.pt")
    test_labels = torch.load(proc_path + "test_labels.pt")

    train = TensorDataset(train_text, train_labels)
    test = TensorDataset(test_text, test_labels)

    return train, test    

if __name__ == "__main__":
    preprocess()