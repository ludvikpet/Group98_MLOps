from pathlib import Path
import json
from typing import Tuple

from datasets import load_dataset 
from torch.utils.data import Dataset, TensorDataset
import datasets
from loguru import logger
from transformers import BertTokenizer
import torch 
from hydra.utils import to_absolute_path #for resolving paths as originally for loading data
import hydra
from omegaconf import DictConfig
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{message}</green> | {level} | {time:HH:mm:ss}")

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path, raw_dir: Path, proc_dir: Path) -> None:
        self.data_path = raw_data_path
        self.raw_dir = Path(to_absolute_path(raw_dir)) / Path(self.data_path).stem
        self.proc_dir = Path(to_absolute_path(proc_dir)) / Path(self.data_path).stem
        logger.info(f"Raw data path: {self.raw_dir}")

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    @logger.catch(level="ERROR")
    def download_data(self, dset_name: str) -> datasets.dataset_dict.DatasetDict:
        logger.info(f"Collecting and unpacking dataset {dset_name}.")
        dataset = load_dataset(dset_name,trust_remote_code=True)
        return dataset

    def tokenize_data(self, text: list, tokenizer: BertTokenizer) -> torch.Tensor:
        encoding = tokenizer(text,# List of input texts
        padding=True,              # Pad to the maximum sequence length
        truncation=False,           # Truncate to the maximum sequence length if necessary
        return_tensors='pt',      # Return PyTorch tensors
        add_special_tokens=True    # Add special tokens CLS and SEP <- possibly uneeded 
        )
        return encoding

    @logger.catch(level="ERROR")
    def preprocess(self, model_name: str) -> None:
        
        # Load data:
        dataset = self.download_data(self.data_path)
        train_text_l, train_labels_l = dataset["train"]["text"], dataset["train"]["label"]
        test_text_l, test_labels_l = dataset["test"]["text"], dataset["test"]["label"]

        #tokenize data:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        train_text = self.tokenize_data(train_text_l,tokenizer) # N x maxSeqLen 
        test_text = self.tokenize_data(test_text_l,tokenizer) # N x maxSeqLen

        train_labels = torch.tensor(train_labels_l).long()
        test_labels = torch.tensor(test_labels_l).long()

        # Save processed data:
        logger.info(f"Saving processed data to {self.proc_dir}.")
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_text, self.proc_dir / "train_text.pt")
        torch.save(train_labels, self.proc_dir / "train_labels.pt")
        torch.save(test_text, self.proc_dir / "test_text.pt")
        torch.save(test_labels, self.proc_dir / "test_labels.pt")

        # Save raw data:
        logger.info(f"Saving raw data to {self.raw_dir}.")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with open(self.raw_dir / "train_text.json", 'w') as f: 
            json.dump(train_text_l,f)

        with open(self.raw_dir / "train_labels.json", 'w') as f: 
            json.dump(train_labels_l,f)

        with open(self.raw_dir / "test_text.json", 'w') as f: 
            json.dump(test_text_l,f)

        with open(self.raw_dir / "test_labels.json", 'w') as f: 
            json.dump(test_labels_l,f)


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.1")
def text_dataset(cfg: DictConfig) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    logger.info(f"Loading processed data: {cfg.dataset.name}")
    proc_path = Path(cfg.basic.proc_path) / Path(cfg.dataset.name).stem

    # Get processed data:
    train_text = torch.load(proc_path / "train_text.pt")
    train_labels = torch.load(proc_path / "train_labels.pt")
    test_text = torch.load(proc_path / "test_text.pt")
    test_labels = torch.load(proc_path / "test_labels.pt")
    train = TensorDataset(train_text["input_ids"], train_text["token_type_ids"], train_text["attention_mask"],train_labels)
    test = TensorDataset(test_text["input_ids"], test_text["token_type_ids"], test_text["attention_mask"], test_labels)

    # Split training data into training and validation sets:
    if cfg.dataset.val_size > 0:
        val_size = int(len(train) * cfg.dataset.val_size / 100)
        train_size = len(train) - val_size
        train, val = torch.utils.data.random_split(train, [train_size, val_size])
        return train, val, test
    
    return train, None, test

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.1")
def preprocess(cfg: DictConfig) -> None:
    logger.info("Preprocessing data...")
    dataset = MyDataset(cfg.dataset.name, raw_dir=cfg.basic.raw_path, proc_dir=cfg.basic.proc_path)
    dataset.preprocess(model_name=cfg.model.name)


if __name__ == "__main__":
    preprocess()