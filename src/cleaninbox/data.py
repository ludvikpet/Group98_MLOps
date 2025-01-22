from pathlib import Path
import json
from typing import Tuple, Optional

from datasets import load_dataset
from torch.utils.data import Dataset, TensorDataset
import datasets
from loguru import logger
from transformers import BertTokenizer
import torch
from hydra.utils import to_absolute_path
import hydra
from omegaconf import DictConfig
import sys
from torch.utils.data import random_split
from google.cloud.storage import Bucket

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{message}</green> | {level} | {time:HH:mm:ss}")
logger.add("logs/preprocessing.log", level="INFO", format="{message} | {level} | {time:YYYY-MM-DD HH:mm:ss}")

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path, raw_dir: Path, proc_dir: Path) -> None:
        self.data_path = raw_data_path
        self.raw_dir = Path(to_absolute_path(raw_dir)) / Path(self.data_path).stem
        self.proc_dir = Path(to_absolute_path(proc_dir)) / Path(self.data_path).stem
        logger.info(f"Raw data path: {self.raw_dir}")

    def __len__(self):  # Could also be empty
        return len(self.train_text) if hasattr(self, "train_text") else 0

    def __getitem__(self, index):  # Could also be empty
        return {
            "input_ids": self.train_text[index],
            "labels": self.train_labels[index]
        }

    @logger.catch(level="ERROR")
    def download_data(self, dset_name: str) -> datasets.dataset_dict.DatasetDict:
        logger.info(f"Collecting and unpacking dataset {dset_name}.")
        dataset = load_dataset(dset_name, trust_remote_code=True)
        assert "train" in dataset and "test" in dataset, "Dataset must contain 'train' and 'test' splits!"
        assert len(dataset["train"]) > 0, "Training dataset is empty!"
        assert len(dataset["test"]) > 0, "Test dataset is empty!"
        return dataset

    @logger.catch(level="ERROR")
    def preprocess(self, model_name: str) -> None:
        dataset = self.download_data(self.data_path)
        train_text_l, train_labels_l = dataset["train"]["text"], dataset["train"]["label"]
        test_text_l, test_labels_l = dataset["test"]["text"], dataset["test"]["label"]

        logger.info(f"Starting tokenization with model: {model_name}")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        train_text = self.tokenize_data(train_text_l, tokenizer)
        test_text = self.tokenize_data(test_text_l, tokenizer)

        train_labels = torch.tensor(train_labels_l).long()
        test_labels = torch.tensor(test_labels_l).long()

        logger.info(f"Saving processed data to {self.proc_dir}.")
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_text, self.proc_dir / "train_text.pt")
        torch.save(train_labels, self.proc_dir / "train_labels.pt")
        torch.save(test_text, self.proc_dir / "test_text.pt")
        torch.save(test_labels, self.proc_dir / "test_labels.pt")

        label_dict = {i: label for i, label in enumerate(dataset["train"].features["label"].names)}
        with open(self.proc_dir / "label_strings.json", "w") as f:
            json.dump(label_dict, f)

        logger.info(f"Saving raw data to {self.raw_dir}.")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        with open(self.raw_dir / "train_text.json", "w") as f:
            json.dump(train_text_l, f)
        with open(self.raw_dir / "train_labels.json", "w") as f:
            json.dump(train_labels_l, f)
        with open(self.raw_dir / "test_text.json", "w") as f:
            json.dump(test_text_l, f)
        with open(self.raw_dir / "test_labels.json", "w") as f:
            json.dump(test_labels_l, f)

    def tokenize_data(self, text: list, tokenizer: BertTokenizer) -> torch.Tensor:
        encoding = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        return encoding
def load_label_strings(proc_path, dataset_name) -> dict[int,str]:
    """
    Loads the integer class-label corresponding string labels for visualizations during training
    Args:
        proc_path (_type_): Path object to the processed dir

    Returns:
        dict[int,str]: dictionary mapping from integer labels to string labels
    """
    logger.info(f"Loading string labels: {dataset_name}")
    proc_path = Path(to_absolute_path(proc_path)) / Path(dataset_name).stem
    with open(proc_path / "label_strings.json","r") as f:
        return json.load(f)

def text_dataset(val_size, proc_path, dataset_name, seed, bucket: Bucket=None) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """ Load the processed text dataset. """
    
    logger.info(f"Loading processed data: {dataset_name}, proc_path: {proc_path}")
    proc_path = Path(to_absolute_path(proc_path)) / Path(dataset_name).stem if not bucket else str(proc_path / Path(dataset_name).stem).replace("\\","/")
    logger.info(f"text_dataset has path: {proc_path}")
    if bucket:
        # Load processed data from GCS:
        train_text = torch.load(bucket.get_blob(proc_path + "/train_text.pt").name)
        train_labels = torch.load(bucket.get_blob(proc_path + "/train_labels.pt").name)
        test_text = torch.load(bucket.get_blob(proc_path + "/test_text.pt").name)
        test_labels = torch.load(bucket.get_blob(proc_path + "/test_labels.pt").name)
    else:
        # Get locally processed data:
        train_text = torch.load(proc_path / "train_text.pt")
        train_labels = torch.load(proc_path / "train_labels.pt")
        test_text = torch.load(proc_path / "test_text.pt")
        test_labels = torch.load(proc_path / "test_labels.pt")

    train = TensorDataset(train_text["input_ids"], train_text["token_type_ids"], train_text["attention_mask"],train_labels)
    test = TensorDataset(test_text["input_ids"], test_text["token_type_ids"], test_text["attention_mask"], test_labels)

    # Split training data into training and validation sets:
    if val_size > 0:
        val_size = int(len(train) * val_size)
        train_size = len(train) - val_size
        train, val = random_split(train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

        return train, val, test

    return train, None, test

def data_split(val_size, dataset: TensorDataset, seed=Optional[int]):
    """ Split the data into training, validation, and test sets. """
    if val_size > 0:
        val_size = int(len(train) * val_size)
        train_size = len(train) - val_size
        train, val = random_split(train, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

        return train, val
    

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.1")
def preprocess(cfg: DictConfig) -> None:
    logger.info("Preprocessing data...")
    dataset = MyDataset(cfg.dataset.name, raw_dir=cfg.basic.raw_path, proc_dir=cfg.basic.proc_path)
    dataset.preprocess(model_name=cfg.model.name)

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.1")
def run_text_dataset(cfg: DictConfig) -> None:
    train, val, test = text_dataset(cfg.dataset.val_size, cfg.basic.proc_path, cfg.dataset.name, cfg.experiment.hyperparameters.seed)
    logger.info(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")

if __name__ == "__main__":
    #preprocess()
    run_text_dataset()
