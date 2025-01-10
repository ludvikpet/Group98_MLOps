from pathlib import Path

import typer
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import BertTokenizer

MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"
DATASET_NAME = "PolyAI/banking77"

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.raw_data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

        # Load the dataset
        data = load_dataset(DATASET_NAME, trust_remote_code=True)

        # Save raw data to raw data directory
        torch.save(data['train'], self.raw_data_path.joinpath("train_data_raw.pt"))
        torch.save(data['test'], self.raw_data_path.joinpath("test_data_raw.pt"))

        # Initialize the tokenizer
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        # Tokenize train and test data and convert to PyTorch format
        train_data = tokenizer(data["train"]["text"],  padding=True, return_tensors="pt")
        test_data = tokenizer(data["test"]["text"], padding=True, return_tensors="pt")

        # Save the preprocessed data to the output folder
        torch.save(train_data, output_folder.joinpath("train_data_processed.pt"))
        torch.save(test_data, output_folder.joinpath("test_data_processed.pt"))

def preprocess(raw_data_path: Path = Path("data/raw"), output_folder: Path = Path("data/processed")) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

def load_processed_data(processed_data_path: Path) -> None:
    """Load processed data."""
    train_data = torch.load(processed_data_path.joinpath("train_data_raw.pt"))
    test_data = torch.load(processed_data_path.joinpath("test_data_raw.pt"))

    return train_data, test_data


if __name__ == "__main__":
    typer.run(preprocess)
