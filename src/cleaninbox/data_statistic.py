from pathlib import Path
import json
from typing import Tuple, Optional

from datasets import load_dataset
from torch.utils.data import Dataset, TensorDataset
import datasets
from loguru import logger
from transformers import BertTokenizer
import torch
from hydra.utils import to_absolute_path #for resolving paths as originally for loading data
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import sys
from torch.utils.data import random_split
from google.cloud.storage import Bucket



def data_statistics(proc_dir: Path, dataset_name: str, bucket: Optional[Bucket] = None) -> None:
    """
    Generate and display statistics about the dataset.

    Args:
        proc_dir (Path): Path to the processed data directory.
        dataset_name (str): Name of the dataset.
        bucket (Optional[Bucket]): If using Google Cloud Storage, pass the bucket instance.
    """
    # Resolve the path to the processed dataset
    proc_path = Path(to_absolute_path(proc_dir)) / Path(dataset_name).stem if not bucket else str(proc_dir / Path(dataset_name).stem).replace("\\", "/")
    logger.info(f"Processed data path: {proc_path}")

    # Load processed data
    try:
        if bucket:
            train_text = torch.load(proc_path + "/train_text.pt")
            train_labels = torch.load(proc_path + "/train_labels.pt")
            test_text = torch.load(proc_path + "/test_text.pt")
            test_labels = torch.load(proc_path + "/test_labels.pt")
        else:
            train_text = torch.load(proc_path / "train_text.pt")
            train_labels = torch.load(proc_path / "train_labels.pt")
            test_text = torch.load(proc_path / "test_text.pt")
            test_labels = torch.load(proc_path / "test_labels.pt")
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return

    # Display statistics for training set
    logger.info("Training Set Statistics:")
    logger.info(f"Number of samples: {len(train_text['input_ids'])}")
    logger.info(f"Label distribution: {torch.bincount(train_labels).tolist()}")
    logger.info(f"Average token length: {torch.tensor([len(tokens) for tokens in train_text['input_ids']]).float().mean().item():.2f}")

    # Display statistics for test set
    logger.info("Test Set Statistics:")
    logger.info(f"Number of samples: {len(test_text['input_ids'])}")
    logger.info(f"Label distribution: {torch.bincount(test_labels).tolist()}")
    logger.info(f"Average token length: {torch.tensor([len(tokens) for tokens in test_text['input_ids']]).float().mean().item():.2f}")


if __name__ == "__main__":
    # Replace with your actual processed data directory and dataset name
    proc_dir = Path("path/to/processed_data")  # Replace with the base directory for processed data
    dataset_name = "your_dataset_name"  # Replace with the dataset name or identifier

    # Call the function
    data_statistics(proc_dir, dataset_name)
