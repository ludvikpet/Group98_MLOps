from pathlib import Path
import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger
from typing import Optional
from hydra.utils import to_absolute_path

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../../configs", config_name="config.yaml", version_base="1.1")
def compute_statistics(cfg: DictConfig) -> None:
    """
    Compute dataset statistics using Hydra configuration.

    Args:
        cfg (DictConfig): Configuration object from Hydra.
    """
    proc_path = Path(cfg.basic.proc_path) / Path(cfg.dataset.name).stem

    logger.info(f"Processed data path: {proc_path}")

    try:
        # Load processed data
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
    compute_statistics()