import os
import json
import omegaconf
import pytest
import torch

from src.cleaninbox.data import text_dataset
from tests import _PROJECT_ROOT, _PATH_DATA
from loguru import logger

logger.add("logs/test_data.log", level="INFO")

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_text_dataset():
    """
    Unit tests the dataset defined in configs/config.yaml, i.e., default dataset in Hydra config.
    Checks:
        - Dataset length of train, val, and test splits corresponds to dataset documentation.
        - Shapes of returned tensors from the dataset.
        - Labels correspond to the dataset documentation.
        - Processed directory exists and contains required files.
    Skips if:
        - Root/data directory does not exist.
    """
    logger.info("Starting dataset tests...")

    cfg = omegaconf.OmegaConf.load(f"{_PROJECT_ROOT}/configs/config.yaml")  # Load default Hydra config

    # Validate dataset configuration
    assert cfg.dataset.name, "Dataset name is not defined in the config!"
    assert cfg.basic.raw_path, "Raw path is not defined in the config!"
    assert cfg.basic.proc_path, "Processed path is not defined in the config!"

    # Dataset properties
    N_CLASSES = cfg.dataset.num_labels
    seed = 45

    # Load datasets
    train, val, test = text_dataset(cfg.dataset.val_size, cfg.basic.proc_path, cfg.dataset.name, seed)

    # Validate dataset types
    assert isinstance(train, torch.utils.data.Subset), "Train dataset is not a Subset!"
    assert isinstance(val, torch.utils.data.Subset), "Validation dataset is not a Subset!"
    assert isinstance(test, torch.utils.data.TensorDataset), "Test dataset is not a TensorDataset!"

    # Validate dataset lengths
    split = len(train) + len(val)
    N_VAL_SAMPLES = int(split * cfg.dataset.val_size)
    N_TRAIN_SAMPLES = split - N_VAL_SAMPLES
    N_TEST_SAMPLES = cfg.dataset.num_test_samples

    assert len(train) == N_TRAIN_SAMPLES, "Train dataset length mismatch with configuration."
    assert len(val) == N_VAL_SAMPLES, "Validation dataset length mismatch with configuration."
    assert len(test) == N_TEST_SAMPLES, "Test dataset length mismatch with configuration."

    # Validate sequence lengths and labels
    for dataset in [train, test]:
        for input_ids, token_type_ids, attention_mask, labels in dataset:
            assert input_ids.shape == token_type_ids.shape == attention_mask.shape, "Mismatch in sequence lengths."
            assert labels.dim() == 0, "Labels should be scalars."
            assert 0 <= labels < N_CLASSES, "Detected labels outside the valid class range."

    # Validate class representation in datasets
    train_targets = torch.unique(train.dataset.tensors[-1][train.indices])
    assert (train_targets == torch.arange(0, N_CLASSES)).all(), "Not all train targets are represented."

    test_targets = torch.unique(test.tensors[-1])
    assert (test_targets == torch.arange(0, N_CLASSES)).all(), "Not all test targets are represented."

    logger.info("Dataset tests passed successfully.")

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Processed data files not found")
def test_processed_data():
    """
    Tests the integrity of processed data files.
    Ensures:
        - Processed data directory exists.
        - Required files are present.
        - Label dictionary matches expected class count.
    """
    logger.info("Starting processed data tests...")

    cfg = omegaconf.OmegaConf.load(f"{_PROJECT_ROOT}/configs/config.yaml")
    proc_dir = os.path.join(cfg.basic.proc_banking_data)

    # Check processed directory
    assert os.path.exists(proc_dir), f"Processed directory does not exist: {proc_dir}"

    # Check required files
    required_files = ["train_text.pt", "train_labels.pt", "test_text.pt", "test_labels.pt", "label_strings.json"]
    for file in required_files:
        file_path = os.path.join(proc_dir, file)
        assert os.path.exists(file_path), f"Required file missing: {file_path}"

    # Validate label dictionary
    label_dict_path = os.path.join(proc_dir, "label_strings.json")
    with open(label_dict_path, "r") as f:
        label_dict = json.load(f)

    assert len(label_dict) == cfg.dataset.num_labels, "Mismatch between label dictionary and expected class count!"

    logger.info("Processed data tests passed successfully.")
