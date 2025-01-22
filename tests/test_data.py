
import os

import omegaconf
import pytest
import torch

from cleaninbox.data import text_dataset
from tests import _PROJECT_ROOT, _PATH_DATA

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_text_dataset():
    """
    Unit tests the dataset defined in configs/config.yaml, ie. default dataset in hydraconfig.
    Checks:
        dataset length of train and test split to be corresponding to dataset documentation
        shapes of returned tensors from the dataset
        labels being corresponding to the dataset documentation
    Skips if:
        directory root/data does not exist
    """

    cfg = omegaconf.OmegaConf.load(f"{_PROJECT_ROOT}/configs/config.yaml") #loads default hydra config

    N_CLASSES = cfg.dataset.num_labels
    seed = 45
    train, val, test = text_dataset(cfg.dataset.val_size, cfg.basic.proc_path, cfg.dataset.name, seed)

    assert isinstance(train,torch.utils.data.Subset)
    assert isinstance(val,torch.utils.data.Subset)
    assert isinstance(test,torch.utils.data.TensorDataset)

    split = len(train) + len(val)
    N_VAL_SAMPLES = int(split * cfg.dataset.val_size)
    N_TRAIN_SAMPLES = split - N_VAL_SAMPLES
    N_TEST_SAMPLES = cfg.dataset.num_test_samples

    assert len(train)==N_TRAIN_SAMPLES, "obtained train dataset length not corresponding to length in configuration"
    assert len(val)==N_VAL_SAMPLES, "obtained val dataset length not corresponding to length in configuration"
    assert len(test)==N_TEST_SAMPLES, "obtained test dataset length not corresponding to length in configuration"

    #test dataset sequence lengths
    for dataset in [train,test]:
        for input_ids, token_type_ids, attention_mask, labels in dataset:
            assert input_ids.shape==token_type_ids.shape==attention_mask.shape, "mismatch in sequence length"
            assert labels.dim() == 0, "Labels should be scalars"
            assert labels in range(N_CLASSES), "Detected labels outside of configuration number of classes."

    train_targets = torch.unique(train.dataset.tensors[-1][train.indices])
    # train_targets = torch.unique(train.tensors[-1])
    assert (train_targets==torch.arange(0,N_CLASSES)).all(), "Not all train targets are represented in train targets"
    test_targets = torch.unique(test.tensors[-1])
    assert (test_targets==torch.arange(0,N_CLASSES)).all(), "Not all test targets are represented in test targets"

