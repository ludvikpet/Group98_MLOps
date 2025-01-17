import omegaconf
import pytest
import torch

from cleaninbox.model import BertTypeClassification
from tests import _PROJECT_ROOT, _VOCAB_SIZE


@pytest.mark.parametrize("batch_size",[1,17,33])
def test_model(batch_size: int):
    cfg = omegaconf.OmegaConf.load(f"{_PROJECT_ROOT}/configs/config.yaml")
    # Access properties correctly
    assert cfg.model.name == "huawei-noah/TinyBERT_General_4L_312D", "Unexpected model name. Use a configuration with huawei-noah/TinyBERT_General_4L_312D for now"
    assert cfg.dataset.num_labels == 77, "Incorrect number of classes. Expected 77 classes in the dataset"
    assert isinstance(cfg, omegaconf.DictConfig)
    assert isinstance(cfg.model, omegaconf.DictConfig)
    model = BertTypeClassification(model_name=cfg.model.name, num_classes = cfg.dataset.num_labels)
    #now forward a fake batch of data
    seq_len = 128
    input_ids = torch.randint(0, _VOCAB_SIZE, (batch_size, seq_len))  # Vocabulary size of 30522 (e.g., BERT)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    logits = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    assert logits.shape == (batch_size, cfg.dataset.num_labels), "Incorrect output shape. Expected (batch_size, num_labels)"

