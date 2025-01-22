import omegaconf
import pytest
import torch
from torch.optim import Adam

from cleaninbox.model import BertTypeClassification
from tests import _PROJECT_ROOT, _VOCAB_SIZE


## Fixtures
@pytest.fixture
def config():
    """Pytest fixture which loads the configuration file"""
    cfg = omegaconf.OmegaConf.load(f"{_PROJECT_ROOT}/configs/tests/test.yaml")
    return cfg

@pytest.fixture
def model(config):
    """Pytest fixture to initialize the model."""
    return BertTypeClassification(model_name=config.model.name, num_classes=config.dataset.num_labels)

@pytest.fixture
def mock_data():
    """Fixture to generate mock input data for the model."""
    def _generate(batch_size, input_size):
        input_ids = torch.randint(0, _VOCAB_SIZE, (batch_size, input_size))  # Vocabulary size of 30522 (e.g., BERT)
        token_type_ids = torch.zeros((batch_size, input_size), dtype=torch.long)
        attention_mask = torch.ones((batch_size, input_size), dtype=torch.long)
        return input_ids, token_type_ids, attention_mask
    return _generate

## The test class
class TestModel:
    '''The TestModel class organizes all tests related to the model.'''

    def test_model_config(self,config):
        assert config.model.name == "huawei-noah/TinyBERT_General_4L_312D", "Unexpected model name. Use a configuration with huawei-noah/TinyBERT_General_4L_312D for now"
        assert config.dataset.num_labels == 77, "Incorrect number of classes. Expected 77 classes in the dataset"

    @pytest.mark.parametrize("batch_size", [1, 17, 33])
    @torch.no_grad()
    def test_model_output_shape(self, config, model, mock_data, batch_size: int):
        """Test that the model produces the correst output shape"""
        input_ids, token_type_ids, attention_mask = mock_data(batch_size, config.dataset.input_size)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        assert logits.shape == (batch_size, config.dataset.num_labels), "Incorrect output shape. Expected (batch_size, num_labels)"

    def test_device_compatibility(self, model):
        """Test that model can be moved to device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for param in model.parameters():
            assert param.device == device, f"Model parameter not on device {device}"
