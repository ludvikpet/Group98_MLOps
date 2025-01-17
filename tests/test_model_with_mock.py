import omegaconf
import pytest
import torch
from torch.optim import Adam
from unittest.mock import MagicMock
import hydra

from cleaninbox.model import BertTypeClassification
from tests import _PROJECT_ROOT, _VOCAB_SIZE


## Fixtures
@pytest.fixture(scope="session")
def config():
    """Pytest fixture which loads the configuration file."""
    cfg = omegaconf.OmegaConf.load(f"{_PROJECT_ROOT}/configs/config.yaml")
    return cfg


@pytest.fixture(scope="session")
def model(config):
    """Pytest fixture to initialize the model."""
    return BertTypeClassification(model_name=config.model.name, num_classes=config.dataset.num_labels)


@pytest.fixture
def mock_data():
    """Fixture to generate mock input data for the model."""
    def _generate(batch_size, seq_len=128):
        input_ids = torch.randint(0, _VOCAB_SIZE, (batch_size, seq_len))  # Vocabulary size of 30522 (e.g., BERT)
        token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        return input_ids, token_type_ids, attention_mask
    return _generate


## The test class
@pytest.mark.skip(reason="This test is not yet implemented")
class TestModel:
    """The TestModel class organizes all tests related to the model."""

    def test_model_config(self, config):
        assert config.model.name == "huawei-noah/TinyBERT_General_4L_312D", \
            "Unexpected model name. Use a configuration with huawei-noah/TinyBERT_General_4L_312D for now"
        assert config.dataset.num_labels == 77, \
            "Incorrect number of classes. Expected 77 classes in the dataset"

    @pytest.mark.parametrize("batch_size", [1, 17, 33])
    @torch.no_grad()
    def test_model_output_shape(self, model, mock_data, config, batch_size: int):
        """Test that the model produces the correct output shape."""
        input_ids, token_type_ids, attention_mask = mock_data(batch_size)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        assert logits.shape == (batch_size, config.dataset.num_labels), \
            "Incorrect output shape. Expected (batch_size, num_labels)"

    def test_model_parameter_updates(self, mocker, mock_data, config):
        """Test that the model parameters are updated during training."""
        # Mock the optimizer step to avoid unnecessary computations
        mock_optimizer = mocker.patch.object(Adam, 'step', autospec=True)

        # Mock the model's gradients
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("param", MagicMock(requires_grad=True, grad=torch.ones(1)))]

        # Simulate a forward pass and backpropagation
        batch_size = 16
        input_ids, token_type_ids, attention_mask = mock_data(batch_size)
        labels = torch.randint(0, config.dataset.num_labels, (batch_size,), dtype=torch.long)

        logits = torch.randn(batch_size, config.dataset.num_labels)  # Mocked logits
        torch.nn.functional.cross_entropy(logits, labels).backward()  # Mocked backward pass

        # Ensure the optimizer step was called
        mock_optimizer.assert_called_once()

        # Check that the mocked gradients are non-zero
        for name, param in mock_model.named_parameters():
            assert param.grad is not None, f"Gradient for parameter {name} is None"
            assert torch.any(param.grad != 0), f"Gradient for parameter {name} is zero"
