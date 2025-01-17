import pytest
import omegaconf
from unittest.mock import MagicMock
import torch
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from torch.optim import Adam
from loguru import logger
from transformers import AutoModel
import wandb
from cleaninbox.model import BertTypeClassification
from cleaninbox.train import train
from tests import _PROJECT_ROOT, _VOCAB_SIZE, _TEST_ROOT

## Fixtures
@pytest.fixture
def config():
    """Pytest fixture which loads the configuration file."""
    cfg = omegaconf.OmegaConf.load(f"{_PROJECT_ROOT}/configs/config.yaml")
    return cfg

@pytest.fixture
def model(config):
    """Pytest fixture to initialize the model."""
    return BertTypeClassification(model_name=config.model.name, num_classes=config.dataset.num_labels)

@pytest.fixture
def mock_data(config):
    """Fixture for a mock dataset."""
    def _generate(size=config.unittest.train.batch_size, seq_len=128):
        data = torch.randint(0, _VOCAB_SIZE, (size, seq_len))
        labels = torch.randint(0, config.dataset.num_labels, (size,))
        return data, labels
    return _generate

@pytest.fixture
def mock_wandb(mocker):
    """Mock wandb integration."""
    return mocker.patch("wandb.init", return_value=MagicMock())

@pytest.fixture
def mock_torch_save(mocker):
    """Mock torch.save."""
    return mocker.patch("torch.save", autospec=True)

@pytest.fixture
def mock_hydra_config(mocker):
    """Mock HydraConfig.get to simulate runtime configuration."""
    # Create a mock HydraConfig object
    mock_runtime = MagicMock()
    mock_runtime.output_dir = "./test_outputs"  # Simulate runtime output directory

    mock_hydra_config = MagicMock()
    mock_hydra_config.runtime = mock_runtime

    # Mock HydraConfig.get to return the mocked config
    mocker.patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_hydra_config)

class TestTraining:
    """Unit tests for training.

    The functions:
    - test_loss_functionality,
    - test_logging,
    - test_device_compatibility,
    - test_checkpoint_saving
    test individual components of the training function but do not integrate train.py

    The function:
    - test_train_integration
    tests the entire training pipeline
    """

    @staticmethod
    def test_loss_functionality(config, model, mock_data):
        """Test that the loss function behaves as expected."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=config.unittest.train.lr)
        data, labels = mock_data(config.unittest.train.batch_size)

        logits = model(data)
        loss = criterion(logits, labels)
        assert loss.item() > 0, "Loss should be positive for random initialization"

        # Perform a training step
        loss.backward()
        optimizer.step()

        # Verify gradients are non-zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, "Gradient is None for a parameter"
                assert torch.any(param.grad != 0), f"Gradient for parameter {name} is zero"

    @staticmethod
    @pytest.mark.skip(reason="This unit test is covered by test_train_integration")
    def test_logging(mock_wandb):
        """Test logging functionality."""
        wandb.log({"loss": 0.5})
        assert mock_wandb.assert_called # Assertion

    @staticmethod
    def test_device_compatibility(model):
        """Test that model and data move to the correct device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for param in model.parameters():
            assert param.device == device, f"Model parameter not on device {device}"

    @staticmethod
    @pytest.mark.skip(reason="This unit test is covered by test_train_integration")
    def test_checkpoint_saving(model, mock_torch_save):
      """Test that model checkpoints are saved correctly using mocker."""
      # mock temporary checkpoint path
      checkpoint_path = _TEST_ROOT / "tmp" / "checkpoint.pth"
      mock_save = mock_torch_save()
      # Call the save function
      torch.save(model.state_dict(), checkpoint_path)
      # Assert the save function was called with the correct arguments
      mock_save.assert_called_once_with(model.state_dict(), checkpoint_path)

    @pytest.mark.slow # this test is slow since it actually calls the train function
    @staticmethod
    def test_train_integration(mock_wandb, mock_torch_save, mock_hydra_config):
        """Test the train function end-to-end."""
        # Initialize Hydra
        with initialize(config_path="../configs", version_base="1.1"):
          # Compose the configuration for the test
          cfg = compose(config_name="config")

          # cfg.hydra = {
          #     "run": {"dir": "./test_outputs"},  # Simulate Hydra's runtime directory
          #     "job_logging": {"level": "INFO"},  # Optional: Logging configuration
          # }
          # # Manually set HydraConfig for the test environment
          # HydraConfig.instance().set_config(cfg)
          # HydraConfig.instance().runtime.output_dir = "./test_outputs"
          # # Mock external dependencies
          # mock_wandb = mocker.patch("wandb.init", return_value=MagicMock())
          # mock_wandb_log = mocker.patch("wandb.log")
          # mock_torch_save = mocker.patch("torch.save")
          # mock_fig = mocker.patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), [MagicMock(), MagicMock()]))

          # # Call the train function
          # train(cfg)

          # # Assertions to verify mocks were called
          # mock_wandb.assert_called_once()
          # mock_wandb_log.assert_called()
          # mock_torch_save.assert_called_once()
          # mock_fig.assert_called_once()

          # Mock external dependencies
          mock_wandb = mock_wandb()
          mock_save = mock_torch_save

          # Call the train function
          train(cfg)

          # Verify that logging and wandb calls were made
          assert mock_wandb.assert_called, "wandb.init not called"
          assert mock_save.assert_called, "torch.save not called"
