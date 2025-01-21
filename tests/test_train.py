from pathlib import Path
import pytest
import omegaconf
from unittest.mock import MagicMock
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from torch.optim import Adam
from cleaninbox.model import BertTypeClassification
from cleaninbox.train import train
from tests import _PROJECT_ROOT, _VOCAB_SIZE, _TEST_ROOT

## Fixtures
@pytest.fixture
def config():
    """Pytest fixture which loads the configuration file."""
    cfg = omegaconf.OmegaConf.load(f"{_PROJECT_ROOT}/configs/tests/test.yaml")
    return cfg

@pytest.fixture
def model(config):
    """Pytest fixture to initialize the model."""
    return BertTypeClassification(model_name=config.model.name, num_classes=config.dataset.num_labels)

@pytest.fixture
def mock_data():
    """Fixture for a mock dataset."""
    def _generate(config, batch_size, input_size):
        data = torch.randint(0, _VOCAB_SIZE, (batch_size, input_size))
        labels = torch.randint(0, config.dataset.num_labels, (batch_size,))
        return data, labels
    return _generate

class TestTraining:
    """Unit tests for training the model."""

    @staticmethod
    def test_parameter_updates(model, mock_data):
        """Test that the loss function behaves as expected."""
        with initialize(config_path="../configs/tests", version_base="1.1"):
          # Compose the configuration for the test
          cfg = compose(config_name="test")

          criterion = torch.nn.CrossEntropyLoss()
          optimizer = Adam(model.parameters(), lr=cfg.experiment.hyperparameters.lr)
          data, labels = mock_data(cfg, cfg.experiment.hyperparameters.batch_size, cfg.dataset.input_size)

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

    @pytest.mark.slow # this test is slow since it actually calls the train function
    @staticmethod
    def test_training(mocker):
        """Test the train function end-to-end."""
        # Create a mock HydraConfig object
        mock_runtime = MagicMock()
        mock_runtime.output_dir = "./test_outputs"  # Simulate runtime output directory

        mock_hydra_config = MagicMock()
        mock_hydra_config.runtime = mock_runtime

        # Mock HydraConfig.get to return the mocked config
        mocker.patch("hydra.core.hydra_config.HydraConfig.get", return_value=mock_hydra_config)

        # Initialize Hydra
        with initialize(config_path="../configs/tests", version_base="1.1"):
            # Compose the configuration for the test
            cfg = compose(config_name="test")

            # Mock wandb and torch.save:
            mock_wandb = mocker.patch("wandb.init", return_value=MagicMock())
            mock_wandb_log = mocker.patch("wandb.log")
            mock_save = mocker.patch("torch.save", autospec=True)

            # Mock optimizer and criterion
            mock_optimizer = mocker.patch('cleaninbox.train.Adam')
            mock_optimizer_instance = MagicMock()
            mock_optimizer.return_value = mock_optimizer_instance
            mock_optimizer_instance.zero_grad = MagicMock()
            mock_optimizer_instance.step = MagicMock()
            mock_criterion = mocker.patch('torch.nn.CrossEntropyLoss')

            # Mock the logger
            mock_logger_add = mocker.patch("loguru.logger.add", return_value=MagicMock())
            mock_logger_info = mocker.patch("loguru.logger.info")
            mock_logger_debug = mocker.patch("loguru.logger.debug")

            # Mock the training statistics figure
            mock_fig = mocker.patch('matplotlib.pyplot.subplots', return_value = (MagicMock(), [MagicMock(), MagicMock()]))

            # Call the train function
            train(cfg)

            # Verify that logging and wandb calls were made
            assert mock_wandb.assert_called, "wandb.init not called"
            assert mock_wandb_log.assert_called, "wandb.log not called"
            assert mock_save.assert_called, "torch.save not called"
            #assert mock_model.called, "Model not called"
            assert mock_criterion.called, "Criterion not called"
            assert mock_optimizer.called, "Optimizer not called"
            assert mock_optimizer_instance.zero_grad.called, "Zero_grad not called"
            assert mock_optimizer_instance.step.called, "Optimizer step not called"

            assert mock_logger_add.called, "Logger add not called"
            assert mock_logger_info.called, "Logger info not called"
            assert mock_logger_debug.called, "Logger debug not called"

            assert mock_fig.called_once(), "Matplotlib figure not created"

            # Check if the log file exists
            log_file_path = f"{mock_hydra_config.runtime.output_dir}/my_logger_hydra.log"
            assert not(Path(log_file_path).exists()), "Log file created. Mock the logger properly."

