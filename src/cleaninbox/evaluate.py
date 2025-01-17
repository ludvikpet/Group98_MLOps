import torch
import typer
from src.cleaninbox.data import text_dataset
from src.cleaninbox.model import BertTypeClassification
from omegaconf import OmegaConf

# Determine the device for computation (CUDA or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_checkpoint: str, config_path: str) -> None:
    """
    Evaluate a trained BERT-based model for classifying email labels.

    Args:
        model_checkpoint (str): Path to the saved model checkpoint.
        config_path (str): Path to the configuration file.

    Returns:
        None: Prints the evaluation metrics (e.g., accuracy).
    """
    print("Starting evaluation...")
    print(f"Using model checkpoint: {model_checkpoint}")
    print(f"Loading configuration from: {config_path}")

    # Load configuration
    cfg = OmegaConf.load(config_path)

    # Initialize the model
    model = BertTypeClassification(cfg.model.name, cfg.model.num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    # Load the test dataset
    _, _, test_dataset = text_dataset(
        val_size=cfg.dataset.val_size,
        proc_path=cfg.basic.proc_path,
        dataset_name=cfg.dataset.name,
        seed=cfg.experiment.hyperparameters.seed
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.experiment.hyperparameters.batch_size)

    # Switch model to evaluation mode
    model.eval()

    # Initialize counters for accuracy computation
    correct = 0
    total = 0

    # Disable gradient calculations for evaluation
    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, labels in test_dataloader:
            # Move inputs and labels to the computation device
            input_ids = input_ids.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass to compute logits
            logits = model(input_ids, attention_mask, token_type_ids)

            # Predicted labels are the indices of the maximum logit
            predictions = torch.argmax(logits, dim=1)

            # Update the counters
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Compute and print accuracy
    accuracy = correct / total
