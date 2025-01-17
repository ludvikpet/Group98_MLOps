import torch
from src.cleaninbox.data import text_dataset
from src.cleaninbox.model import BertTypeClassification
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import hydra

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path=to_absolute_path("configs"), config_name="config", version_base="1.1")
def evaluate(cfg: DictConfig) -> None:
    print("Starting evaluation...")
    print(f"Using model checkpoint: {cfg.basic.model_ckpt}")
    print(f"Loading configuration from Hydra-composed configs.")

    # Initialize and load model
    try:
        print(f"Initializing model: {cfg.model.name}")
        model_name = cfg.model.name
        num_classes = cfg.dataset.num_labels  # Use num_labels from the dataset section
        model = BertTypeClassification(model_name, num_classes).to(DEVICE)
        print(f"Loading model weights from {cfg.basic.model_ckpt}")
        model.load_state_dict(torch.load(to_absolute_path(cfg.basic.model_ckpt), map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    print("Composed Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Load test dataset
    try:
        print("Loading test dataset...")
        _, _, test_dataset = text_dataset(
            val_size=cfg.dataset.val_size,
            proc_path=to_absolute_path(cfg.basic.proc_path),
            dataset_name=cfg.dataset.name,
            seed=cfg.experiment.hyperparameters.seed  # Correct reference to seed
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=cfg.experiment.hyperparameters.batch_size  # Correct reference to batch_size
        )

        print(f"Dataloader created with batch size {cfg.experiment.hyperparameters.batch_size}.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Evaluate model
    print("Starting evaluation loop...")
    model.eval()
    correct = 0
    total = 0
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, token_type_ids, attention_mask, labels) in enumerate(test_dataloader):
                print(f"Evaluating batch {batch_idx + 1}...")
                input_ids = input_ids.to(DEVICE)
                token_type_ids = token_type_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(input_ids, attention_mask, token_type_ids)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    # Calculate accuracy
    if total > 0:
        accuracy = correct / total
        print(f"Evaluation completed. Accuracy: {accuracy * 100:.2f}%")
    else:
        print("No samples found in the test dataset.")


if __name__ == "__main__":
    evaluate()
