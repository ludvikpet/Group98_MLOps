from pathlib import Path

from datasets import load_dataset 
#import typer
from torch.utils.data import Dataset
import datasets
from loguru import logger


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

@logger.catch(level="ERROR")
def download_data(dset_name: str) -> datasets.dataset_dict.DatasetDict:
    logger.info(f"Collecting and unpacking dataset {dset_name}.")
    dataset = load_dataset(dset_name,trust_remote_code=True)
    return dataset

def tensorize_and_save_data(dset_name: str) -> None:
    dataset = download_data(dset_name)
    return 
    


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    logger.info("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    #typer.run(preprocess)
    dataset = load_dataset("PolyAI/banking77",trust_remote_code=True)
    # Load model directly
    from transformers import AutoModel
    model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")  
    print(model)
    print("end")
