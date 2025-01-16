from pathlib import Path
import json

from datasets import load_dataset 
import typer
from torch.utils.data import Dataset, TensorDataset
import datasets
from loguru import logger
from transformers import BertTokenizer
import torch 
from hydra.utils import to_absolute_path #for resolving paths as originally for loading data
##

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    @logger.catch(level="ERROR")
    def preprocess(self, proc_dir: Path, dset_name: str, model_name: str) -> None:
        dataset = self.download_data(dset_name)
        train_text_l, train_labels_l = dataset["train"]["text"], dataset["train"]["label"]
        test_text_l, test_labels_l = dataset["test"]["text"], dataset["test"]["label"]
        
        tokenizer = BertTokenizer.from_pretrained(model_name)


        #tokenize data 
        
        train_text = self.tokenize_data(train_text_l,tokenizer) # N x maxSeqLen 
        test_text = self.tokenize_data(test_text_l,tokenizer) # N x maxSeqLen

        train_labels = torch.tensor(train_labels_l).long()
        test_labels = torch.tensor(test_labels_l).long()

        torch.save(train_text, proc_dir / "train_text.pt")
        torch.save(train_labels, proc_dir / "train_labels.pt")
        torch.save(test_text, proc_dir / "test_text.pt")
        torch.save(test_labels, proc_dir / "test_labels.pt")

        raw_dir = self.data_path #lazy
        with open(raw_dir/"train_text.json", 'w') as f: 
            json.dump(train_text_l,f)

        with open(raw_dir/"train_labels.json", 'w') as f: 
            json.dump(train_labels_l,f)

        with open(raw_dir/"test_text.json", 'w') as f: 
            json.dump(test_text_l,f)

        with open(raw_dir/"test_labels.json", 'w') as f: 
            json.dump(test_labels_l,f)

        return None


    @logger.catch(level="ERROR")
    def download_data(self, dset_name: str) -> datasets.dataset_dict.DatasetDict:
        logger.info(f"Collecting and unpacking dataset {dset_name}.")
        dataset = load_dataset(dset_name,trust_remote_code=True)
        return dataset

    def tokenize_data(self, text: list, tokenizer: BertTokenizer) -> torch.Tensor:
        encoding = tokenizer(text,# List of input texts
        padding=True,              # Pad to the maximum sequence length
        truncation=False,           # Truncate to the maximum sequence length if necessary
        return_tensors='pt',      # Return PyTorch tensors
        add_special_tokens=True    # Add special tokens CLS and SEP <- possibly uneeded 
        )
        return encoding


def text_dataset():
    proc_path = "data/processed/"
    proc_path = to_absolute_path(proc_path)+"/"
    train_text = torch.load(proc_path + "train_text.pt")
    train_labels = torch.load(proc_path + "train_labels.pt")
    test_text = torch.load(proc_path + "test_text.pt")
    test_labels = torch.load(proc_path + "test_labels.pt")
    train = TensorDataset(train_text["input_ids"], train_text["token_type_ids"], train_text["attention_mask"],train_labels)
    test = TensorDataset(test_text["input_ids"], test_text["token_type_ids"], test_text["attention_mask"], test_labels)

    return train, test


def preprocess(raw_data_path: Path, output_folder: Path, dataset_name: str, model: str) -> None:
    logger.info("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(proc_dir=output_folder,dset_name=dataset_name,model_name=model)



if __name__ == "__main__":
    #typer.run(preprocess)
    _, _ = text_dataset()
    #dataset = load_dataset("PolyAI/banking77",trust_remote_code=True)
    #raw_data_path = Path("data/raw")
    #output_folder = Path("data/processed")
    #dataset = "PolyAI/banking77"
    #model = "huawei-noah/TinyBERT_General_4L_312D"
    #preprocess(raw_data_path,output_folder,dataset,model)
    
    
    # Load model directly
    #from transformers import AutoModel
    #model = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")  
    #print(model)
    #print("end")
