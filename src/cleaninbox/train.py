#import logging before loguru
import os
import textwrap

import torch
# import typer
from cleaninbox.data import text_dataset, load_label_strings
from cleaninbox.model import BertTypeClassification
from google.cloud import storage
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from dotenv import load_dotenv

from loguru import logger
import wandb
from transformers import AutoModel

"""
Possible fix for the FutureWarning regarding torch.load: https://github.com/openai/whisper/pull/2451#issuecomment-2516971867
import functools
whisper.torch.load = functools.partial(whisper.torch.load, weights_only=True)
"""

def save_model_to_bucket(input_path, output_path) -> None:
    client = storage.Client()
    bucket = client.bucket("banking77")
    blob = bucket.blob(output_path)
    blob.upload_from_filename(input_path)



@hydra.main(version_base="1.1", config_path="../../configs",config_name="config")
def train(cfg: DictConfig):
    """Train a model on banking77."""
    environment_cfg = cfg.environment
    if(environment_cfg.run_in_cloud==True):
        cloud_model_path = "/models/model.pth" #used to register trained model outside of docker container
        cfg.basic.proc_path = ("/gcs/banking77"+cfg.basic.proc_path).replace(".","") #append cloud bucket to path string format
        cfg.basic.raw_path = ("/gcs/banking77"+cfg.basic.raw_path).replace(".","") #append cloud bucket to path string format

    print(OmegaConf.to_yaml(cfg))

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add a log file to the logger
    hyperparameters = cfg.experiment.hyperparameters
    lr = hyperparameters.lr
    batch_size = hyperparameters.batch_size
    epochs = hyperparameters.epochs
    seed = hyperparameters.seed
    num_samples = hyperparameters.num_samples

    model_name = cfg.model.name

    logger.info(f"Fetching model {model_name}")
    model = BertTypeClassification(model_name,num_classes=cfg.dataset.num_labels) #should be read from dataset config
    print(model)

#join logger and hydra log
    logger.add(os.path.join(hydra_path, "my_logger_hydra.log"))
    logger.info(cfg)
    logger.info("Training day and night")
    #handle wandb based on config
    if(environment_cfg.log_wandb==True):
        #load_dotenv()
        #following could be optimized using specific wandb config with entity and project fields.
        wandb.init(entity="cleaninbox_02476",project="banking77",config=dict(hyperparameters)) #inherits API key from environment by directly passing it when running container. Now using vertex injection. Note: other args should in principle also be inherited from config json.
        logger.debug("using wandb")
    #wandb.login(key=os.getenv("WANDB_API_KEY"))
    torch.manual_seed(seed)

    logger.info(f"{lr=}, {batch_size=}, {epochs=}")

    model = model.to(DEVICE)
    train_set, _, _ = text_dataset(cfg.dataset.val_size, cfg.basic.proc_path, cfg.dataset.name, seed) #need to be variable based on cfg
    string_labels = load_label_strings(cfg.basic.proc_path,cfg.dataset.name)


    if num_samples!=-1: #allow to only use a subset.
        N_SAMPLES = len(train_set)
        num_samples = min(num_samples, N_SAMPLES)
        subset_sizes = [num_samples, N_SAMPLES-num_samples] #train-subset and "other" subset (unused)
        train_set, _ = random_split(train_set, subset_sizes, generator=torch.Generator().manual_seed(seed))
    logger.debug(f"training on {len(train_set)} samples. The rest is discarded.")
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    del train_set

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, labels = input_ids.to(DEVICE), token_type_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            #note: need to add classification head
            logits = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            loss = criterion(logits, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            statistics["train_accuracy"].append(accuracy)


        epoch_loss = running_loss.item() / N_SAMPLES
        logger.info(f"Epoch {epoch}: {epoch_loss}")
        statistics["train_loss"].append(epoch_loss)
        if(environment_cfg.log_wandb==True):
            wandb.log({"train_loss":epoch_loss})
            #make plot sample
            if(epoch%5 == 0): #plot 4 samples of the training-set and corresponding top-5 distribution
                logits = logits[:4]
                labels = labels[:4]
                topk_ = torch.topk(logits,dim=1,k=5)
                sentences = model.decode_input_ids(input_ids[:4]) #returns a list of sentences
                top_probs = topk_.values
                top_labs = topk_.indices
                fig, axes = plt.subplots(1,4,figsize=(16,4))
                xticks = [x for x in range(5)]
                for j, (ax,probs,top_labs,labs) in enumerate(zip(axes.flat,top_probs,top_labs,labels)):
                    labs = labs.item()
                    top_labs = top_labs.detach().numpy()
                    probs = probs.detach().numpy()
                    string_labels_xaxis = [string_labels[str(idx)] for idx in top_labs]
                    # Wrap tick labels for better readability
                    wrapped_labels = ['\n'.join(textwrap.wrap(label, width=10)) for label in string_labels_xaxis] #nice todo: only split on underscores, but requires more time than i have atm
                    wrapped_title = '\n'.join(textwrap.wrap(sentences[j], width=30))
                    colors = ["tab:orange" if idx==labs else "tab:blue" for idx in top_labs] #plot blue if correct label is present in topk
                    ax.bar(x=xticks,height=probs,align="center",color=colors)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(wrapped_labels, rotation=45, ha='center')
                    ax.set_title(wrapped_title)
                plt.tight_layout()
                wandb.log({"training predictions": wandb.Image(fig)})

        plt.close()

    logger.info("Finished training")

    if cfg.model.save_model:
        torch.save(model.state_dict(), f"{os.getcwd()}/model.pth") #save to hydra output (hopefully using chdir true)
        logger.info(f"Saved model to: f{os.getcwd()}/model.pth")
        if environment_cfg.run_in_cloud==True:
            save_model_to_bucket(input_path=f"{os.getcwd()}/model.pth",output_path=cloud_model_path)
            logger.info(f"Saved model to cloud: {cloud_model_path}")
            #register model to vertex model registry <- didn't work, we stopped this for now
            # aiplatform.init(project="cleaninbox-448011", location="europe-west1")
            # model = aiplatform.Model.upload(
            #     display_name="banking77-classifier",
            #     artifact_uri=f"gs://banking77{cloud_model_path}",
            #     serving_container_image_uri=None,  # No inference container for now
            # )


    if(environment_cfg.log_wandb==True):
        artifact = wandb.Artifact(name="model",type="model")
        artifact.add_file(local_path=f"{os.getcwd()}/model.pth",name="model.pth")
        artifact.save()
        logger.info(f"Saved model artifact to f{os.getcwd()}/model.pth")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    #fig.savefig("reports/figures/training_statistics.png") <- with no hydra configuration
    fig.savefig(f"{os.getcwd()}/training_statistics.png")
    if(environment_cfg.log_wandb==True):
        wandb.log({"training statistics":wandb.Image(fig)}) #try to log an image

if __name__ == "__main__":
    train()
    #typer.run(train)
