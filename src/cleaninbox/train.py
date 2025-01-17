#import logging before loguru
import os

import torch
# import typer
from cleaninbox.data import text_dataset
from cleaninbox.model import BertTypeClassification
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

#



@hydra.main(version_base="1.1", config_path="../../configs",config_name="config")
def train(cfg: DictConfig):
    """Train a model on banking77."""

    print(OmegaConf.to_yaml(cfg))

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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
    if(cfg.experiment.logging.log_wandb==True):
        #load_dotenv()
        #following could be optimized using specific wandb config with entity and project fields.
        wandb.init(entity="cleaninbox_02476",project="banking77",config=dict(hyperparameters)) #inherits API key from environment by directly passing it when running container
        logger.debug("using wandb")
    #wandb.login(key=os.getenv("WANDB_API_KEY"))
    torch.manual_seed(seed)

    logger.info(f"{lr=}, {batch_size=}, {epochs=}")

    model = model.to(DEVICE)
    train_set, _, _ = text_dataset(cfg.dataset.val_size, cfg.basic.proc_path, cfg.dataset.name, seed) #need to be variable based on cfg

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
            #let's plot the first 16 images of the first batch with corresponding predictions
            # if(i==0):
            #     images = images.permute(0,2,3,1).detach().numpy()[:16] #need permute for plt plotting
            #     labels = labels.detach().numpy()[:16]
            #     predicted_classes = torch.argmax(logits,dim=1).detach().numpy()[:16]
            #     fig, axes = plt.subplots(4,4)
            #     for j, ax in enumerate(axes.flat):
            #         ax.imshow(images[j])
            #         ax.set_axis_off()
            #         ax.text(3,5, f"{predicted_classes[j]}",color="red",fontweight="bold")
            #     plt.tight_layout(pad=0.0,w_pad=0.0,h_pad=0.0)
            #     fig.suptitle(f"epoch {epoch}",fontweight="bold")
            #     wandb.log({"training predictions": wandb.Image(fig)})
            #     plt.close()
            #     # add a plot of histogram of the gradients
            #     grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
            #     wandb.log({"gradients": wandb.Histogram(grads)})

        epoch_loss = running_loss.item() / N_SAMPLES
        logger.info(f"Epoch {epoch}: {epoch_loss}")
        statistics["train_loss"].append(epoch_loss)
        if(cfg.experiment.logging.log_wandb==True):
            wandb.log({"train_loss":epoch_loss})

    logger.info("Finished training")

    if cfg.model.save_model:
        torch.save(model.state_dict(), f"{os.getcwd()}/model.pth") #save to hydra output (hopefully using chdir true)
        logger.info(f"Saved model to: f{os.getcwd()}/model.pth")

    if(cfg.experiment.logging.log_wandb==True):
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
    if(cfg.experiment.logging.log_wandb==True):
        wandb.log({"training statistics":wandb.Image(fig)}) #try to log an image

if __name__ == "__main__":
    train()
    #typer.run(train)
