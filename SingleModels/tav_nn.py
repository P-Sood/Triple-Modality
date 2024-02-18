import os
import sys

sys.path.insert(0, "/".join(os.getcwd().split("/")[:-2]))
__package__ = "SingleModels"
from utils.trainer import Trainer
from transformers import logging

logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")

from .models.tav import TAVForMAE_HDF5, collate_batch
import wandb
from utils.uni_data_loaders import TextAudioVideoDataset
import pandas as pd
import torch
import numpy as np
from utils.global_functions import arg_parse, Metrics, MySampler, NewCrossEntropyLoss, set_seed
from torch.utils.data import DataLoader


class BatchCollation:
    def __init__(self, must) -> None:
        self.must = must

    def __call__(self, batch):
        return collate_batch(batch, self.must)


def prepare_dataloader(
    df,
    dataset,
    batch_size,
    label_task,
    epoch_switch,
    pin_memory=True,
    num_workers=0,
    check="train",
    accum=False,
    sampler = None,
):
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each
    """
    must = True if "must" in str(dataset).lower() or "urfunny" in str(dataset).lower() else False
    dataset = TextAudioVideoDataset(
        df, 
        dataset, 
        batch_size = 1 if sampler == "Both" else batch_size, 
        feature_col1="audio_path", 
        feature_col2="video_path", 
        feature_col3="text", 
        label_col=label_task, 
        timings="timings", 
        accum=accum, 
        check=check,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=True,
        collate_fn = BatchCollation(must),
        
    )

    return dataloader


def runModel(accelerator, df_train, df_val, df_test, param_dict, model_param):
    """
    Start by getting all the required values from our dictionary
    Then when all the stuff is done, we start to apply our multi-processing to our model and start it up
    """
    device = accelerator
    
    batch_size = param_dict["batch_size"]
    id2label = param_dict["id2label"]
    label_task = param_dict["label_task"]
    epoch_switch = param_dict["epoch_switch"]

    num_labels = model_param["output_dim"]
    dataset = model_param["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    Metric = Metrics(num_classes=num_labels, id2label=id2label, rank=device)
    
    df_train = prepare_dataloader(
        df_train, dataset, batch_size, label_task, epoch_switch, check="train"
    )
    df_val = prepare_dataloader(
        df_val, dataset, batch_size, label_task, epoch_switch, check="val"
    )
    df_test = prepare_dataloader(
        df_test, dataset, batch_size, label_task, epoch_switch, check="test"
    )

    model = TAVForMAE_HDF5(model_param).to(device)
    wandb.watch(model, log="all")
    

    trainer = Trainer(big_batch=3 , num_steps=1)
    
    
    trainer.evaluate(model, df_val, Metric , name = "val")
    trainer.evaluate(model, df_test, Metric , name = "test")
    trainer.evaluate(model, df_train, Metric , name = "train")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    project_name = "MLP_test_text"
    config = arg_parse(project_name)

    wandb.init(entity="ddi", config=config)
    config = wandb.config

    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    set_seed(config.seed)

    param_dict = {
        "epoch": config.epoch,
        "patience": config.patience,
        "lr": config.learning_rate,
        "clip": config.clip,
        "batch_size": config.batch_size,
        "weight_decay": config.weight_decay,
        "T_max": config.T_max,
        "seed": config.seed,
        "label_task": config.label_task,
        "epoch_switch": config.epoch_switch,
        "sampler": config.sampler,
        "LOSS": config.LOSS,
    }

    df = pd.read_pickle(f"{config.dataset}.pkl")
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    df_val = df[df["split"] == "val"]

    if param_dict["label_task"] == "emotion":
        number_index = "emotion"
        label_index = "emotion_label"
    elif param_dict["label_task"] == "sarcasm":
        number_index = "sarcasm"
        label_index = "sarcasm_label"
        df = df[df["context"] == False]
    elif param_dict["label_task"] == "content":  # Needs this to be content too not tiktok
        number_index = "content"
        label_index = "content_label"
    else:
        number_index = "humour"
        label_index = "humour_label"
        df = df[df["context"] == False]

    """
    Due to data imbalance we are going to reweigh our CrossEntropyLoss
    To do this we calculate 1 - (num_class/len(df)) the rest of the functions are just to order them properly and then convert to a tensor
    """

    # weights = torch.Tensor(list(dict(sorted((dict(1 - (df[number_index].value_counts()/len(df))).items()))).values()))
    weights = torch.sort(
        torch.Tensor(
            list(
                dict(
                    sorted(
                        (dict(1 / np.sqrt((df[number_index].value_counts()))).items())
                    )
                ).values()
            )
        )
    ).values
    weights = weights / weights.sum()
    if "iemo" in config.dataset.lower():
        weights = torch.Tensor([weights[1], weights[0] , weights[3] , weights[2] , weights[4] , weights[5]])
        #This is because the text model has different labels then what we use, which is weird but it is what it is
    label2id = (
        df.drop_duplicates(label_index).set_index(label_index).to_dict()[number_index]
    )
    id2label = {v: k for k, v in label2id.items()}

    model_param = {
        "output_dim": len(weights),
        "dropout": config.dropout,
        "dataset": config.dataset,
        "LOSS": config.LOSS,
    }
    param_dict["weights"] = weights
    param_dict["label2id"] = label2id
    param_dict["id2label"] = id2label

    print(
        f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {config.dataset} , with df = {len(df)} \n "
    )
    runModel("cuda" if torch.cuda.is_available() else "cpu", df_train, df_val, df_test, param_dict, model_param)


if __name__ == "__main__":
    main()