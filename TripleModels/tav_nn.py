import os
import sys

sys.path.insert(0, "/".join(os.getcwd().split("/")[:-2]))
__package__ = "TripleModels"

from transformers import logging

logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")

from utils.trainer import Trainer
from models.tav import TAVForMAE, collate_batch
import wandb
from utils.data_loaders import TextAudioVideoDataset
import pandas as pd
import torch
import numpy as np
from utils.global_functions import arg_parse, Metrics, MySampler, NewCrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler


TESTING_PIPELINE = False
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
    Take in pandas dataframe, name of dataset, batch size, label task, whether we are training or testing, or if we are accumulating gradients or not

    If we are training then we create two dataloaders, one for accumulating gradients with iterative sampler and one for regular with weighted sampling

    Otherwise we just create a regular dataloader for val/test that just do shuffle
    """
    must = True if "must" in str(dataset).lower() else False
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

    if check == "train":
        labels = df[label_task].value_counts()
        class_counts = torch.Tensor(
            list(dict(sorted((dict((labels)).items()))).values())
        ).to(int)

        samples_weight = torch.tensor([1 / class_counts[t] for t in dataset.labels])
        print(len(samples_weight))

        if accum:
            sampler = MySampler(
                list(samples_weight),
                len(samples_weight),
                replacement=True,
                epoch=epoch_switch - 1,
                epoch_switch=epoch_switch,
            )
        else:
            sampler = MySampler(
                list(samples_weight),
                len(samples_weight),
                replacement=True,
                epoch=0,
                epoch_switch=epoch_switch,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            sampler=sampler,
            collate_fn=BatchCollation(must),
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn=BatchCollation(must),
        )

    return dataloader


def runModel(accelerator, df_train, df_val, df_test, param_dict, model_param):
    """
    Start by getting all the required values from our dictionary
    Start and init all classes required for training
    """
    device = accelerator
    epoch = param_dict["epoch"]
    lr = param_dict["lr"]
    patience = param_dict["patience"]
    clip = param_dict["clip"]
    T_max = param_dict["T_max"]
    batch_size = param_dict["batch_size"]
    loss = param_dict["loss"]
    weight_decay = param_dict["weight_decay"]
    weights = param_dict["weights"]
    id2label = param_dict["id2label"]
    label_task = param_dict["label_task"]
    model_name = param_dict["model"]
    mask = param_dict["mask"]
    epoch_switch = param_dict["epoch_switch"]
    sampler = param_dict["sampler"]

    num_labels = model_param["output_dim"]
    dataset = model_param["dataset"]

    if loss == "CrossEntropy":
        # criterion = NewCrossEntropyLoss(class_weights=weights.to(device)).to(device) # TODO: have to call epoch in forward function
        criterion = torch.nn.CrossEntropyLoss().to(device)

    elif loss == "NewCrossEntropy":
        # criterion = PrecisionLoss(num_classes = num_labels,weights=weights.to(device)).to(device)
        criterion = NewCrossEntropyLoss(
            class_weights=weights.to(device), epoch_switch=epoch_switch
        ).to(device)
    elif loss == "WeightedCrossEntropy":
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device)
        ).to(device)
        

    print(loss, flush=True)
    Metric = Metrics(num_classes=num_labels, id2label=id2label, rank=device)
    df_train_accum = prepare_dataloader(
        df_train, dataset, batch_size, label_task, epoch_switch, check="train", accum=True , sampler=sampler
    )
    df_train_no_accum = prepare_dataloader(
        df_train,
        dataset,
        batch_size,
        label_task,
        epoch_switch,
        check="train",
        accum=False,
    )
    df_val = prepare_dataloader(
        df_val,
        dataset,
        batch_size,
        label_task,
        epoch_switch,
        check="val" if TESTING_PIPELINE == False else "train",
    )
    df_test = prepare_dataloader(
        df_test,
        dataset,
        batch_size,
        label_task,
        epoch_switch,
        check="test" if TESTING_PIPELINE == False else "train",
    )

    model = TAVForMAE(model_param).to(device)

    wandb.watch(model, log="all")
    
    trainer = Trainer(2**5 , 4)

    model = trainer.train_network(
        model,
        [df_train_no_accum, df_train_accum , sampler],
        df_val,
        criterion,
        lr,
        epoch,
        weight_decay,
        T_max,
        Metric,
        patience,
        clip,
        epoch_switch,
        checkpoint = None,
    )
    
    trainer.evaluate(model, df_test, Metric)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    project_name = "MLP_test_text"
    config = arg_parse(project_name)

    wandb.init(entity="ddi", config=config)
    config = wandb.config

    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    param_dict = {
        "epoch": config.epoch,
        "patience": config.patience,
        "lr": config.learning_rate,
        "clip": config.clip,
        "batch_size": 256, #config.batch_size,
        "weight_decay": config.weight_decay,
        "model": config.model,
        "T_max": config.T_max,
        "seed": config.seed,
        "label_task": config.label_task,
        "mask": config.mask,
        "loss": config.loss,
        "epoch_switch": config.epoch_switch,
        "sampler": config.sampler,
    }
    s = param_dict['sampler']
    l = param_dict['loss']
    if s == "Weighted" and l == "WeightedCrossEntropy":
        print(f"We are not going to learn anything with sampler == {s } and loss == {l}. \nKill it" , flush=True)
        return 0
    elif s == "Iterative" and l == "CrossEntropy":
        print(f"We are not going to learn anything with sampler == {s } and loss == {l}. \nKill it" , flush=True)
        return 0
    elif s == "Iter_Accum" and l == "CrossEntropy":
        print(f"We are not going to learn anything with sampler == {s } and loss == {l}. \nKill it" , flush=True)
        return 0
    elif (s == "Iterative" or s == "Weighted" or s == "Iter_Accum") and l == "NewCrossEntropy":
        print(f"We are not going to learn anything with sampler == {s } and loss == {l}. \nKill it" , flush=True)
        return 0
    # elif (s == "Both_NoAccum" or s == "Both") and (l == "WeightedCrossEntropy" or l == "CrossEntropy"):
    #     print(f"We are not going to learn anything with sampler == {s } and loss == {l}. \nKill it" , flush=True)
    #     return 0
        

    df = pd.read_pickle(f"{config.dataset}.pkl")

    if TESTING_PIPELINE:
        df_train = df[df["split"] == "train"].head(100)
        df_test = df[df["split"] == "train"].head(100)
        df_val = df[df["split"] == "train"].head(100)
    else:
        df_train = df[df["split"] == "train"]
        df_test = df[df["split"] == "test"]
        df_val = df[df["split"] == "val"]

    if param_dict["label_task"] == "sentiment":
        number_index = "sentiment"
        label_index = "sentiment_label"
    elif param_dict["label_task"] == "sarcasm":
        number_index = "sarcasm"
        label_index = "sarcasm_label"
        df = df[df["context"] == False]
    elif param_dict["label_task"] == "content":  # Needs this to be content too not tiktok
        number_index = "content"
        label_index = "content_label"
    else:
        number_index = "emotion"
        label_index = "emotion_label"

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
    label2id = (
        df.drop_duplicates(label_index).set_index(label_index).to_dict()[number_index]
    )
    id2label = {v: k for k, v in label2id.items()}

    print(config.hidden_layers)
    model_param = {
        "output_dim": len(weights),
        "dropout": config.dropout,
        "early_div": config.early_div,
        "num_layers": config.num_layers,
        "num_encoders": config.num_encoders,
        "learn_PosEmbeddings": config.learn_PosEmbeddings,
        "dataset": config.dataset,
        "fusion": config.fusion,
        "hidden_size": int(config.hidden_layers),
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
