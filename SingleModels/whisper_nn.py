import os
import sys

sys.path.insert(0, "/".join(os.getcwd().split("/")[:-2]))
__package__ = "SingleModels"

from transformers import logging

logging.set_verbosity_error()
import warnings

warnings.filterwarnings("ignore")

from .models.whisper import collate_batch , WhisperForEmotionClassification 
import wandb
from utils.uni_data_loaders import WhisperDataset
import pandas as pd
import torch
import numpy as np
from utils.global_functions import arg_parse, Metrics, MySampler, NewCrossEntropyLoss, set_seed
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils.trainer import Trainer
    
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
    print(f"Are we running on mustard? {must}", flush=True) 
    dataset = WhisperDataset(
        df, dataset, batch_size = 1 if (sampler == "Both" or sampler == "Iter_Accum") else batch_size, 
        feature_col="audio_path", label_col=label_task , accum=accum ,check=check
    )

    if check == "train":
        df = df[df['context'] == False] if must else df
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
            shuffle=True,
            collate_fn=BatchCollation(must),
        )

    return dataloader


def runModel(accelerator, df_train, df_val, df_test, param_dict, model_param):
    """
    Start by getting all the required values from our dictionary
    Then when all the stuff is done, we start to apply our multi-processing to our model and start it up
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
    epoch_switch = param_dict["epoch_switch"]
    sampler = param_dict["sampler"]
    LOSS = param_dict["LOSS"]

    num_labels = model_param["output_dim"]
    dataset = model_param["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loss == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif loss == "NewCrossEntropy":
        criterion = NewCrossEntropyLoss(
            class_weights=weights.to(device), epoch_switch=epoch_switch
        ).to(device)
    elif loss == "WeightedCrossEntropy":
        criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device)
        ).to(device)

    print(loss, flush=True)
    Metric = Metrics(num_classes=num_labels, id2label=id2label, rank=device)
    df_train_accum = prepare_dataloader(
        df_train, dataset, batch_size, label_task, epoch_switch, check="train", accum=True, sampler = sampler )
    df_train_no_accum = prepare_dataloader(
        df_train,
        dataset,
        batch_size,
        label_task,
        epoch_switch,
        check="train",
        accum=False, 
        sampler = sampler )
    df_val = prepare_dataloader(
        df_val, dataset, batch_size, label_task, epoch_switch, check="val", sampler  = sampler )
    df_test = prepare_dataloader(
        df_test, dataset, batch_size, label_task, epoch_switch, check="test", sampler = sampler )

    model = WhisperForEmotionClassification(model_param).to(device)
    
    wandb.watch(model, log="all")
   

    trainer = Trainer(big_batch=3 , num_steps=1,  LOSS = LOSS)
    
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
    wandb.finish()


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

    s = param_dict['sampler']
    if s == "Weighted":
        param_dict['loss'] = "CrossEntropy"
        
    elif s == "Iterative" or s == "Iter_Accum":
        param_dict['loss'] = "WeightedCrossEntropy"
        
    elif (s == "Both" or s == "Both_NoAccum") :
        param_dict['loss'] = "NewCrossEntropy"

    

    df = pd.read_pickle(f"{config.dataset}.pkl")
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    df_val = df[df["split"] == "val"]

    if param_dict["label_task"] == "emotion":
        number_index = "emotion"
        label_index = "emotion_label"
        # df = df[df["sentiment_label"] != "Neutral"] if "mosei" in config.dataset else df
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
    }
    param_dict["weights"] = weights
    param_dict["label2id"] = label2id
    param_dict["id2label"] = id2label

    print(
        f" in main \n param_dict = {param_dict} \n model_param = {model_param} \n df {config.dataset} , with df = {len(df)} \n "
    )
    runModel("cuda", df_train, df_val, df_test, param_dict, model_param)


if __name__ == "__main__":
    main()
