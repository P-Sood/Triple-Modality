import sys
import os

sys.path.insert(0, "/".join(os.getcwd().split("/")[:-2]))
__package__ = "SingleModels"
from .train_model.audio_training import train_audio_network, evaluate_audio
from .models.audio import collate_batch, Wav2Vec2ForSpeechClassification
import wandb
from utils.data_loaders import Wav2VecAudioDataset
from utils.global_functions import arg_parse, Metrics, MySampler, NewCrossEntropyLoss
import torch
import pandas as pd
from transformers import AutoConfig
import numpy as np


import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from .models.audio import collate_batch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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
    num_workers=4,
    check="train",
    accum=False,
):
    """
    we load in our dataset, and we just make a random distributed sampler to evenly partition our
    dataset on each GPU
    say we have 32 data points, if batch size = 8 then it will make 4 dataloaders of size 8 each
    """
    num_workers = 4
    must = True if "must" in str(dataset).lower() or "urfunny" in str(dataset).lower() else False
    dataset = Wav2VecAudioDataset( df, dataset, batch_size = 1 if sampler == "Both" else batch_size, feature_col="audio_path", label_col=label_task, accum=accum, check=check)

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
    beta = param_dict["beta"]
    weight_decay = param_dict["weight_decay"]
    weights = param_dict["weights"]
    id2label = param_dict["id2label"]
    label_task = param_dict["label_task"]
    model_name = param_dict["model"]
    mask = param_dict["mask"]
    epoch_switch = param_dict["epoch_switch"]

    num_labels = model_param["output_dim"]
    dataset = model_param["dataset"]

    if loss == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss().to(device)

    elif loss == "NewCrossEntropy":
        criterion = NewCrossEntropyLoss(
            class_weights=weights.to(device), epoch_switch=epoch_switch
        ).to(device)

    print(loss, flush=True)
    Metric = Metrics(num_classes=num_labels, id2label=id2label, rank=device)
    df_train_accum = prepare_dataloader(
        df_train, dataset, 1, label_task, epoch_switch, check="train", accum=True
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
        df_val, dataset, batch_size, label_task, epoch_switch, check="val"
    )
    df_test = prepare_dataloader(
        df_test, dataset, batch_size, label_task, epoch_switch, check="test"
    )

    model = Wav2Vec2ForSpeechClassification(model_param).to(device)

    wandb.watch(model, log="all")
    # wandb.watch(PREFormer, log = "all")
    checkpoint = None  # torch.load(f"/home/prsood/projects/ctb-whkchun/prsood/TAV_Train/MAEncoder/aht69be1/lively-sweep-11/7.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # criterion = checkpoint['loss']
    # PREFormer = checkpoint['PREFormer']

    model = train_audio_network(
        model,
        [df_train_no_accum, df_train_accum],
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
        checkpoint,
    )
    evaluate_audio(model, df_test, Metric)


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
        "batch_size": 1,  # config.batch_size ,
        "weight_decay": config.weight_decay,
        "model": config.model,
        "T_max": config.T_max,
        "seed": config.seed,
        "label_task": config.label_task,
        "mask": config.mask,
        "loss": config.loss,
        "beta": config.beta,
        "epoch_switch": config.epoch_switch,
    }

    df = pd.read_pickle(f"{config.dataset}.pkl")
    # df = pd.read_pickle("/home/jupyter/multi-modal-emotion/data/TAV_MELD_bounding_box.pkl")
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
    elif (
        param_dict["label_task"] == "content"
    ):  # Needs this to be content too not tiktok
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
    label2id = (
        df.drop_duplicates(label_index).set_index(label_index).to_dict()[number_index]
    )
    id2label = {v: k for k, v in label2id.items()}

    model_param = {
        "output_dim": len(weights),
        "dropout": config.dropout,
        "early_div": config.early_div,
        "num_layers": config.num_layers,
        "learn_PosEmbeddings": config.learn_PosEmbeddings,
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
