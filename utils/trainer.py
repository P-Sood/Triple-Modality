
import os
import wandb
import torch
import warnings
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import logging
from torch.utils.checkpoint import checkpoint
from utils.early_stopping import EarlyStopping
from utils.global_functions import save_model, load_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch import is_tensor

logging.set_verbosity_error()
warnings.filterwarnings("ignore")
# from transformers.optimization import AdamW

class Trainer:
    """ big_batch: if batch > big_batch then we will use checkpointing to save memory.
        num_steps: number of steps to divide an epoch into. Used in conjunction with patience for early stopping
    """
    def __init__(self , big_batch: int , num_steps : int , ) -> None:
        
        self.big_batch = big_batch
        self.num_steps = num_steps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patient_iter = 0    
    
    def get_statistics(self , 
        input: dict, label: np.array, model, criterion, Metric, check="train", epoch=None
    ):
        batch_loss = None
        label = label.to(self.device)
        input = {k:v.to(self.device) if is_tensor(v) else v for k, v in input.items()}
        
        output = model(**input, check=check)

        for k, v in input.items():
            input[k] = v.cpu() if is_tensor(v) else v
            del input[k]

        Metric.update_metrics(torch.argmax(output, dim=1), label)
        if criterion is not None:
            if criterion.__class__.__name__ == "NewCrossEntropyLoss":
                batch_loss = criterion(
                    output, label, epoch=epoch if epoch is not None else 0
                )  # TODO: Turn this on with Sampler
            else:
                batch_loss = criterion(output, label)
        del output
        del label
        return batch_loss


    def get_statistics_big_batch(self , 
        input: dict, label: np.array, model, criterion, Metric, check="train", epoch=None
    ):
        batch_loss = None
        label = label.to(self.device)
        
        input = {k:v.to(self.device) if is_tensor(v) else v for k, v in input.items()}
        
        output = checkpoint(
            model,
            **input,
            check=check,
            use_reentrant=False,
        )
        for k, v in input.items():
            input[k] = v.cpu() if is_tensor(v) else v
            del input[k]

        Metric.update_metrics(torch.argmax(output, dim=1), label)
        if criterion is not None:
            if criterion.__class__.__name__ == "NewCrossEntropyLoss":
                batch_loss = criterion(
                    output, label, epoch=epoch if epoch is not None else 0
                )  # TODO: Turn this on with Sampler
            else:
                batch_loss = criterion(output, label)
        del output
        del label
        return batch_loss


    def grad_accum(self , 
        epoch,
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        clip,
        patience,
        Metric,
        prev_val_loss,
        prev_f1,
        total_loss_train,
        iters,
        log_val,
        path,
    ):
        gen = iter(train_dataloader)
        batch_size = train_dataloader.batch_size
        fn = self.get_statistics_big_batch if batch_size > self.big_batch else self.get_statistics
        steps = iters // log_val + 1 if iters % log_val != 0 else iters // log_val
        for i in tqdm(range(steps), desc="steps"):
            for j in tqdm(range(log_val), desc="iter"):
                try:
                    batch_idx = i * log_val + j
                    train_input, train_label = next(gen)
                    accum_iter, accum_sum = train_dataloader.dataset.retGradAccum(
                        i=batch_idx
                    )
                    train_batch_loss = (
                        fn(
                            train_input,
                            train_label,
                            model,
                            criterion,
                            Metric,
                            check="train",
                            epoch=epoch,
                        )
                        / accum_iter
                    )
                    del train_input
                    del train_label
                    total_loss_train += train_batch_loss.item()

                    # backward pass
                    train_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [
                            param
                            for param in model.parameters()
                            if param.requires_grad == True
                        ],
                        clip,
                    )
                    # do gradient accumulation here for dialogues
                    if ((batch_idx + 1) % accum_sum == 0) or (batch_idx + 1 == iters):
                        optimizer.step()
                        scheduler.step(epoch + batch_idx / iters)
                        model.zero_grad()
                except StopIteration:
                    break

            prev_val_loss, prev_f1 = self.run_validation(
                epoch,
                val_dataloader,
                model,
                criterion,
                optimizer,
                scheduler,
                Metric,
                prev_val_loss,
                prev_f1,
                total_loss_train / iters,
                batch_idx,
                log_val,
                path,
            )
            if self.patient_iter == patience:
                break
            # Do logging every log_val steps

        return model, optimizer, criterion, prev_val_loss, prev_f1


    def not_grad_accum(self , 
        epoch,
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        clip,
        patience,
        Metric,
        prev_val_loss,
        prev_f1,
        total_loss_train,
        iters,
        log_val,
        path,
    ):
        gen = iter(train_dataloader)
        batch_size = train_dataloader.batch_size
        fn = self.get_statistics_big_batch if batch_size > self.big_batch else self.get_statistics
        steps = iters // log_val + 1 if iters % log_val != 0 else iters // log_val
        for i in tqdm(range(steps), desc="steps"):
            for j in tqdm(range(log_val), desc="iter"):
                try:
                    batch_idx = i * log_val + j
                    train_input, train_label = next(gen)
                    train_batch_loss = fn(
                        train_input,
                        train_label,
                        model,
                        criterion,
                        Metric,
                        check="train",
                        epoch=epoch,
                    )
                    del train_input
                    del train_label
                    total_loss_train += train_batch_loss.item()

                    # backward pass
                    train_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [
                            param
                            for param in model.parameters()
                            if param.requires_grad == True
                        ],
                        clip,
                    )
                    optimizer.step()
                    scheduler.step(epoch + batch_idx / iters)
                    model.zero_grad()
                except StopIteration:
                    break
            prev_val_loss, prev_f1 = self.run_validation(
                epoch,
                val_dataloader,
                model,
                criterion,
                optimizer,
                scheduler,
                Metric,
                prev_val_loss,
                prev_f1,
                total_loss_train / iters,
                batch_idx,
                log_val,
                path,
            )
            if self.patient_iter == patience:
                break
        return model, optimizer, criterion, prev_val_loss, prev_f1


    def run_validation(self , 
        epoch,
        val_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        Metric,
        prev_val_loss,
        prev_f1,
        curr_loss,
        step,
        log_val,
        path,
    ):
        self.log(Metric, curr_loss, "train")
        val_loss, weightedF1 = self.validate(
            val_dataloader, model, criterion, Metric, name="val"
        )
        if weightedF1 > prev_f1:
            print(
                f"we have seen weightedF1 increase the previous best and we are updating our best f1 score to {weightedF1}"
            )
            prev_f1 = weightedF1
            save_model(model, optimizer, criterion, scheduler, epoch, step, path, log_val)
        if val_loss < prev_val_loss:
            self.patient_iter = 0
            prev_val_loss = val_loss
        else:
            self.patient_iter += 1
            print(
                f"we have seen loss increase for {self.patient_iter} steps and validation loss is {val_loss}, and previous best validation loss is {prev_val_loss}"
            )
        return prev_val_loss, prev_f1


    def validate(self , val_dataloader, model, criterion, Metric, name="val"):
        total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label in tqdm(val_dataloader, desc=name):
                val_batch_loss = self.get_statistics(
                    val_input, val_label, model, criterion, Metric, name, epoch=None
                )
                if criterion is not None:
                    total_loss_val += val_batch_loss.item()
                del val_input
                del val_label
            weightedF1 = self.log(
                Metric,
                total_loss_val / len(val_dataloader) if criterion is not None else 0,
                name,
            )
        return total_loss_val / len(val_dataloader), weightedF1


    def one_epoch(self , 
        epoch,
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        clip,
        epoch_switch,
        patience,
        Metric,
        prev_val_loss,
        prev_f1,
    ):
        total_loss_train = 0
        iters1 = len(train_dataloader[0])
        iters2 = len(train_dataloader[1])
        log_val1 = iters1 // self.num_steps
        log_val2 = iters2 // self.num_steps
        wandb.log({"log_val_multinomial": log_val1, "log_val_iterative": log_val2})
        
        sampler = train_dataloader[-1]
        path = "/".join(os.getcwd().split("/")[:-3]) + "/TAV_Train"
        if sampler == 'Both':
            # Do both 
            if epoch % epoch_switch == 0:
                model, optimizer, criterion, prev_val_loss, prev_f1 = self.not_grad_accum(
                    epoch,
                    train_dataloader[0],
                    val_dataloader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    clip,
                    patience,
                    Metric,
                    prev_val_loss,
                    prev_f1,
                    total_loss_train,
                    iters1,
                    log_val1,
                    path,
                )
            else:
                model, optimizer, criterion, prev_val_loss, prev_f1 = self.grad_accum(
                    epoch,
                    train_dataloader[1],
                    val_dataloader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    clip,
                    patience,
                    Metric,
                    prev_val_loss,
                    prev_f1,
                    total_loss_train,
                    iters2,
                    log_val2,
                    path,
                )
        else:
            # Do either weightedSampling or iterativeSampling
            model, optimizer, criterion, prev_val_loss, prev_f1 = self.not_grad_accum(
                    epoch,
                    train_dataloader[0] if sampler == 'Weighted' else train_dataloader[1],
                    val_dataloader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    clip,
                    patience,
                    Metric,
                    prev_val_loss,
                    prev_f1,
                    total_loss_train,
                    iters1,
                    log_val1,
                    path,
                )
            

        return model, optimizer, criterion, scheduler, prev_val_loss, prev_f1


    def train_network(self , 
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        learning_rate,
        epochs,
        weight_decay,
        T_max,
        Metric,
        patience,
        clip,
        epoch_switch,
        checkpoint=None,
    ):
        optimizer = AdamW(
            [param for param in model.parameters() if param.requires_grad == True],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_max
        )  # To prevent fitting to local minima != global minima
        prev_val_loss = 100
        prev_f1 = 0
        path = "/".join(os.getcwd().split("/")[:-3]) + "/TAV_Train"
        if checkpoint is not None:
            # epoch_num = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        for epoch_num in tqdm(range(epochs), desc="epochs"):
            wandb.log({"epoch": epoch_num, "learning_rate": scheduler.get_last_lr()[0]})
            optimizer.zero_grad()  # Zero out gradients before each epoch.

            model, optimizer, criterion, scheduler, prev_val_loss, prev_f1 = self.one_epoch(
                epoch_num,
                train_dataloader,
                val_dataloader,
                model,
                criterion,
                optimizer,
                scheduler,
                clip,
                epoch_switch,
                patience,
                Metric,
                prev_val_loss,
                prev_f1,
            )
            if self.patient_iter == patience:
                break
        model, _, _ = load_model(model, optimizer, criterion, path)
        return model


    def evaluate(self , model, test_dataloader, Metric):
        self.validate(test_dataloader, model, None, Metric, name="test")


    def log(self , Metric, loss, check="train"):
        (
            multiAcc,
            multiF1,
            multiRec,
            multiPrec,
            Acc,
            F1Macro,
            F1Weighted,
            Rec,
            Prec,
            _,
        ) = Metric.compute_scores(f"{check}")
        d1 = {
            f"{check}/loss": loss,
            f"{check}/acc": Acc,
            f"{check}/precision": Prec,
            f"{check}/recall": Rec,
            f"{check}/weighted-f1-score": F1Weighted,
            f"{check}/macro-f1-score": F1Macro,
        }
        print(f"\n in {check} \n Confusion Matrix = \n{_} \n", flush=True)
        wandb.log({**d1, **multiF1, **multiRec, **multiPrec, **multiAcc})
        Metric.reset_metrics()
        return F1Weighted