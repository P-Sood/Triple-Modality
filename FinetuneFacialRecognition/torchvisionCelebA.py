import os
import sys
sys.path.insert(0,'/'.join(os.getcwd().split('/')[:-2])) 
__package__ = 'FinetuneFacialRecognition'
import torchvision.datasets as datasets
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import wandb
import numpy as np
import random
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self , num_classes):
        super(MyModel, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.facenet(x)
        x = self.fc(x)
        return x

def train(model, epoch , dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    iters = len(dataloader)
    correct = 0
    total = 0
    for batch_idx , (inputs, labels) in enumerate(tqdm(dataloader)):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + batch_idx / iters)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader) , correct/total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(dataloader) , correct/total

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main(param_dict):
    data_dir = param_dict['data_dir']
    epochs = param_dict['epoch']
    lr = param_dict['lr']
    batch_size = param_dict['batch_size']
    weight_decay = param_dict['weight_decay']
    T_max = param_dict['T_max']
    
    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define the data transformation
    data_transform = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(160),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        fixed_image_standardization
    ])


    # Load the CelebA dataset

    train_dataset = datasets.CelebA(data_dir, split='train', transform=train_transform , target_type = "identity")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = datasets.CelebA(data_dir, split='valid', transform=data_transform , target_type = "identity")
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    test_dataset = datasets.CelebA(data_dir, split='test', transform=data_transform , target_type = "identity")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Load the pre-trained model and move it to the GPU
    model = MyModel(10177+1).to(device).eval()

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr , weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = T_max)
    # Fine-tune the model on the new data
    for epoch in tqdm(range(epochs)):
        print("Started training" , flush = True)
        train_loss , train_acc  = train(model, epoch , train_dataloader, criterion, optimizer , scheduler, device)
        wandb.log(
            {
                'train/loss': train_loss , 
                'train/acc': train_acc , 
                })
        
        print("Started validation" , flush = True)
        valid_loss , valid_acc  = validate(model, valid_dataloader, criterion, device)
        wandb.log(
            {
                'val/loss': valid_loss , 
                'val/acc': valid_acc , 
                })
    # Test the fine-tuned model on the test data
    accuracy = test(model, test_dataloader, device)

    # Log metrics to wandb
    wandb.log({'test/accuracy': accuracy})

    
def parser():
    parser = argparse.ArgumentParser(description='Fine-tune InceptionResnetV1 on CelebA dataset')
    parser.add_argument('--data_dir', type=str, help='path to CelebA dataset' , default='/home/jupyter/multi-modal-emotion/FinetuneData/celeba')
    parser.add_argument('--epoch', type=int, help='number of epochs to train for' , default=10)
    parser.add_argument('--learning_rate', type=float, help='learning rate for optimizer' , default=0.001)
    parser.add_argument('--batch_size', type=int, help='batch size for training and validation' , default=32)
    parser.add_argument('--weight_decay', type=float, help='weight decay for optimizer' , default=0.0001)
    parser.add_argument('--T_max', type=int, help='T_max for cosine annealing scheduler' , default=3)
    parser.add_argument('--seed', type=int, help='seed for random number generator' , default=42)
    
    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

if __name__ == '__main__':
    config = parser()
    wandb.init(entity="ddi" , config = config)
    config = wandb.config
    
    # seed_everything(config.seed)
    param_dict = {
        'data_dir':"/home/jupyter/multi-modal-emotion/FinetuneData/celeba/",#config.data_dir,
        'epoch':config.epoch,
        'lr': config.learning_rate,
        'batch_size': 512,#config.batch_size ,
        'weight_decay':config.weight_decay ,
        'T_max':config.T_max ,
    }
    print(param_dict)
    main(param_dict)
    # wandb agent ddi/FinetuneCelebA/6linc105
    