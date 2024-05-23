from models import GMF, MLP, NeuMF
from dataset import loadTrain, mlDataset, get_train_instances
from torch.utils.data import DataLoader
from training import train
import torch
import os
import numpy as np

## Set seed
def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load dataset
dataset = loadTrain("./Data/"+"ml-1m")
trains = dataset.trainMatrix
num_users, num_items = trains.shape

## Create dataloader
user_input, item_input, labels = get_train_instances(trains, num_items, num_negatives=4)
train_dataset = mlDataset(user_input, item_input, labels)
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=2)

## Model initializations
GMF_config = {'num_users': num_users, 'num_items': num_items, 'factor': 8, 'layer_X': 0}
MLP_config = {'num_users': num_users, 'num_items': num_items, 'factor': 8, 'layer_X': 3}
NeuMF_config = {'num_users': num_users, 'num_items': num_items, 'factor': 8, 'layer_X': 3}

GMF_model = GMF(GMF_config).to(device)
MLP_model = MLP(MLP_config).to(device)
NeuMF_model = NeuMF(NeuMF_config).to(device)

## Training
# train(GMF_model, "GMF", train_loader, num_epochs=100)
# train(MLP_model, "MLP", train_loader, num_epochs=100)
# train(NeuMF_model, "NeuMF", train_loader, num_epochs=100)

## MLP with different number of layers
MLP_config_0 = {'num_users': num_users, 'num_items': num_items, 'factor': 8, 'layer_X': 0}
MLP_config_1 = {'num_users': num_users, 'num_items': num_items, 'factor': 8, 'layer_X': 1}
MLP_config_2 = {'num_users': num_users, 'num_items': num_items, 'factor': 8, 'layer_X': 2}
MLP_config_3 = {'num_users': num_users, 'num_items': num_items, 'factor': 8, 'layer_X': 3}
MLP_config_4 = {'num_users': num_users, 'num_items': num_items, 'factor': 8, 'layer_X': 4}

MLP_model_0 = MLP(MLP_config_0).to(device)
MLP_model_1 = MLP(MLP_config_1).to(device)
MLP_model_2 = MLP(MLP_config_2).to(device)
MLP_model_3 = MLP(MLP_config_3).to(device)
MLP_model_4 = MLP(MLP_config_4).to(device)

train(MLP_model_0, "MLP_0", train_loader, num_epochs=100)
train(MLP_model_1, "MLP_1", train_loader, num_epochs=100)
train(MLP_model_2, "MLP_2", train_loader, num_epochs=100)
train(MLP_model_3, "MLP_3", train_loader, num_epochs=100)
train(MLP_model_4, "MLP_4", train_loader, num_epochs=100)