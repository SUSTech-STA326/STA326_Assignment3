import torch.nn as nn
from model import *
import numpy as np
import os
import random
from tqdm import tqdm
from dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader


def model_train(config, num_of_negatives, batch_size, num_of_epochs, seed):
    ####* set seed and device
    seed_everything(seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    ####* init dataset and dataloader
    rating_mat, num_of_user, num_of_item = load_rating_file_as_sparse("Data/ml-1m.train.rating")
    negative_sample_list = load_negative_file("Data/ml-1m.test.negative")

    train_dataset = RatingDataset(rating_mat, negative_sample_list, num_of_user, num_of_item, num_of_negatives)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ####* set model, loss, and optimizer
    model = RecommenderModel(config)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ####* train model
    model.train()
    for epoch in tqdm(range(num_of_epochs)):
        for user, item, label in train_dataloader:
            optimizer.zero_grad()
            user,item,label = user.to(device), item.to(device), label.float().to(device)
            output = model(user, item)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True