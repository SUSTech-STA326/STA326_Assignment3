import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import heapq
import os
import datetime
import time

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchsummary import summary

import warnings

warnings.filterwarnings('ignore')

print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class GMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, reg=[0, 0]):
        super(GMF, self).__init__()
        self.MF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        self.MF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=embed_dim)
        self.linear = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.long()
        MF_Embedding_User = self.MF_Embedding_User(input[:, 0])
        MF_Embedding_Item = self.MF_Embedding_Item(input[:, 1])

        predict = torch.mul(MF_Embedding_User, MF_Embedding_Item)

        linear = self.linear(predict)
        output = self.sigmoid(linear)
        output = output.squeeze(-1)
        return output
