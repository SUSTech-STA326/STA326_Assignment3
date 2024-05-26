import pandas as pd
import numpy as np
import math
from collections import defaultdict
import heapq
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import os
from dataset import NCFData
from load import load_dataset,hit,ndcg,metrics
from MLP import MLP
from tqdm import tqdm

    
dropout=0.0
lr = 0.001
epoch_num = 20
top_k = 10

    
train_data, test_data, user_num, item_num, train_mat = load_dataset()

train_dataset = NCFData(train_data, item_num, train_mat, num_ng=4, is_training=True)  # neg_items=4,default
test_dataset = NCFData(test_data, item_num, train_mat, num_ng=0, is_training=False)  # 100
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
# every user have 99 negative items and one positive items，so batch_size=100
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=99 + 1, shuffle=False, num_workers=2)


if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU
    cudnn.benchmark = True

    for factor_num in [8, 16, 32, 64]:
        for num_layers in range(5):
            model = MLP(int(user_num), int(item_num), factor_num, num_layers, dropout)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_function = nn.BCELoss()

            best_hr, best_ndcg = 0.0, 0.0

            for epoch in tqdm(range(epoch_num), desc="Training Progress"):
                model.train()
                train_loader.dataset.ng_sample()
                for user, item, label in train_loader:
                    user = user.to(device)
                    item = item.to(device)
                    label = label.float().to(device)

                    model.zero_grad()
                    prediction = model(user, item)
                    loss = loss_function(prediction, label)
                    loss.backward()
                    optimizer.step()

                model.eval()
                HR, NDCG = metrics(model, test_loader, top_k, device)
                if HR > best_hr or NDCG > best_ndcg:
                    best_hr, best_ndcg = HR, NDCG
                    # 仅在性能提升时保存模型
                    torch.save(model.state_dict(), f'diff_layer_dict/MLP_layer_{num_layers}_factor_{factor_num}.pth')

            print(f"num of factor: {factor_num}\nnum of layer: {num_layers}\nbest hr: {best_hr}\nbest_ndcg: {best_ndcg}")

