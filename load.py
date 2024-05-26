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


def load_dataset(test_num=100):
    
    train_data = pd.read_csv("./Dataset/ml-1m.train.rating",
                             sep='\t', header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
 
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
 
    train_data = train_data.values.tolist()
 
    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
 
    test_data = []
    with open("./Dataset/ml-1m.test.negative", 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])  # one postive item
            for i in arr[1:]:
                test_data.append([u, int(i)])  # 99 negative items
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat
 
def hit(gt_item,pred_items):
    if gt_item in pred_items:
        return 1
    return 0 
 
def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0
  
def metrics(model, test_loader, top_k, device):
    HR, NDCG = [], []
 
    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)
 
        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()
 
        gt_item = item[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))
    return np.mean(HR), np.mean(NDCG)


