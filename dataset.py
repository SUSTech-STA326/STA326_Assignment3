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


class NCFData(Dataset):  # define the dataset
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        # Note that the labels are only useful when training, we thus add them in the ng_sample() function.
        self.features_ps = features #其中每个元素是一个 [user, item] 对，代表用户和物品的交互
        self.num_item = num_item 
        self.train_mat = train_mat #用户-物品对已有交互
        self.num_ng = num_ng #表示每个正样本要生成的负样本的数量
        self.is_training = is_training #指示是否处于训练模式
        self.labels = [0 for _ in range(len(features))]
 
    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])
 
        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]
 
        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng
 
    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)
 
    def __getitem__(self, idx):
        '''
        if self.is_training:
            self.ng_sample()
            features = self.features_fill
            labels = self.labels_fill
        else:
            features = self.features_ps
            labels = self.labels
        '''
        features = self.features_fill if self.is_training else self.features_ps
        labels = self.labels_fill if self.is_training else self.labels
 
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label