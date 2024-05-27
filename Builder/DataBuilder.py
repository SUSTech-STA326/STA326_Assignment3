import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class DataProcessor:  # 数据处理类
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def preprocess_data(self):
        # 先读取数据，然后进行一系列的处理
        train_data = pd.read_csv(self.train_path, sep='\t',
                                 names=['user', 'item', 'rating', 'timestamp'])

        train_data.drop(columns='timestamp', inplace=True)
        train_data['rating'] = 1

        min_item = min(train_data['item'].tolist())
        max_item = max(train_data['item'].tolist())
        all_items = set(range(min_item, max_item + 1))

        # 用于存放negative样本的字典
        negative_samples = {u_id: [] for u_id in train_data['user'].unique()}

        # 给每个用户获取negative样本
        for u_id in negative_samples:
            # positive样本
            pos_items = set(train_data[train_data['user'] == u_id]['item'].unique())

            neg_items = list(all_items - pos_items)

            selected_negatives = np.random.choice(neg_items, 8, replace=False)

            negative_samples[u_id].extend(selected_negatives)

        # 创建一个DataFrame来存放negative样本
        neg_data_list = []
        for u_id, items in negative_samples.items():
            for item in items:
                neg_data_list.append([u_id, item, 0])

        df_negatives = pd.DataFrame(neg_data_list, columns=['user', 'item', 'rating'])

        # 将negative样本拼接到原始数据中
        final_data = pd.concat([train_data, df_negatives], ignore_index=True)
        final_data.to_csv(f'{self.train_path}.final', sep='\t', header=False, index=False)

        test_data = pd.read_csv(self.test_path, sep='\t',
                                names=['user', 'item', 'rating', 'timestamp'])
        test_data.drop(columns='timestamp', inplace=True)
        test_data['rating'] = 1
        test_data.to_csv(f'{self.test_path}.final', sep='\t', header=False, index=False)


class RatingDataset(Dataset):
    """Dataset for loading user-item ratings."""

    def __init__(self, file):
        self.data = pd.read_csv(file, sep='\t', names=['user', 'item', 'rating'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Convert data to tensors
        user = torch.tensor(row['user'], dtype=torch.int64)
        item = torch.tensor(row['item'], dtype=torch.int64)
        rating = torch.tensor(row['rating'], dtype=torch.float32)
        return user, item, rating


class NegativeDataset(Dataset):
    """Dataset for loading negative samples for each user-item pair."""

    def __init__(self, file):
        self.negatives = self.load_negative(file)

    @staticmethod
    def load_negative(n_file):
        negs = {}
        with open(n_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                user, item = eval(line[0])
                negs[(user, item)] = list(map(int, line[1:]))
        return negs

    def __len__(self):
        return len(self.negatives)

    def __getitem__(self, idx):
        user_item = list(self.negatives.keys())[idx]
        user = torch.tensor(user_item[0], dtype=torch.int64)
        item = torch.tensor(user_item[1], dtype=torch.int64)
        negs = torch.tensor(self.negatives[user_item], dtype=torch.int64)
        return user, item, negs
