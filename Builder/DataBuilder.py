import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Preprocess:  # 数据预处理
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def generate_final(self):
        # 先读取原始数据，之后进行进一步处理
        ori_train_data = pd.read_csv(self.train_path, sep='\t',
                                     names=['user_id', 'item_id', 'rating', 'timestamp'])

        ori_train_data.drop(columns='timestamp', inplace=True)
        ori_train_data['rating'] = 1

        min_item_id = min(ori_train_data['item_id'].tolist())
        max_item_id = max(ori_train_data['item_id'].tolist())
        all_item_ids = set(range(min_item_id, max_item_id + 1))

        # 用于根据user id存放negative samples
        negative_samples = {user_id: [] for user_id in ori_train_data['user_id'].unique()}

        # 给每个user id获取negative samples
        for user_id in negative_samples:
            # positive samples
            p_items = set(ori_train_data[ori_train_data['user_id'] == user_id]['item_id'].unique())

            # 计算潜在的negative samples
            n_items = list(all_item_ids - p_items)

            # 随机选择样本，论文中是4个，这里设置8个
            selected_negatives = np.random.choice(n_items, 8, replace=False)

            negative_samples[user_id].extend(selected_negatives)

        # Create a DataFrame for negative samples
        neg_data_list = []
        for user_id, items in negative_samples.items():
            for item in items:
                neg_data_list.append([user_id, item, 0])

        df_negatives = pd.DataFrame(neg_data_list, columns=['user_id', 'item_id', 'rating'])

        # Concatenate the negative samples to the original data
        df_final = pd.concat([ori_train_data, df_negatives], ignore_index=True)
        df_final.to_csv(f'{self.train_path}.final', sep='\t', header=False, index=False)

        test_data = pd.read_csv(self.test_path, sep='\t',
                                names=['user_id', 'item_id', 'rating', 'timestamp'])
        test_data.drop(columns='timestamp', inplace=True)
        test_data['rating'] = 1
        test_data.to_csv(f'{self.test_path}.final', sep='\t', header=False, index=False)


class RatingData(Dataset):
    """Dataset for loading user-item ratings."""

    def __init__(self, file):
        self.ratings = pd.read_csv(file, sep='\t', names=['user_id', 'item_id', 'rating'])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        # Convert data to tensors
        user_id = torch.tensor(row['user_id'], dtype=torch.int64)
        item_id = torch.tensor(row['item_id'], dtype=torch.int64)
        rating = torch.tensor(row['rating'], dtype=torch.float32)
        return user_id, item_id, rating


class NegativeData(Dataset):
    """Dataset for loading negative samples for each user-item pair."""

    def __init__(self, file):
        self.negatives = self.load_negative(file)

    @staticmethod
    def load_negative(n_file):
        negatives = {}
        with open(n_file, 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                user, item = eval(line[0])
                negatives[(user, item)] = list(map(int, line[1:]))
        return negatives

    def __len__(self):
        return len(self.negatives)

    def __getitem__(self, idx):
        user_item = list(self.negatives.keys())[idx]
        user_id = torch.tensor(user_item[0], dtype=torch.int64)
        item_id = torch.tensor(user_item[1], dtype=torch.int64)
        negative_ids = torch.tensor(self.negatives[user_item], dtype=torch.int64)
        return user_id, item_id, negative_ids
