import pandas as pd
import torch
from torch.utils.data import Dataset


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
        return (user_id, item_id, negative_ids)