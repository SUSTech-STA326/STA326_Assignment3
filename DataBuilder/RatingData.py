import pandas as pd
import torch
from torch.utils.data import Dataset


class RatingsData(Dataset):
    """Dataset for loading user-item ratings."""

    def __init__(self, ratings_file):
        self.user_item_ratings = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating'])

    def __len__(self):
        return len(self.user_item_ratings)

    def __getitem__(self, idx):
        row = self.user_item_ratings.iloc[idx]
        # Convert data to tensors
        user_id = torch.tensor(row['user_id'], dtype=torch.int64)
        item_id = torch.tensor(row['item_id'], dtype=torch.int64)
        rating = torch.tensor(row['rating'], dtype=torch.float32)
        return (user_id, item_id, rating)