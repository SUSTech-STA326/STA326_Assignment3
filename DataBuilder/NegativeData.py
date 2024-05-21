import torch
from torch.utils.data import Dataset


class NegativeData(Dataset):
    """Dataset for loading negative samples for each user-item pair."""
    def __init__(self, negative_file):
        self.negatives = self._load_negative(negative_file)

    def _load_negative(self, negative_file):
        negatives = {}
        with open(negative_file, 'r') as file:
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