import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

class UserItemRatingDataset(Dataset):
    def __init__(self, user, item, target) -> None:
        super().__init__()
        self.user = user
        self.item = item
        self.target = target
    
    def __getitem__(self, index):
        return self.user[index], self.item[index], self.target[index]
    
    def __len__(self):
        return self.user.size(0)

class Generator:
    def __init__(self, ratings):
        self.ratings = ratings
        self.preprocess_ratings = self._binarize(ratings)
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        self.negatives = self._negative(self.ratings)
        self.train_ratings, self.test_ratings = self._split(self.preprocess_ratings)

    # For explicit feedback
    def _binarize(self, ratings):
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings
    
    def _negative(self, ratings):
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'}
        )
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]
    
    def _split(self, ratings):
        ratings['rank_latest'] = ratings.groupby('userId')['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]
    
    def get_train_loader(self, num_negatives, batch_size):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0)) # negative samples get 0 rating
        
        dataset = UserItemRatingDataset(user = torch.LongTensor(users),
                                        item = torch.LongTensor(items), 
                                        target = torch.FloatTensor(ratings))
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    @property
    def get_evaluate_data(self):
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users), torch.LongTensor(negative_items)]
    