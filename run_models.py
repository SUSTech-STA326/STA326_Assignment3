import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd

from model import GMF, MLP, NeuMF
from evaluation import evaluate
import re 
from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import Dataset

learning_rate = 0.001
epochs = 100
data_dir = 'ml-1m'
batch_size = 64


class NegativeDataset(Dataset):
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
        return user_id, item_id, negative_ids


train_data = pd.read_csv(f'{data_dir}/ml-1m.train.rating', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
train_data = train_data.drop(columns='timestamp')
test_data = pd.read_csv(f'{data_dir}/ml-1m.test.rating', sep='\t', header=None, names=['user', 'item', 'rating', 'timestamp'])
test_data = test_data.drop(columns='timestamp')
negative_dataset = NegativeDataset('./ml-1m/ml-1m.test.negative')
negative_loader = DataLoader(negative_dataset, batch_size=batch_size, shuffle=False)
train_dataset = TensorDataset(torch.tensor(train_data['user'].values),
                          torch.tensor(train_data['item'].values),
                          torch.tensor(train_data['rating'].values))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(torch.tensor(test_data['user'].values),
                          torch.tensor(test_data['item'].values),
                          torch.tensor(test_data['rating'].values))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_users, test_items = torch.tensor(test_data['user'].values), torch.tensor(test_data['item'].values)

num_users = max(train_data['user'].max(), test_data['user'].max()) + 1
num_items = max(train_data['item'].max(), test_data['item'].max()) + 1

embedding_size = 64
gmf_model = GMF(num_users, num_items, embedding_size)
hidden_sizes = [16]  
mlp_model_1 = MLP(num_users, num_items, hidden_sizes)
hidden_sizes = [16, 8] 
mlp_model_2 = MLP(num_users, num_items, hidden_sizes)
hidden_sizes = [16, 64, 8] 
mlp_model_3 = MLP(num_users, num_items, hidden_sizes)
hidden_sizes = [16, 64, 16, 8] 
mlp_model_4 = MLP(num_users, num_items, hidden_sizes)
neumf_model = NeuMF(num_users, num_items, mf_dim=10, mlp_layers=[64, 32], dropout=0.2)


results = []
model = gmf_model

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()
for epoch in range(30):
    for users, items, ratings in train_loader:
        criterion = nn.BCEWithLogitsLoss()
        predictions = model(users, items).squeeze()
        predictions = predictions.view(-1)
        loss = criterion(predictions, ratings.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{30}')
    hr, ndcg = evaluate(model, test_loader, negative_loader, top_k=10)
    results.append({
        'Model': 'GMF',
        'HR@10': hr,
        'NDCG@10': ndcg
    })
    print(f"Model: {model.__class__.__name__}, HR@10: {hr}, NDCG@10: {ndcg}")
    
results_df = pd.DataFrame(results)
results_df.to_csv('GMF.csv', index=False)

results = []
count = 0
for model in [mlp_model_1, mlp_model_2, mlp_model_3, mlp_model_4]:
    count = count + 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(30):
        for users, items, ratings in train_loader:
            criterion = nn.BCEWithLogitsLoss()
            predictions = model(users, items).squeeze()
            predictions = predictions.view(-1)
            loss = criterion(predictions, ratings.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{30}')
        hr, ndcg = evaluate(model, test_loader, negative_loader, top_k=10)
        results.append({
            'Model': 'MLP_' + str(count),
            'HR@10': hr,
            'NDCG@10': ndcg
        })
        print(f"Model: {model.__class__.__name__}, HR@10: {hr}, NDCG@10: {ndcg}")
        
        
results_df = pd.DataFrame(results)
results_df.to_csv('MLP.csv', index=False)

results = []
model = neumf_model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()
for epoch in range(30):
    for users, items, ratings in train_loader:
        criterion = nn.BCEWithLogitsLoss()
        predictions = model(users, items).squeeze()
        predictions = predictions.view(-1)
        loss = criterion(predictions, ratings.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{30}')
    hr, ndcg = evaluate(model, test_loader, negative_loader, top_k=10)
    results.append({
        'Model': model.__class__.__name__,
#             'Loss': loss.item(),
        'HR@10': hr,
        'NDCG@10': ndcg
    })
    print(f"Model: {model.__class__.__name__}, HR@10: {hr}, NDCG@10: {ndcg}")

results_df = pd.DataFrame(results)
results_df.to_csv('NeuMF.csv', index=False)
