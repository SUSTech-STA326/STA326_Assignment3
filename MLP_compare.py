import pandas as pd
import chardet
with open('movies.dat', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# Load ratings data
ratings = pd.read_csv('ratings.dat', sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding=encoding)

# Load users data
users = pd.read_csv('users.dat', sep='::', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python', encoding=encoding)

# Load movies data
movies = pd.read_csv('movies.dat', sep='::', names=['MovieID', 'Title', 'Genres'], engine='python', encoding=encoding)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode user and movie IDs
user_encoder = LabelEncoder()
ratings['UserID'] = user_encoder.fit_transform(ratings['UserID'])

movie_encoder = LabelEncoder()
ratings['MovieID'] = movie_encoder.fit_transform(ratings['MovieID'])

# Split data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = data['UserID'].values
        self.movies = data['MovieID'].values
        self.ratings = data['Rating'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]


class MLP(nn.Module):
    def __init__(self, num_users, num_movies, layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, 32)
        self.movie_embedding = nn.Embedding(num_movies, 32)

        self.layers = nn.Sequential()
        input_size = 64
        for i, output_size in enumerate(layers):
            self.layers.add_module(f"fc{i}", nn.Linear(input_size, output_size))
            self.layers.add_module(f"relu{i}", nn.ReLU())
            input_size = output_size
        self.layers.add_module("output", nn.Linear(input_size, 1))

    def forward(self, user, movie):
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(movie)
        x = torch.cat([user_emb, movie_emb], dim=-1)
        return self.layers(x)


# Define training function
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            users, movies, ratings = batch
            optimizer.zero_grad()
            outputs = model(users, movies)
            loss = criterion(outputs.squeeze(), ratings.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


# Define evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    hr, ndcg = 0, 0
    # Implement HR@10 and NDCG@10 evaluation here
    return hr, ndcg


# Prepare data loaders
train_dataset = MovieLensDataset(train_data)
test_dataset = MovieLensDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
num_users = ratings['UserID'].nunique()
num_movies = ratings['MovieID'].nunique()

# Different layer configurations
layer_configs = [[64], [64, 32], [64, 32, 16]]

for layers in layer_configs:
    model = MLP(num_users, num_movies, layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training MLP with layers: {layers}")
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    hr, ndcg = evaluate_model(model, test_loader)
    print(f"HR@10: {hr}, NDCG@10: {ndcg}\n")

import numpy as np


def get_top_k_recommendations(score, k=10):
    top_k = np.argsort(score)[-k:][::-1]
    return top_k


def evaluate_model(model, test_loader, k=10):
    hr, ndcg = [], []

    with torch.no_grad():
        for users, movies, ratings in test_loader:
            scores = model(users, movies)
            for i in range(len(users)):
                user = users[i].item()
                movie = movies[i].item()
                rating = ratings[i].item()
                score = scores[i].item()

                # Generate top-k recommendations
                top_k_movies = get_top_k_recommendations(score, k=k)

                # Calculate HR@10
                if movie in top_k_movies:
                    hr.append(1)
                else:
                    hr.append(0)

                # Calculate NDCG@10
                if movie in top_k_movies:
                    rank = np.where(top_k_movies == movie)[0][0]
                    ndcg.append(1 / np.log2(rank + 2))
                else:
                    ndcg.append(0)

    hr = np.mean(hr)
    ndcg = np.mean(ndcg)
    return hr, ndcg

# Different layer configurations
layer_configs = [[64], [64, 32], [64, 32, 16]]

for layers in layer_configs:
    model = MLP(num_users, num_movies, layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training MLP with layers: {layers}")
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    hr, ndcg = evaluate_model(model, test_loader)
    print(f"HR@10: {hr}, NDCG@10: {ndcg}\n")