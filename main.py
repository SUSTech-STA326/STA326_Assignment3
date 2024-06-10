import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ndcg_score

import chardet

# 检测文件编码
with open('movies.dat', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# 使用检测到的编码读取文件
movies_df = pd.read_csv('movies.dat', sep='::', header=None, names=['MovieID', 'Title', 'Genres'], engine='python', encoding=encoding)
# Load the data
ratings_df = pd.read_csv('ratings.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
users_df = pd.read_csv('users.dat', sep='::', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python')

# Prepare the data
num_users = users_df['UserID'].max() + 1
num_movies = movies_df['MovieID'].max() + 1
ratings_df['UserID'] -= 1
ratings_df['MovieID'] -= 1

# Split the data into train and test sets
train_data = ratings_df.sample(frac=0.8, random_state=42)
test_data = ratings_df.drop(train_data.index)

# Convert data to tensors
train_user_tensor = torch.LongTensor(train_data['UserID'].values)
train_movie_tensor = torch.LongTensor(train_data['MovieID'].values)
train_rating_tensor = torch.FloatTensor(train_data['Rating'].values)

test_user_tensor = torch.LongTensor(test_data['UserID'].values)
test_movie_tensor = torch.LongTensor(test_data['MovieID'].values)
test_rating_tensor = torch.FloatTensor(test_data['Rating'].values)

# Define the models
class GMF(nn.Module):
    def __init__(self, num_users, num_movies, latent_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.movie_embedding = nn.Embedding(num_movies, latent_dim)
        self.affine_output = nn.Linear(latent_dim, 1)

    def forward(self, user_indices, movie_indices):
        user_embedding = self.user_embedding(user_indices)
        movie_embedding = self.movie_embedding(movie_indices)
        element_product = torch.mul(user_embedding, movie_embedding)
        rating = self.affine_output(element_product)
        return rating

class MLP(nn.Module):
    def __init__(self, num_users, num_movies, latent_dim, layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.movie_embedding = nn.Embedding(num_movies, latent_dim)
        self.fc_layers = nn.ModuleList()
        for idx in range(len(layers)):
            if idx == 0:
                self.fc_layers.append(nn.Linear(latent_dim * 2, layers[idx]))
            else:
                self.fc_layers.append(nn.Linear(layers[idx - 1], layers[idx]))
        self.affine_output = nn.Linear(layers[-1], 1)
        self.relu = nn.ReLU()

    def forward(self, user_indices, movie_indices):
        user_embedding = self.user_embedding(user_indices)
        movie_embedding = self.movie_embedding(movie_indices)
        vector = torch.cat([user_embedding, movie_embedding], dim=-1)
        for layer in self.fc_layers:
            vector = self.relu(layer(vector))
        rating = self.affine_output(vector)
        return rating

class NeuMF(nn.Module):
    def __init__(self, num_users, num_movies, latent_dim_mf, latent_dim_mlp, layers):
        super(NeuMF, self).__init__()
        self.GMF = GMF(num_users, num_movies, latent_dim_mf)
        self.MLP = MLP(num_users, num_movies, latent_dim_mlp, layers)
        self.affine_output = nn.Linear(latent_dim_mf + layers[-1], 1)

    def forward(self, user_indices, movie_indices):
        gmf_output = self.GMF(user_indices, movie_indices)
        mlp_output = self.MLP(user_indices, movie_indices)
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        rating = self.affine_output(concat)
        return rating
# Training function
def train(model, optimizer, criterion, train_user_tensor, train_movie_tensor, train_rating_tensor, num_epochs, batch_size):
    for epoch in range(num_epochs):
        for i in range(0, len(train_user_tensor), batch_size):
            batch_user = train_user_tensor[i:i+batch_size]
            batch_movie = train_movie_tensor[i:i+batch_size]
            batch_rating = train_rating_tensor[i:i+batch_size]

            optimizer.zero_grad()
            predicted_ratings = model(batch_user, batch_movie)
            loss = criterion(predicted_ratings, batch_rating.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation function
def evaluate(model, test_user_tensor, test_movie_tensor, test_rating_tensor, top_k):
    model.eval()
    with torch.no_grad():
        predicted_ratings = model(test_user_tensor, test_movie_tensor)
        _, indices = torch.topk(predicted_ratings.squeeze(), top_k)
        recommended_movies = test_movie_tensor[indices].tolist()
        actual_movies = test_movie_tensor[test_rating_tensor >= 4].tolist()

        hr = len(set(recommended_movies) & set(actual_movies)) / len(set(actual_movies))

        # Convert actual_movies and recommended_movies to binary relevance format
        actual_relevance = np.isin(test_movie_tensor.numpy(), actual_movies).astype(int)
        recommended_relevance = np.isin(test_movie_tensor.numpy(), recommended_movies).astype(int)

        ndcg = ndcg_score(actual_relevance.reshape(1, -1), recommended_relevance.reshape(1, -1))

    return hr, ndcg
# Main function
def main():
    latent_dim_mf = 8
    latent_dim_mlp = 8
    layers = [16, 32, 16, 8]
    num_epochs = 10
    batch_size = 256
    learning_rate = 0.001
    top_k = 10

    gmf_model = GMF(num_users, num_movies, latent_dim_mf)
    mlp_model = MLP(num_users, num_movies, latent_dim_mlp, layers)
    neumf_model = NeuMF(num_users, num_movies, latent_dim_mf, latent_dim_mlp, layers)

    criterion = nn.MSELoss()
    gmf_optimizer = optim.Adam(gmf_model.parameters(), lr=learning_rate)
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
    neumf_optimizer = optim.Adam(neumf_model.parameters(), lr=learning_rate)

    print("Training GMF model...")
    train(gmf_model, gmf_optimizer, criterion, train_user_tensor, train_movie_tensor, train_rating_tensor, num_epochs, batch_size)
    gmf_hr, gmf_ndcg = evaluate(gmf_model, test_user_tensor, test_movie_tensor, test_rating_tensor, top_k)
    print(f"GMF: HR@{top_k} = {gmf_hr:.4f}, NDCG@{top_k} = {gmf_ndcg:.4f}")

    print("Training MLP model...")
    train(mlp_model, mlp_optimizer, criterion, train_user_tensor, train_movie_tensor, train_rating_tensor, num_epochs, batch_size)
    mlp_hr, mlp_ndcg = evaluate(mlp_model, test_user_tensor, test_movie_tensor, test_rating_tensor, top_k)
    print(f"MLP: HR@{top_k} = {mlp_hr:.4f}, NDCG@{top_k} = {mlp_ndcg:.4f}")

    print("Training NeuMF model...")
    train(neumf_model, neumf_optimizer, criterion, train_user_tensor, train_movie_tensor, train_rating_tensor, num_epochs, batch_size)
    neumf_hr, neumf_ndcg = evaluate(neumf_model, test_user_tensor, test_movie_tensor, test_rating_tensor, top_k)
    print(f"NeuMF: HR@{top_k} = {neumf_hr:.4f}, NDCG@{top_k} = {neumf_ndcg:.4f}")

    print("Ablation study on MLP model...")
    layers_list = [[8], [16, 8], [32, 16, 8], [64, 32, 16, 8]]
    for layers in layers_list:
        mlp_model = MLP(num_users, num_movies, latent_dim_mlp, layers)
        mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
        train(mlp_model, mlp_optimizer, criterion, train_user_tensor, train_movie_tensor, train_rating_tensor, num_epochs, batch_size)
        mlp_hr, mlp_ndcg = evaluate(mlp_model, test_user_tensor, test_movie_tensor, test_rating_tensor, top_k)
        print(f"MLP (layers={layers}): HR@{top_k} = {mlp_hr:.4f}, NDCG@{top_k} = {mlp_ndcg:.4f}")

if __name__ == "__main__":
    main()