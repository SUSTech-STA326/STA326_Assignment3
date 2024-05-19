
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MovieLensDataset(Dataset):
    def __init__(self, ratings, all_movie_ids, all_user_ids):
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(all_user_ids)}
        self.movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(all_movie_ids)}
        
        self.users = torch.tensor([self.user_to_idx[user_id] for user_id in ratings['userId'].values], dtype=torch.long)
        self.movies = torch.tensor([self.movie_to_idx[movie_id] for movie_id in ratings['movieId'].values], dtype=torch.long)
        self.ratings = torch.tensor((ratings['rating'].values - 0.5) / 4.5, dtype=torch.float32)  # 缩放到 [0, 1]

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

def load_data():
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv('ml-latest-small/movies.csv')
    users = pd.read_csv('ml-latest-small/ratings.csv')['userId'].unique()
    
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    all_movie_ids = movies['movieId'].unique()
    all_user_ids = users
    
    train_dataset = MovieLensDataset(train_ratings, all_movie_ids, all_user_ids)
    test_dataset = MovieLensDataset(test_ratings, all_movie_ids, all_user_ids)
    
    return train_dataset, test_dataset, len(all_user_ids), len(all_movie_ids), test_ratings

train_dataset, test_dataset, num_users, num_items, test_ratings = load_data()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)
        self.output = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_emb(user_indices)
        item_embedding = self.item_emb(item_indices)
        interaction = user_embedding * item_embedding
        logits = self.output(interaction)
        return self.sigmoid(logits).view(-1, 1)  # 保持输出为 (batch_size, 1)

class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers):
        super(MLP, self).__init__()
        self.user_emb = nn.Embedding(num_users, layers[0] // 2)
        self.item_emb = nn.Embedding(num_items, layers[0] // 2)
        self.fc_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(layers[i], layers[i+1]), nn.ReLU())
              for i in range(len(layers)-1)]
        )
        self.output = nn.Linear(layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_emb(user_indices)
        item_embedding = self.item_emb(item_indices)
        x = torch.cat([user_embedding, item_embedding], dim=-1)
        x = self.fc_layers(x)
        logits = self.output(x)
        return self.sigmoid(logits).view(-1, 1)  # 保持输出为 (batch_size, 1)

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, gmf_dim, mlp_layers):
        super(NeuMF, self).__init__()
        self.gmf = GMF(num_users, num_items, gmf_dim)
        self.mlp = MLP(num_users, num_items, mlp_layers)
        self.output = nn.Linear(2, 1)  # GMF 和 MLP 合并后为 2 个特征
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        gmf_output = self.gmf(user_indices, item_indices)
        mlp_output = self.mlp(user_indices, item_indices)
        x = torch.cat([gmf_output, mlp_output], dim=1)  # 确保正确合并两个模型的输出
        logits = self.output(x)
        return self.sigmoid(logits)


def train_model(model, train_loader, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    
    model.train()
    for epoch in range(epochs):
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            optimizer.zero_grad()
            outputs = model(users, items).squeeze()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}: Loss = {loss.item()}')

# 初始化GMF模型
gmf_model = GMF(num_users, num_items, latent_dim=10)
# 训练GMF模型
train_model(gmf_model, train_loader, epochs=5, lr=0.1)

# 初始化MLP模型，假设有两层，每层10个神经元
mlp_model = MLP(num_users, num_items, [20, 10])  # 层数和每层的大小可以根据需要进行调整
# 训练MLP模型
train_model(mlp_model, train_loader, epochs=5, lr=0.01)

# 初始化并训练NeuMF模型
neumf_model = NeuMF(num_users, num_items, gmf_dim=10, mlp_layers=[20, 10])
train_model(neumf_model, train_loader, epochs=5, lr=1)

# 计算DCG的函数
def dcg_at_k(scores, k):
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0

# 计算理想DCG的函数
def idcg_at_k(k):
    ideal_scores = np.ones(k)
    return dcg_at_k(ideal_scores, k)

# 计算NDCG的函数
def ndcg_at_k(user_top_k, user_likes, k):
    ndcg_scores = []
    for user, items in user_top_k.items():
        test_items = user_likes.get(user, [])
        if not test_items:
            continue
        actual = np.zeros(len(items))
        for i, (item, score) in enumerate(items):
            if item in test_items:
                actual[i] = 1  # 如果项目在用户喜欢的项目列表中，标记为1
        dcg = dcg_at_k(actual, k)
        idcg = idcg_at_k(min(len(test_items), k))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        # 输出中间结果
        #print(f"User {user} - DCG: {dcg}, IDCG: {idcg}, NDCG: {ndcg}, Actual: {actual[:k]}")
    return np.mean(ndcg_scores)

# 获取用户前k个预测项目
def get_user_top_k(predictions, k):
    user_pred = {}
    for user, item, score in predictions:
        if user not in user_pred:
            user_pred[user] = []
        user_pred[user].append((item, score))
    user_top_k = {user: sorted(items, key=lambda x: x[1], reverse=True)[:k] for user, items in user_pred.items()}
    return user_top_k

# 获取用户喜欢的电影
def get_user_likes(test_ratings, threshold=4.0):
    user_likes = {}
    for user, group in test_ratings.groupby('userId'):
        liked_movies = group[group['rating'] >= threshold]['movieId'].tolist()
        user_likes[user] = liked_movies
    return user_likes

# 评估模型的函数
def evaluate_model(model, test_loader, test_ratings, k=10):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_preds = []
    for users, items, _ in test_loader:
        users, items = users.to(device), items.to(device)
        with torch.no_grad():
            outputs = model(users, items).squeeze()
        predictions = list(zip(users.cpu().numpy(), items.cpu().numpy(), outputs.cpu().numpy()))
        all_preds.extend(predictions)
    
    user_top_k = get_user_top_k(all_preds, k)
    
    # 获取用户喜欢的电影
    user_likes = get_user_likes(test_ratings)
    
    # 输出部分中间结果检查
    #print(f"Number of unique users in test set: {len(test_ratings['userId'].unique())}")
    #print(f"Sample user predictions (first 5): {list(user_top_k.items())[:5]}")
    
    # 新增详细输出：实际和预测的评分项
    for user, items in list(user_top_k.items())[:5]:
        test_items = user_likes.get(user, [])
        #print(f"User {user} - Test items: {test_items}")
        #print(f"User {user} - Predicted top {k} items: {[item for item, score in items]}")
    
    hr = hit_rate_at_k(user_top_k, user_likes, k)
    ndcg = ndcg_at_k(user_top_k, user_likes, k)
    
    return hr, ndcg

# 计算命中率的函数
def hit_rate_at_k(user_top_k, user_likes, k):
    hits = 0
    total = 0
    for user, items in user_top_k.items():
        test_items = user_likes.get(user, [])
        top_k_items = [item for item, score in items]
        hits += len(set(top_k_items) & set(test_items))
        total += len(test_items)
    return hits / total if total > 0 else 0

# 使用 GMF 模型进行评估
hr, ndcg = evaluate_model(gmf_model, test_loader, test_ratings, k=10)
print(f'GMF Model - HR@10: {hr}, NDCG@10: {ndcg}')

# 使用 MLP 模型进行评估
hr, ndcg = evaluate_model(mlp_model, test_loader, test_ratings, k=10)
print(f'MLP Model - HR@10: {hr}, NDCG@10: {ndcg}')

# 使用 NeuMF 模型进行评估
hr, ndcg = evaluate_model(neumf_model, test_loader, test_ratings, k=10)
print(f'NeuMF Model - HR@10: {hr}, NDCG@10: {ndcg}')

