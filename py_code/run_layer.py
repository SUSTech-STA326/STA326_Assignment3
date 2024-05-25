import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# noinspection PyUnresolvedReferences
from torch.utils.data import DataLoader, TensorDataset
from Dataset import Dataset
from evaluate import evaluate_model
from time import time
import argparse
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim, sparse=False)
        self.item_embedding = nn.Embedding(num_items, latent_dim, sparse=False)
        self.predict_layer = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices).to(device)
        item_embedding = self.item_embedding(item_indices).to(device)
        element_product = user_embedding * item_embedding
        prediction = self.predict_layer(element_product.sum(dim=1, keepdim=True))
        return prediction.view(-1)


class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers):
        super(MLP, self).__init__()
        self.embedding_user = nn.Embedding(num_users, layers[0] // 2)
        self.embedding_item = nn.Embedding(num_items, layers[0] // 2)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers[i - 1], layers[i]),
                nn.ReLU()
            ) for i in range(1, len(layers))
        ])
        self.predict = nn.Sequential(
            nn.Linear(layers[-1], 1),
            nn.Sigmoid()
        )
        

    def forward(self, user, item):
        user_emb = self.embedding_user(user)
        item_emb = self.embedding_item(item)
        vector = torch.cat([user_emb, item_emb], dim=-1)
        for layer in self.layers:
            vector = layer(vector)
        pred = self.predict(vector)
        return pred.view(-1)


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, layers, mf_dim):
        super(NeuMF, self).__init__()
        self.mf_embedding_user = nn.Embedding(num_users, mf_dim)
        self.mf_embedding_item = nn.Embedding(num_items, mf_dim)
        self.mlp_embedding_user = nn.Embedding(num_users, layers[0] // 2)
        self.mlp_embedding_item = nn.Embedding(num_items, layers[0] // 2)
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers[i - 1], layers[i]),
                nn.ReLU()
            ) for i in range(1, len(layers))
        ])
        self.predict = nn.Sequential(
            nn.Linear(mf_dim + layers[-1], 1),
            nn.Sigmoid()
        )

    def forward(self, user, item):
        mf_user_emb = self.mf_embedding_user(user)
        mf_item_emb = self.mf_embedding_item(item)
        mf_vector = mf_user_emb * mf_item_emb

        mlp_user_emb = self.mlp_embedding_user(user)
        mlp_item_emb = self.mlp_embedding_item(item)
        mlp_vector = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        pred = self.predict(predict_vector)
        return pred.view(-1)


def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:  # 使用in操作符替换has_key
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


if __name__ == '__main__':
    result_data = pd.DataFrame(columns=['Model', '# Factor', 'Iteration', 'Train Loss', 'HR@10', 'NDCG@10'])
    num_negatives = 4
    learner = 'adam'
    learning_rate = 0.001
    epochs = 5
    batch_size = 2048
    verbose = 1
    topK = 10
    evaluation_threads = 1

    # Loading data
    t1 = time()
    dataset = Dataset('./Data/ml-1m')
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # 回到这里
    for num_factors in [[8],[16, 8], [32, 16, 8], [64, 32, 16, 8], [128, 64, 32, 16, 8]]:
        for m in [ 'MLP']:
            if m == 'GMF':
                if num_factors == [8]:
                    model = GMF(num_users, num_items, 3)
                else:
                    model = None
            elif m == 'MLP':
                model = MLP(num_users, num_items, num_factors)
            else:
                model = NeuMF(num_users, num_items, num_factors, 3)
            if model is not None:
                model = model.to(device)
                if learner.lower() == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

                # Generate training instances
                user_input, item_input, labels = get_train_instances(train, num_negatives)
                train_dataset = TensorDataset(torch.tensor(user_input), torch.tensor(item_input), torch.tensor(labels))
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

                # Train model
                best_hr, best_ndcg, best_iter = 0, 0, -1
                for epoch in range(epochs):
                    t1 = time()
                    # Training
                    model.train()
                    total_loss = 0
                    for user_indices, item_indices, labels in train_loader:
                        user_indices = user_indices.to(device)
                        item_indices = item_indices.to(device)
                        labels = labels.to(device)
                        predictions = model(user_indices, item_indices)
                        loss = nn.BCEWithLogitsLoss()(predictions, labels.float())
                        total_loss += loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Evaluation
                    if epoch % verbose == 0:
                        model.eval()
                        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
                        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                        result = {
                            'Model': m, '# mlp layer': len(num_factors), 'Iteration': epoch + 1,
                            'Train Loss': total_loss, 'HR@10': hr, 'NDCG@10': ndcg
                        }
                        print(result, time() - t1)
                        result_df = pd.DataFrame([result])
                        result_data = pd.concat([result_data, result_df], ignore_index=True)
    result_data.to_csv('result_layer' + '.csv', index=False)
