import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import evaluate
import model
import data_utils

####### Data Pre #######

ratings = pd.read_csv('./Dataset/ml-1m/ratings.dat', sep='::', engine='python', names=['user_id', 'item_id', 'rating', 'timestamp'])
data = data_utils.NCF_Data(ratings)
train_loader =data.get_train_instance()
test_loader =data.get_test_instance()

####### Start #######

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 30
top_k = 10
num_users = ratings['user_id'].nunique()+1
num_items = ratings['item_id'].nunique()+1

# gmf_model = model.Generalized_Matrix_Factorization(num_users, num_items).to(device)
# mlp_model = model.Multi_Layer_Perceptron(num_users, num_items).to(device)
neumf_model = model.NeuMF(num_users, num_items).to(device)

loss_function = nn.BCELoss()
# optimizer_gmf = torch.optim.Adam(gmf_model.parameters(), lr=0.001)
# optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
optimizer_neumf = torch.optim.Adam(neumf_model.parameters(), lr=0.001)

# HRs_gmf = []
# NDCGs_gmf = []
# HRs_mlp = []
# NDCGs_mlp = []
HRs_neumf = []
NDCGs_neumf = []


for epoch in range(1, epochs + 1):
#     gmf_model.train()
#     mlp_model.train()
    neumf_model.train()

    start_time = time.time()

    for user, item, label in train_loader:
        user = user.to(device)
        item = item.to(device)
        label = label.to(device)

#         optimizer_gmf.zero_grad()
#         gmf_prediction = gmf_model(user, item).squeeze()
#         gmf_loss = loss_function(gmf_prediction, label)
#         gmf_loss.backward()
#         optimizer_gmf.step()

#         optimizer_mlp.zero_grad()
#         mlp_prediction = mlp_model(user, item).squeeze()
#         mlp_loss = loss_function(mlp_prediction, label)
#         mlp_loss.backward()
#         optimizer_mlp.step()

        optimizer_neumf.zero_grad()
        neumf_prediction = neumf_model(user, item).squeeze()
        neumf_loss = loss_function(neumf_prediction, label)
        neumf_loss.backward()
        optimizer_neumf.step()

#     gmf_model.eval()
#     mlp_model.eval()
    neumf_model.eval()

#     HR_gmf, NDCG_gmf = evaluate.metrics(gmf_model, test_loader, top_k, device)
#     HR_mlp, NDCG_mlp = evaluate.metrics(mlp_model, test_loader, top_k, device)
    HR_neumf, NDCG_neumf = evaluate.metrics(neumf_model, test_loader, top_k, device)

    HRs_neumf.append(HR_neumf)
    NDCGs_neumf.append(NDCG_neumf)

    elapsed_time = time.time() - start_time
    print("Epoch: {} | Time: {:.2f}s".format(epoch, elapsed_time))
#     print("GMF | HR: {:.4f} | NDCG: {:.4f}".format(np.mean(HR_gmf), np.mean(NDCG_gmf)))
#     print("MLP | HR: {:.4f} | NDCG: {:.4f}".format(np.mean(HR_mlp), np.mean(NDCG_mlp)))
    print("NeuMF | HR: {:.4f} | NDCG: {:.4f}".format(np.mean(HR_neumf), np.mean(NDCG_neumf)))

print("Training finished.")

# os.makedirs('./results_gmf', exist_ok=True)
# os.makedirs('results_mlp', exist_ok=True)
os.makedirs('results_neumf', exist_ok=True)
np.save('./results_neumf/HRs_neumf.npy', HRs_neumf)
np.save('./results_neumf/NDCGs_neumf.npy', NDCGs_neumf)