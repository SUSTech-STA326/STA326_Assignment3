import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0")

def train(model, train_loader, optimizer, criterion):  # 修改函数名
    model.train()
    total_loss = 0
    for u_ids, i_ids, ratings in train_loader:  # 修改变量名
        # Move data to the appropriate device if using CUDA
        u_ids = u_ids.to(device)  # 修改变量名
        i_ids = i_ids.to(device)  # 修改变量名
        ratings = ratings.float().to(device)

        optimizer.zero_grad()
        outputs = model(u_ids, i_ids)  # ratings are not passed to the model
        loss = criterion(outputs.squeeze(), ratings.float())  # Ensure ratings are float for loss calculation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, negative_loader, top_k=10):  # 修改函数名
    model.eval()
    HR_list = []
    NDCG_list = []

    with torch.no_grad():
        for (user_ids, pos_item_ids, _), (neg_user_ids, _, neg_item_ids) in zip(test_loader, negative_loader):
            # Combine positive and negative items for scoring
            items = torch.cat([pos_item_ids.unsqueeze(1), neg_item_ids], dim=1)
            user_ids = user_ids.unsqueeze(1).expand(-1, items.size(1))  # Expand user_ids to match the number of items
            
            # Move data to the correct device
            items = items.to(device)
            user_ids = user_ids.to(device)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)

            # Predict the scores for these items
            predictions = model(user_ids.reshape(-1), items.reshape(-1)).reshape(-1, items.size(1))

            # Get the index of the highest scored items
            _, indices = torch.topk(predictions, k=top_k, dim=1)
            recommended_items = items.gather(1, indices)

            # Check if the positive test item is among the recommended items
            HR = (recommended_items == pos_item_ids.unsqueeze(1)).any(dim=1).float()
            HR_list.append(HR.mean().item())

            # Compute NDCG
            relevant = (recommended_items == pos_item_ids.unsqueeze(1))
            rank = relevant.nonzero(as_tuple=True)[1]
            NDCG = (1 / torch.log2(rank.float() + 2)).mean().item()  # Compute NDCG score
            NDCG_list.append(NDCG)

    # Compute the average HR and NDCG
    mean_HR = np.mean(HR_list)
    mean_NDCG = np.mean(NDCG_list)

    return mean_HR, mean_NDCG
