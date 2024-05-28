import numpy as np
import torch
import math

def evaluate(model, test_loader, negative_loader, top_k=10):
    model.eval()
    HR_list = []
    NDCG_list = []

    for (user_ids, pos_item_ids, _), (neg_user_ids, _, neg_item_ids) in zip(test_loader, negative_loader):
        items = torch.cat([pos_item_ids.unsqueeze(1), neg_item_ids], dim=1)
        user_ids = user_ids.unsqueeze(1).expand(-1, items.size(1))

        user_ids = user_ids.reshape(-1)
        item_ids = items.reshape(-1)
        predictions = model(user_ids, item_ids).squeeze()
        predictions = predictions.reshape(-1, 100)
        _, indices = torch.topk(predictions, k=top_k, dim=1)
        recommended_items = items.gather(1, indices)

        HR = (recommended_items == pos_item_ids.unsqueeze(1)).any(dim=1).float()
        HR_list.append(HR.mean().item())

        relevant = (recommended_items == pos_item_ids.unsqueeze(1))
        rank = relevant.nonzero(as_tuple=True)[1]
        NDCG = (1 / torch.log2(rank.float() + 2)).mean().item()
        NDCG_list.append(NDCG)

    mean_HR = np.mean(HR_list)
    mean_NDCG = np.mean(NDCG_list)
    
    return mean_HR, mean_NDCG