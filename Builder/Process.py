import torch
import numpy as np
import matplotlib.pyplot as plt


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for user_ids, item_ids, ratings in train_loader:
        # Move data to the appropriate device if using CUDA
        user_ids = user_ids.to(model.device)
        item_ids = item_ids.to(model.device)
        ratings = ratings.float().to(model.device)

        optimizer.zero_grad()
        outputs = model(user_ids, item_ids)  # ratings are not passed to the model
        loss = criterion(outputs.squeeze(), ratings.float())  # Ensure ratings are float for loss calculation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, test_loader, negative_loader, top_k=10):
    model.eval()
    HR_list = []
    NDCG_list = []

    # Iterate over test_loader and negative_loader simultaneously
    for (user_ids, pos_item_ids, _), (neg_user_ids, _, neg_item_ids) in zip(test_loader, negative_loader):
        # Assume user_ids, pos_item_ids, neg_user_ids, neg_item_ids are all tensors of the appropriate shape
        # neg_item_ids是一个batchsize*99维的tensor
        # print(neg_item_ids.shape)#256*99
        # print(pos_item_ids.shape)#256

        # print(pos_item_ids.unsqueeze(1).shape)#256*1
        # Combine positive and negative items for scoring
        items = torch.cat([pos_item_ids.unsqueeze(1), neg_item_ids], dim=1)
        user_ids = user_ids.unsqueeze(1).expand(-1, items.size(1))  # Expand user_ids to match the number of items
        # 在上一个代码执行后，user_ids的维度变为了256*100，每一行都是user_id的复制
        # Move data to the correct device
        items = items.to(model.device)
        user_ids = user_ids.to(model.device)

        # Predict the scores for these items
        # print("user_ids shape is:",user_ids.shape)#user_ids shape is: torch.Size([256, 100])
        # print("items shape is:",items.shape)#user_ids shape is: torch.Size([256, 100])
        user_ids = user_ids.reshape(-1)
        item_ids = items.reshape(-1)
        # print("user_ids shape is:",user_ids.shape)#user_ids shape is: torch.Size([25600])
        # print("items shape is:",item_ids.shape)#user_ids shape is: torch.Size([25600])
        predictions = model(user_ids, item_ids).squeeze()
        # print("predictions shape is:",predictions.shape)#predictions shape is: torch.Size([25600])
        # Get the index of the highest scored items
        predictions = predictions.reshape(-1, 100)  # 重新变回256*100
        # predictions中每一行的第一个probability是正样本的概率，后面的是负样本的概率
        # print("predictions shape after reshape again is:",predictions.shape)
        # predictions shape is: torch.Size([256, 100]
        _, indices = torch.topk(predictions, k=top_k, dim=1)
        # print(indices.shape)#256*10
        recommended_items = items.gather(1, indices)  # Gather the top-k recommended item_ids
        # print("recommended_items shape is:",recommended_items.shape)#recommended_items shape is: torch.Size([256, 10]
        # print("pos_item_ids shape is:",pos_item_ids.shape)#pos_item_ids shape is: torch.Size([256])
        # Check if the positive test item is among the recommended items
        HR = (recommended_items == pos_item_ids.unsqueeze(1)).any(dim=1).float()
        HR_list.append(HR.mean().item())

        # Compute NDCG
        relevant = (recommended_items == pos_item_ids.unsqueeze(1))
        rank = relevant.nonzero(as_tuple=True)[1]  # Get rank positions of relevant items
        # 这行代码找出了relevant张量中所有非零（即True）元素的索引。nonzero(as_tuple=True)函数返回一个元组，其中包含了所有非零元素的索引。由于relevant是一个二维张量，所以返回的元组包含两个元素，分别是非零元素的行索引和列索引。[1]操作取出了所有非零元素的列索引，即在推荐列表中的排名
        NDCG = (1 / torch.log2(rank.float() + 2)).mean().item()  # Compute NDCG score
        NDCG_list.append(NDCG)

    # Compute the average HR and NDCG
    mean_HR = np.mean(HR_list)
    mean_NDCG = np.mean(NDCG_list)

    return mean_HR, mean_NDCG


def draw(results, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    titles = ['HR@10 Across Epochs', 'NDCG@10 Across Epochs']
    models = ['MLP-4', 'NeuMF', 'GMF']
    colors = {'MLP-4': '#1939B7', 'NeuMF': '#4F7C03', 'GMF': '#DB8A2C'}
    metrics = ['HR@10', 'NDCG@10']

    for i, metric in enumerate(metrics):
        for model in models:
            axes[i].plot(results[model][metric], label=model, color=colors[model], linewidth=2)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(path)
    plt.show()
